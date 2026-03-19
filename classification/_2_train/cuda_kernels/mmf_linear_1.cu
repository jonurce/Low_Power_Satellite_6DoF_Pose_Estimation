#include <torch/extension.h>

#define TILE 16  // tile size for shared memory -> could be 32

// Warp shuffle reduction: sums val across all 32 threads in a warp
__device__ float warp_reduce_sum(float val) {
    // Each iteration halves the active threads, shuffling values down
    // offset=16: thread i gets value from thread i+16
    // offset=8:  thread i gets value from thread i+8
    // ... until offset=1
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;  // only thread 0 of each warp holds the full sum
}

// Grid mapping:
// blockIdx.x → tile of output features (o)
// blockIdx.y → tile of batch (b)
// threadIdx.x → lane within warp (0-31), used for coalesced memory access
// threadIdx.y → which output feature within the tile (0-TILE-1)
//
// Each warp (threadIdx.y = constant, threadIdx.x = 0..31) collectively
// computes the dot product for one (b, o) output element,
// iterating over in_features in chunks of 32 (warp width)
__global__ void mmf_linear_kernel(
    const float* __restrict__ x,       // [B, in_features]
    const float* __restrict__ weight,  // [out_features, in_features]
    const float* __restrict__ bias,    // [out_features]
    float* __restrict__ out,           // [B, out_features]
    float scale,
    int B,
    int in_features,
    int out_features
) {
    // Shared memory tiles for x and w
    // Each block loads TILE rows of x and TILE rows of w at a time
    __shared__ float x_tile[TILE][TILE];       // [TILE batch rows, TILE input features]
    __shared__ float w_tile[TILE][TILE];       // [TILE output channels, TILE input features]

    // Which output feature and batch element does this thread own?
    // threadIdx.x → batch dimension  (coalesced memory access)
    // threadIdx.y → output feature dimension
    int o = blockIdx.x * TILE + threadIdx.y;   // output feature index
    int b = blockIdx.y * TILE + threadIdx.x;   // batch index — coalesced: adjacent threads = adjacent b

    float acc = 0.0f;

    // Iterate over tiles of in_features
    for (int t = 0; t < (in_features + TILE - 1) / TILE; t++) {
        int feat_idx = t * TILE + threadIdx.x;  // which input feature this thread loads

        // --- Coalesced load into shared memory ---
        // threadIdx.x varies across the warp → consecutive threads load consecutive
        // memory addresses → single memory transaction per warp (coalesced)

        // Load x tile: x[b, feat_idx]
        x_tile[threadIdx.y][threadIdx.x] =
            (b < B && feat_idx < in_features)
            ? x[b * in_features + feat_idx]
            : 0.0f;

        // Load weight tile: weight[o, feat_idx]
        w_tile[threadIdx.y][threadIdx.x] =
            (o < out_features && feat_idx < in_features)
            ? weight[o * in_features + feat_idx]
            : 0.0f;

        // Sync: ensure all threads have finished loading before computing
        // Without this, some threads might start reading tiles that other threads haven't finished writing yet.
        __syncthreads();

        // --- Compute partial dot product from this tile ---
        // Each thread accumulates TILE multiply-free additions
        // Both x_tile and w_tile are in fast shared memory (~5 cycle latency)
        for (int k = 0; k < TILE; k++) {
            float w = w_tile[threadIdx.y][k];
            float xi = x_tile[threadIdx.x][k];  // note: threadIdx.x indexes batch here

            // MMF: branch on ternary weight, no multiplication
            if      (w ==  1.0f) acc += xi;
            else if (w == -1.0f) acc -= xi;
        }

        // Sync: ensure all threads are done computing before next tile load (next loop iteration)
        __syncthreads();
    }

    // --- Warp shuffle reduction ---
    // At this point each thread holds a partial sum
    // Threads with the same threadIdx.y (same output feature) but different
    // threadIdx.x (different batch slices) need to be reduced
    // warp_reduce_sum sums across threadIdx.x (the warp lane dimension)
    acc = warp_reduce_sum(acc);

    // Only thread 0 of each warp writes the final result
    if (threadIdx.x == 0 && b < B && o < out_features) {
        out[b * out_features + o] = scale * acc + bias[o];
    }
}

torch::Tensor mmf_linear(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    float scale
) {
    int B            = x.size(0);
    int in_features  = x.size(1);
    int out_features = weight.size(0);

    auto out = torch::empty({B, out_features}, x.options());

    // 2D thread block: (TILE, TILE) = 256 threads
    // threadIdx.x → batch dimension  (coalesced memory access)
    // threadIdx.y → output feature dimension
    // Each thread is responsible for one (b, o) pair
    dim3 threads(TILE, TILE);


    dim3 blocks(
        (out_features + TILE - 1) / TILE,
        (B            + TILE - 1) / TILE
    );
    

    mmf_linear_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        scale,
        B,
        in_features,
        out_features
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mmf_linear", &mmf_linear, "MMF Linear forward optimized (CUDA)");
}