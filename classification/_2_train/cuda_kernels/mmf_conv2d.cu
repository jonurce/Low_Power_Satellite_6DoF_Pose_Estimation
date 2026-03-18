#include <torch/extension.h>

#define TILE 16

__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Grid mapping:
// blockIdx.x → tile of output spatial positions (h_out * W_out)
// blockIdx.y → tile of output channels (c_out)
// blockIdx.z → batch (b)
// threadIdx.x → lane within warp (0-31), iterates over flattened (C_in*kH*kW) tiles
// threadIdx.y → which output channel within the tile (0-TILE-1)
//
// Each warp collectively computes one (b, c_out, spatial) output element
// by iterating over flattened (C_in * kH * kW) in chunks of TILE
__global__ void mmf_conv2d_kernel(
    const float* __restrict__ x,       // [B, C_in, H_in, W_in]
    const float* __restrict__ weight,  // [C_out, C_in, kH, kW]
    const float* __restrict__ bias,    // [C_out]
    float* __restrict__ out,           // [B, C_out, H_out, W_out]
    float scale,
    int B,
    int C_in,
    int C_out,
    int H_in, int W_in,
    int H_out, int W_out,
    int kH, int kW,
    int stride,
    int padding
) {
    // Shared memory tiles
    // x_tile: patches of input [TILE spatial positions, TILE flattened (C_in*kH*kW)]
    // w_tile: patches of weight [TILE output channels,  TILE flattened (C_in*kH*kW)]
    __shared__ float x_tile[TILE][TILE];
    __shared__ float w_tile[TILE][TILE];

    int b           = blockIdx.z;                               // batch index
    int c_out_base  = blockIdx.y * TILE;                        // base output channel for this block
    int spatial_base = blockIdx.x * TILE;                       // base spatial index for this block

    int c_out        = c_out_base  + threadIdx.y;               // this thread's output channel
    int spatial_idx  = spatial_base + threadIdx.x;              // this thread's spatial position

    // Recover 2D output position from flat spatial index
    int h_out = (spatial_idx < H_out * W_out) ? spatial_idx / W_out : 0;
    int w_out = (spatial_idx < H_out * W_out) ? spatial_idx % W_out : 0;

    float acc = 0.0f;

    // Flatten (C_in, kH, kW) into a single dimension of length C_in*kH*kW
    int inner_size = C_in * kH * kW;
    int num_tiles  = (inner_size + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; t++) {
        int inner_idx = t * TILE + threadIdx.x;  // which (c_in, kh, kw) this thread loads

        // --- Load weight tile ---
        // w_tile[threadIdx.y][threadIdx.x]: output channel × flattened inner
        // threadIdx.x varies across warp → consecutive threads load consecutive
        // weight memory addresses → coalesced
        w_tile[threadIdx.y][threadIdx.x] =
            (c_out < C_out && inner_idx < inner_size)
            ? weight[c_out * inner_size + inner_idx]
            : 0.0f;

        // --- Load x tile ---
        // Decode flattened inner_idx → (c_in, kh, kw) to find input position
        // Each spatial position (h_out, w_out) maps to a different input patch
        // We load for the spatial position owned by threadIdx.y in this block
        int sp = spatial_base + threadIdx.y;  // spatial position for this row of x_tile
        int h_sp = (sp < H_out * W_out) ? sp / W_out : 0;
        int w_sp = (sp < H_out * W_out) ? sp % W_out : 0;

        float x_val = 0.0f;
        if (sp < H_out * W_out && inner_idx < inner_size) {
            // Decode flattened inner index back to (c_in, kh, kw)
            int c_in = inner_idx / (kH * kW);
            int rem  = inner_idx % (kH * kW);
            int kh   = rem / kW;
            int kw   = rem % kW;

            // Corresponding input position
            int h_in = h_sp * stride - padding + kh;
            int w_in = w_sp * stride - padding + kw;

            // Bounds check — implicit zero padding
            if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in)
                x_val = x[b * (C_in * H_in * W_in) +
                           c_in * (H_in * W_in) +
                           h_in * W_in +
                           w_in];
        }
        x_tile[threadIdx.y][threadIdx.x] = x_val;

        __syncthreads();

        // --- Compute partial dot product from this tile ---
        // Both tiles are now in shared memory (~5 cycle latency)
        for (int k = 0; k < TILE; k++) {
            float w  = w_tile[threadIdx.y][k];
            float xi = x_tile[threadIdx.x][k];  // threadIdx.x indexes spatial here

            // MMF: no multiplication
            if      (w ==  1.0f) acc += xi;
            else if (w == -1.0f) acc -= xi;
        }

        __syncthreads();
    }

    // --- Warp shuffle reduction ---
    // Threads with same threadIdx.y (same c_out) but different threadIdx.x
    // (different spatial positions) hold partial sums — reduce across warp
    acc = warp_reduce_sum(acc);

    // Only thread 0 of each warp writes the final result
    if (threadIdx.x == 0 && b < B && c_out < C_out && spatial_idx < H_out * W_out) {
        out[b     * (C_out * H_out * W_out) +
            c_out * (H_out * W_out) +
            h_out * W_out +
            w_out] = scale * acc + bias[c_out];
    }
}

torch::Tensor mmf_conv2d(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    float scale,
    int stride,
    int padding
) {
    int B    = x.size(0);
    int C_in = x.size(1);
    int H_in = x.size(2);
    int W_in = x.size(3);

    int C_out = weight.size(0);
    int kH    = weight.size(2);
    int kW    = weight.size(3);

    int H_out = (H_in + 2 * padding - kH) / stride + 1;
    int W_out = (W_in + 2 * padding - kW) / stride + 1;

    auto out = torch::empty({B, C_out, H_out, W_out}, x.options());

    // 3D grid:
    // x → spatial tiles  (H_out * W_out)
    // y → c_out tiles    (C_out)
    // z → batch          (B)
    dim3 threads(TILE, TILE);
    dim3 blocks(
        (H_out * W_out + TILE - 1) / TILE,
        (C_out         + TILE - 1) / TILE,
        B
    );

    mmf_conv2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        scale,
        B, C_in, C_out,
        H_in, W_in,
        H_out, W_out,
        kH, kW,
        stride, padding
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mmf_conv2d", &mmf_conv2d, "MMF Conv2d forward optimized (CUDA)");
}