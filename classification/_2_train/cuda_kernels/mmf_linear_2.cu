#include <torch/extension.h>

#define TILE 16

// ─────────────────────────────────────────────────────────────────
// DEVICE HELPERS
// ─────────────────────────────────────────────────────────────────

__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

// Block-wide reduction: reduces across all threads in block
// Uses shared memory scratch of size (blockDim.x / 32)
__device__ float block_reduce_sum(float val, float* smem_scratch) {
    int lane   = threadIdx.x % 32;
    int warp   = threadIdx.x / 32;
    val = warp_reduce_sum(val);
    if (lane == 0) smem_scratch[warp] = val;
    __syncthreads();
    val = (threadIdx.x < (blockDim.x / 32)) ? smem_scratch[threadIdx.x] : 0.0f;
    if (warp == 0) val = warp_reduce_sum(val);
    return val;
}

__device__ float block_reduce_max(float val, float* smem_scratch) {
    int lane   = threadIdx.x % 32;
    int warp   = threadIdx.x / 32;
    val = warp_reduce_max(val);
    if (lane == 0) smem_scratch[warp] = val;
    __syncthreads();
    val = (threadIdx.x < (blockDim.x / 32)) ? smem_scratch[threadIdx.x] : 0.0f;
    if (warp == 0) val = warp_reduce_max(val);
    return val;
}


// ─────────────────────────────────────────────────────────────────
// FORWARD KERNEL
// Fused: RMSNorm + activation_quant + weight_quant + MMF linear
//
// One block per output row m (one batch element)
// Phase 1 (reduction):  compute μ, σ², r, s_act for row m
// Phase 2 (tiled matmul): compute O[m, :] using quantized activations
//
// Each block uses blockDim.x = 256 threads in a 1D layout for Phase 1
// then logically remaps to 2D (TILE x TILE) tiles for Phase 2
//
// Grid: (M,) — one block per batch row
// ─────────────────────────────────────────────────────────────────
__global__ void mmf_linear_fused_forward_kernel(
    const float* __restrict__ X,       // [M, N]
    const float* __restrict__ W,       // [N, K]
    const float* __restrict__ b,       // [K]
    float* __restrict__ O,             // [M, K]
    float* __restrict__ Y_hat,         // [M, N]  save for backward
    float* __restrict__ W_tilde,       // [N, K]  save for backward (written by row 0 only)
    float* __restrict__ mu_out,        // [M]
    float* __restrict__ var_out,       // [M]
    float* __restrict__ r_out,         // [M]
    float  s_w,                        // weight quant scale: 1/mean(|W|), precomputed
    int M,
    int N,
    int K,
    float eps
) {
    int m = blockIdx.x;
    if (m >= M) return;

    const float* x_row = X + m * N;
    float*       o_row = O + m * K;
    float*       y_row = Y_hat + m * N;

    // ── Shared memory layout ──
    // We reuse shared memory across phases to save space
    // Phase 1 needs: scratch for block reductions [blockDim.x/32]
    // Phase 2 needs: y_tile [TILE][TILE], w_tile [TILE][TILE]
    // We allocate enough for both (phase 1 uses a small prefix)
    extern __shared__ float smem[];
    float* scratch = smem;                        // [blockDim.x/32] for reductions
    float* y_tile  = smem;                        // [TILE*TILE] reused in phase 2
    float* w_tile  = smem + TILE * TILE;          // [TILE*TILE] reused in phase 2

    // ── Phase 1a: compute mean and variance ──
    float sum = 0.0f, sum_sq = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float v  = x_row[i];
        sum     += v;
        sum_sq  += v * v;
    }
    sum    = block_reduce_sum(sum,    scratch);
    __syncthreads();
    sum_sq = block_reduce_sum(sum_sq, scratch);
    __syncthreads();

    __shared__ float row_mu, row_var, row_r;
    if (threadIdx.x == 0) {
        row_mu  = sum / N;
        row_var = sum_sq / N - row_mu * row_mu;
        row_r   = rsqrtf(row_var + eps);
        mu_out[m]  = row_mu;
        var_out[m] = row_var;
        r_out[m]   = row_r;
    }
    __syncthreads();

    // ── Phase 1b: find max|Y_norm| for activation_quant ──
    float local_max = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float y_norm = row_r * (x_row[i] - row_mu);
        local_max = fmaxf(local_max, fabsf(y_norm));
    }
    local_max = block_reduce_max(local_max, scratch);
    __syncthreads();

    __shared__ float row_s_act;
    if (threadIdx.x == 0)
        row_s_act = 127.0f / fmaxf(local_max, 1e-8f);
    __syncthreads();

    // ── Phase 1c: write Y_hat (quantized normalized activations) ──
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float y_norm = row_r * (x_row[i] - row_mu);
        float y_q    = rintf(row_s_act * y_norm);
        y_q          = fmaxf(-128.0f, fminf(127.0f, y_q));
        y_row[i]     = y_q / row_s_act;  // dequantized float32
    }
    __syncthreads();

    // ── Phase 2: tiled MMF linear ──
    // Thread layout remapped: thread t → (ty, tx) = (t/TILE, t%TILE)
    // Each (ty, tx) pair covers one (output_feature_in_tile, N_feature_in_tile) pair
    int tx = threadIdx.x % TILE;   // N dimension (inner)
    int ty = threadIdx.x / TILE;   // K dimension (output feature)

    // Iterate over K in tiles of TILE output features
    for (int k_base = 0; k_base < K; k_base += TILE) {
        float acc = 0.0f;

        // Iterate over N in tiles of TILE
        for (int n_base = 0; n_base < N; n_base += TILE) {
            int n_idx = n_base + tx;
            int k_idx = k_base + ty;

            // Load y_tile: Y_hat[m, n_idx]
            y_tile[ty * TILE + tx] =
                (n_idx < N) ? y_row[n_idx] : 0.0f;

            // Load and quantize w_tile: W[n_idx, k_idx]
            float w_raw = (n_idx < N && k_idx < K)
                          ? W[n_idx * K + k_idx]
                          : 0.0f;
            float w_q   = rintf(s_w * w_raw);
            w_q         = fmaxf(-1.0f, fminf(1.0f, w_q));
            float w_dq  = w_q / s_w;
            w_tile[ty * TILE + tx] = w_dq;

            // Save W_tilde — only batch row 0 writes, avoids redundant writes
            if (m == 0 && n_base + tx < N && k_base + ty < K)
                W_tilde[(n_base + tx) * K + (k_base + ty)] = w_dq;

            __syncthreads();

            // Compute partial dot product — MMF: no multiplication
            for (int i = 0; i < TILE; i++) {
                float w  = w_tile[ty * TILE + i];
                float yi = y_tile[ty * TILE + i]; // same ty row

                // Each thread handles its own (m, k) output
                // yi must come from the row tx is reading
                // Corrected indexing: y is indexed by [tx][i], w by [ty][i]
                yi = y_tile[tx * TILE + i];  // batch element tx, N feature i

                if      (w >  0.5f / s_w) acc += yi;
                else if (w < -0.5f / s_w) acc -= yi;
            }
            __syncthreads();
        }

        // Write output: thread (ty, tx) owns output element (m=m, k=k_base+ty)
        // but we need one acc per (m, k) — reduce across tx (N tiles)
        // Use warp shuffle: threads with same ty but different tx hold partial sums
        acc = warp_reduce_sum(acc);

        if (tx == 0 && k_base + ty < K)
            o_row[k_base + ty] = acc + b[k_base + ty];

        __syncthreads();
    }
}


// ─────────────────────────────────────────────────────────────────
// BACKWARD KERNEL 1: Linear backward
// Computes dY = dO × W_tilde^T  [M, N]
//          dW = Y_hat^T × dO    [N, K]
//          db = sum(dO, dim=0)  [K]
//
// Thread layout: same tiled structure as forward linear
// Grid: (ceil(N/TILE), ceil(M/TILE)) for dY
//       (ceil(N/TILE), ceil(K/TILE)) for dW  — computed in same kernel
// ─────────────────────────────────────────────────────────────────
__global__ void mmf_linear_backward_linear_kernel(
    const float* __restrict__ dO,       // [M, K]
    const float* __restrict__ W_tilde,  // [N, K]
    const float* __restrict__ Y_hat,    // [M, N]
    float* __restrict__ dY,             // [M, N]
    float* __restrict__ dW,             // [N, K]
    float* __restrict__ db,             // [K]
    int M, int N, int K
) {
    // ── dY = dO × W_tilde^T ──
    // Each thread computes one element of dY[m, n]
    // dY[m, n] = sum_k dO[m, k] * W_tilde[n, k]
    __shared__ float dO_tile[TILE][TILE];   // [M tile, K tile]
    __shared__ float wt_tile[TILE][TILE];   // [N tile, K tile]

    int n = blockIdx.x * TILE + threadIdx.y;  // N dimension
    int m = blockIdx.y * TILE + threadIdx.x;  // M dimension — coalesced

    float acc_dy = 0.0f;
    float acc_dw = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int k_idx = t * TILE + threadIdx.x;

        // Load dO tile: dO[m_row, k_idx]
        int m_row = blockIdx.y * TILE + threadIdx.y;
        dO_tile[threadIdx.y][threadIdx.x] =
            (m_row < M && k_idx < K) ? dO[m_row * K + k_idx] : 0.0f;

        // Load W_tilde tile: W_tilde[n_row, k_idx]
        int n_row = blockIdx.x * TILE + threadIdx.y;
        wt_tile[threadIdx.y][threadIdx.x] =
            (n_row < N && k_idx < K) ? W_tilde[n_row * K + k_idx] : 0.0f;

        __syncthreads();

        // dY[m, n] += dO[m, k] * W_tilde[n, k]  for k in tile
        // Note: W_tilde values are in {-1/s, 0, +1/s} — still use multiply here
        // (backward pass does not need to be MMF)
        for (int k = 0; k < TILE; k++) {
            acc_dy += dO_tile[threadIdx.x][k] * wt_tile[threadIdx.y][k];
        }

        __syncthreads();
    }

    if (m < M && n < N)
        dY[m * N + n] = acc_dy;

    // ── dW = Y_hat^T × dO ──
    // dW[n, k] = sum_m Y_hat[m, n] * dO[m, k]
    // Reuse the same block to compute a tile of dW
    __shared__ float yh_tile[TILE][TILE];   // [M tile, N tile]
    __shared__ float do_tile[TILE][TILE];   // [M tile, K tile]

    int n2 = blockIdx.x * TILE + threadIdx.y;
    int k2 = blockIdx.y * TILE + threadIdx.x;  // repurpose blockIdx.y for K

    float acc_dw2 = 0.0f;

    for (int t = 0; t < (M + TILE - 1) / TILE; t++) {
        int m_idx = t * TILE + threadIdx.x;

        int m_row2 = t * TILE + threadIdx.y;
        yh_tile[threadIdx.y][threadIdx.x] =
            (m_row2 < M && n2 < N) ? Y_hat[m_row2 * N + n2] : 0.0f;

        int m_row3 = t * TILE + threadIdx.y;
        do_tile[threadIdx.y][threadIdx.x] =
            (m_row3 < M && k2 < K) ? dO[m_row3 * K + k2] : 0.0f;

        __syncthreads();

        for (int mi = 0; mi < TILE; mi++)
            acc_dw2 += yh_tile[mi][threadIdx.y] * do_tile[mi][threadIdx.x];

        __syncthreads();
    }

    if (n2 < N && k2 < K)
        atomicAdd(&dW[n2 * K + k2], acc_dw2);

    // ── db = sum(dO, dim=0) ──
    if (m < M && n < N && threadIdx.y == 0) {
        for (int k = 0; k < K; k++)
            atomicAdd(&db[k], dO[m * K + k]);
    }
}


// ─────────────────────────────────────────────────────────────────
// BACKWARD KERNEL 2: rms_norm_bwd
// Computes dX from dY, X, μ, σ², r (per paper's formula)
//
// Two-pass reduction per row:
// Pass 1: compute dσ² and dμ (require full sums over N)
// Pass 2: compute dX elementwise
//
// Grid: (M,) — one block per batch row
// ─────────────────────────────────────────────────────────────────
__global__ void mmf_linear_backward_rmsnorm_kernel(
    const float* __restrict__ dY,   // [M, N]
    const float* __restrict__ X,    // [M, N]
    const float* __restrict__ mu,   // [M]
    const float* __restrict__ var_, // [M]
    const float* __restrict__ r,    // [M]
    float* __restrict__ dX,         // [M, N]
    int M, int N
) {
    int m = blockIdx.x;
    if (m >= M) return;

    const float* dy_row = dY + m * N;
    const float* x_row  = X  + m * N;
    float*       dx_row = dX + m * N;

    float row_mu  = mu[m];
    float row_r   = r[m];
    float row_r3  = row_r * row_r * row_r;

    extern __shared__ float smem[];
    float* scratch = smem;  // [blockDim.x/32]

    // ── Pass 1a: compute dσ² = sum(dY × (X - μ)) × -0.5 × r³ ──
    float sum_dsigma = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        sum_dsigma += dy_row[i] * (x_row[i] - row_mu);
    }
    sum_dsigma = block_reduce_sum(sum_dsigma, scratch);
    __syncthreads();

    __shared__ float row_dsigma2, row_dmu;
    if (threadIdx.x == 0) {
        row_dsigma2 = sum_dsigma * (-0.5f) * row_r3;
    }
    __syncthreads();

    // ── Pass 1b: compute dμ = sum(-r × dY) + dσ² × mean(X - μ) × 2 ──
    float sum_dmu_a = 0.0f, sum_xmu = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        sum_dmu_a += -row_r * dy_row[i];
        sum_xmu   += (x_row[i] - row_mu);
    }
    sum_dmu_a = block_reduce_sum(sum_dmu_a, scratch);
    __syncthreads();
    sum_xmu   = block_reduce_sum(sum_xmu,   scratch);
    __syncthreads();

    if (threadIdx.x == 0)
        row_dmu = sum_dmu_a + row_dsigma2 * (sum_xmu / N) * 2.0f;
    __syncthreads();

    // ── Pass 2: compute dX = r × dY + 2dσ²(X - μ)/N + dμ/N ──
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        dx_row[i] = row_r * dy_row[i]
                  + 2.0f * row_dsigma2 * (x_row[i] - row_mu) / N
                  + row_dmu / N;
    }
}


// ─────────────────────────────────────────────────────────────────
// C++ ENTRY POINTS
// ─────────────────────────────────────────────────────────────────
std::vector<torch::Tensor> mmf_linear_forward(
    torch::Tensor X,   // [M, N]
    torch::Tensor W,   // [N, K]
    torch::Tensor b,   // [K]
    float eps
) {
    int M = X.size(0), N = X.size(1), K = W.size(1);

    auto opts    = X.options();
    auto O       = torch::empty({M, K}, opts);
    auto Y_hat   = torch::empty({M, N}, opts);
    auto W_tilde = torch::empty({N, K}, opts);
    auto mu      = torch::empty({M},    opts);
    auto var_    = torch::empty({M},    opts);
    auto r       = torch::empty({M},    opts);

    // Weight quant scale: precomputed on CPU (single scalar, negligible cost)
    float s_w = 1.0f / W.abs().mean().item<float>();

    // Shared memory: max(blockDim.x/32, 2*TILE*TILE) floats
    int threads     = 256;
    size_t smem     = sizeof(float) * (2 * TILE * TILE);

    mmf_linear_fused_forward_kernel<<<M, threads, smem>>>(
        X.data_ptr<float>(),
        W.data_ptr<float>(),
        b.data_ptr<float>(),
        O.data_ptr<float>(),
        Y_hat.data_ptr<float>(),
        W_tilde.data_ptr<float>(),
        mu.data_ptr<float>(),
        var_.data_ptr<float>(),
        r.data_ptr<float>(),
        s_w,
        M,
        N,
        K,
        eps
    );

    return {O, Y_hat, W_tilde, mu, var_, r};
}

std::vector<torch::Tensor> mmf_linear_backward(
    torch::Tensor dO,      // [M, K]
    torch::Tensor W_tilde, // [N, K]
    torch::Tensor Y_hat,   // [M, N]
    torch::Tensor X,       // [M, N]
    torch::Tensor mu,      // [M]
    torch::Tensor var_,    // [M]
    torch::Tensor r        // [M]
) {
    int M = dO.size(0), K = dO.size(1);
    int N = X.size(1);

    auto opts = dO.options();
    auto dY   = torch::empty({M, N}, opts);
    auto dW   = torch::zeros({N, K}, opts);  // zeros: atomicAdd accumulates into it
    auto db   = torch::zeros({K},    opts);
    auto dX   = torch::empty({M, N}, opts);

    // ── Backward kernel 1: linear backward (dY, dW, db) ──
    dim3 threads1(TILE, TILE);
    dim3 blocks1(
        (N + TILE - 1) / TILE,
        (M + TILE - 1) / TILE
    );
    mmf_linear_backward_linear_kernel<<<blocks1, threads1>>>(
        dO.data_ptr<float>(),
        W_tilde.data_ptr<float>(),
        Y_hat.data_ptr<float>(),
        dY.data_ptr<float>(),
        dW.data_ptr<float>(),
        db.data_ptr<float>(),
        M,
        N,
        K
    );

    // ── Backward kernel 2: rms_norm_bwd (dX) ──
    int threads2  = 256;
    size_t smem2  = sizeof(float) * (threads2 / 32);
    mmf_linear_backward_rmsnorm_kernel<<<M, threads2, smem2>>>(
        dY.data_ptr<float>(),
        X.data_ptr<float>(),
        mu.data_ptr<float>(),
        var_.data_ptr<float>(),
        r.data_ptr<float>(),
        dX.data_ptr<float>(),
        M,
        N
    );

    return {dX, dW, db};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mmf_linear_forward",  &mmf_linear_forward,  "MMF Linear fused forward (CUDA)");
    m.def("mmf_linear_backward", &mmf_linear_backward, "MMF Linear fused backward (CUDA)");
}