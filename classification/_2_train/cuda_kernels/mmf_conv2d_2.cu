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

__device__ float block_reduce_sum(float val, float* smem_scratch) {
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    val = warp_reduce_sum(val);
    if (lane == 0) smem_scratch[warp] = val;
    __syncthreads();
    val = (threadIdx.x < (blockDim.x / 32)) ? smem_scratch[threadIdx.x] : 0.0f;
    if (warp == 0) val = warp_reduce_sum(val);
    return val;
}

__device__ float block_reduce_max(float val, float* smem_scratch) {
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    val = warp_reduce_max(val);
    if (lane == 0) smem_scratch[warp] = val;
    __syncthreads();
    val = (threadIdx.x < (blockDim.x / 32)) ? smem_scratch[threadIdx.x] : 0.0f;
    if (warp == 0) val = warp_reduce_max(val);
    return val;
}


// ─────────────────────────────────────────────────────────────────
// FORWARD KERNEL
// Fused: RMSNorm (over C_in) + activation_quant + weight_quant + MMF conv
//
// Grid: (B * H_out * W_out,) — one block per output spatial position per batch
//
// Each block:
//   Phase 1: normalize the C_in input vector at this (b, h_out, w_out) position
//   Phase 2: for each output channel c_out, accumulate over (C_in, kH, kW)
//
// Saved for backward:
//   Y_hat  [B, C_in, H_in, W_in]  — quantized normalized input (same shape as X)
//   W_tilde[C_out, C_in, kH, kW]  — quantized weights
//   mu     [B, H_out, W_out]       — mean per spatial position
//   var_   [B, H_out, W_out]       — variance per spatial position
//   r      [B, H_out, W_out]       — RMS scale per spatial position
// ─────────────────────────────────────────────────────────────────
__global__ void mmf_conv2d_fused_forward_kernel(
    const float* __restrict__ X,        // [B, C_in, H_in, W_in]
    const float* __restrict__ W,        // [C_out, C_in, kH, kW]
    const float* __restrict__ b,        // [C_out]
    float* __restrict__ O,              // [B, C_out, H_out, W_out]
    float* __restrict__ Y_hat,          // [B, C_in, H_in, W_in]
    float* __restrict__ W_tilde,        // [C_out, C_in, kH, kW]
    float* __restrict__ mu_out,         // [B, H_out, W_out]
    float* __restrict__ var_out,        // [B, H_out, W_out]
    float* __restrict__ r_out,          // [B, H_out, W_out]
    float  s_w,
    int B, int C_in, int C_out,
    int H_in, int W_in,
    int H_out, int W_out,
    int kH, int kW,
    int stride, int padding,
    float eps
) {
    // Each block handles one (b, h_out, w_out) output position
    int spatial_idx = blockIdx.x;
    int total_spatial = B * H_out * W_out;
    if (spatial_idx >= total_spatial) return;

    // Decode flat index → (b, h_out, w_out)
    int b_idx  = spatial_idx / (H_out * W_out);
    int hw_idx = spatial_idx % (H_out * W_out);
    int h_out  = hw_idx / W_out;
    int w_out  = hw_idx % W_out;

    // Shared memory: scratch for reductions + row_mu/row_r/row_s_act
    extern __shared__ float smem[];
    float* scratch = smem;   // [blockDim.x / 32] for block reductions

    // ── Phase 1a: RMSNorm over C_in at this spatial position ──
    // For each c_in, the input value is X[b, c_in, h_out*stride - padding + ?, ...]
    // RMSNorm normalizes over C_in — we use the center of the receptive field
    // (h_in = h_out*stride, w_in = w_out*stride) as the normalization anchor,
    // consistent with treating each spatial position as an independent token
    float sum = 0.0f, sum_sq = 0.0f;
    for (int c = threadIdx.x; c < C_in; c += blockDim.x) {
        int h_in_center = h_out * stride - padding + kH / 2;
        int w_in_center = w_out * stride - padding + kW / 2;

        // Clamp to valid input range (zero padding)
        float v = 0.0f;
        if (h_in_center >= 0 && h_in_center < H_in &&
            w_in_center >= 0 && w_in_center < W_in) {
            v = X[b_idx * (C_in * H_in * W_in) +
                  c     * (H_in * W_in) +
                  h_in_center * W_in +
                  w_in_center];
        }
        sum    += v;
        sum_sq += v * v;
    }
    sum    = block_reduce_sum(sum,    scratch);
    __syncthreads();
    sum_sq = block_reduce_sum(sum_sq, scratch);
    __syncthreads();

    __shared__ float row_mu, row_var, row_r;
    if (threadIdx.x == 0) {
        row_mu  = sum / C_in;
        row_var = sum_sq / C_in - row_mu * row_mu;
        row_r   = rsqrtf(row_var + eps);

        // Save to output tensors
        int idx = b_idx * (H_out * W_out) + hw_idx;
        mu_out[idx]  = row_mu;
        var_out[idx] = row_var;
        r_out[idx]   = row_r;
    }
    __syncthreads();

    // ── Phase 1b: activation_quant — find max|Y_norm| over C_in ──
    float local_max = 0.0f;
    for (int c = threadIdx.x; c < C_in; c += blockDim.x) {
        int h_in_center = h_out * stride - padding + kH / 2;
        int w_in_center = w_out * stride - padding + kW / 2;
        float v = 0.0f;
        if (h_in_center >= 0 && h_in_center < H_in &&
            w_in_center >= 0 && w_in_center < W_in) {
            v = X[b_idx * (C_in * H_in * W_in) +
                  c     * (H_in * W_in) +
                  h_in_center * W_in +
                  w_in_center];
        }
        float y_norm = row_r * (v - row_mu);
        local_max = fmaxf(local_max, fabsf(y_norm));
    }
    local_max = block_reduce_max(local_max, scratch);
    __syncthreads();

    __shared__ float row_s_act;
    if (threadIdx.x == 0)
        row_s_act = 127.0f / fmaxf(local_max, 1e-8f);
    __syncthreads();

    // ── Phase 1c: write Y_hat (quantized normalized activations) ──
    // Y_hat has same shape as X — we normalize and quantize the full
    // spatial neighborhood needed for convolution at this position
    for (int c = threadIdx.x; c < C_in; c += blockDim.x) {
        for (int kh = 0; kh < kH; kh++) {
            for (int kw_i = 0; kw_i < kW; kw_i++) {
                int h_in = h_out * stride - padding + kh;
                int w_in = w_out * stride - padding + kw_i;

                float v = 0.0f;
                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    v = X[b_idx * (C_in * H_in * W_in) +
                          c     * (H_in * W_in) +
                          h_in  * W_in +
                          w_in];
                }

                float y_norm = row_r * (v - row_mu);
                float y_q    = rintf(row_s_act * y_norm);
                y_q          = fmaxf(-128.0f, fminf(127.0f, y_q));
                float y_dq   = y_q / row_s_act;

                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    Y_hat[b_idx * (C_in * H_in * W_in) +
                          c     * (H_in * W_in) +
                          h_in  * W_in +
                          w_in] = y_dq;
                }
            }
        }
    }
    __syncthreads();

    // ── Phase 2: MMF convolution over (C_out) for this spatial position ──
    // Each thread computes partial sum for one or more output channels
    // Inner loop: flattened (C_in * kH * kW)
    int inner_size = C_in * kH * kW;

    for (int c_out = threadIdx.x; c_out < C_out; c_out += blockDim.x) {
        float acc = 0.0f;

        for (int inner = 0; inner < inner_size; inner++) {
            // Decode inner → (c_in, kh, kw)
            int c_in   = inner / (kH * kW);
            int rem    = inner % (kH * kW);
            int kh     = rem / kW;
            int kw_i   = rem % kW;

            int h_in = h_out * stride - padding + kh;
            int w_in = w_out * stride - padding + kw_i;

            // Get quantized activation
            float yi = 0.0f;
            if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                yi = Y_hat[b_idx * (C_in * H_in * W_in) +
                           c_in  * (H_in * W_in) +
                           h_in  * W_in +
                           w_in];
            }

            // Get and quantize weight on-chip
            float w_raw = W[c_out  * (C_in * kH * kW) +
                            c_in   * (kH * kW) +
                            kh     * kW +
                            kw_i];
            float w_q   = rintf(s_w * w_raw);
            w_q         = fmaxf(-1.0f, fminf(1.0f, w_q));
            float w_dq  = w_q / s_w;

            // Save W_tilde — only first spatial block writes
            if (spatial_idx == 0) {
                W_tilde[c_out * (C_in * kH * kW) +
                        c_in  * (kH * kW) +
                        kh    * kW +
                        kw_i] = w_dq;
            }

            // MMF: no multiplication
            if      (w_q >  0.5f) acc += yi;
            else if (w_q < -0.5f) acc -= yi;
        }

        // Write output
        O[b_idx * (C_out * H_out * W_out) +
          c_out  * (H_out * W_out) +
          h_out  * W_out +
          w_out] = acc + b[c_out];
    }
}


// ─────────────────────────────────────────────────────────────────
// BACKWARD KERNEL 1: Conv linear backward
// dY_hat [B, C_in, H_in, W_in] — gradient w.r.t. quantized activations
// dW     [C_out, C_in, kH, kW] — gradient w.r.t. weights
// db     [C_out]                — gradient w.r.t. bias
//
// Grid: (B * H_out * W_out,) — one block per output spatial position
// ─────────────────────────────────────────────────────────────────
__global__ void mmf_conv2d_backward_conv_kernel(
    const float* __restrict__ dO,       // [B, C_out, H_out, W_out]
    const float* __restrict__ W_tilde,  // [C_out, C_in, kH, kW]
    const float* __restrict__ Y_hat,    // [B, C_in, H_in, W_in]
    float* __restrict__ dY_hat,         // [B, C_in, H_in, W_in]
    float* __restrict__ dW,             // [C_out, C_in, kH, kW]
    float* __restrict__ db,             // [C_out]
    int B, int C_in, int C_out,
    int H_in, int W_in,
    int H_out, int W_out,
    int kH, int kW,
    int stride, int padding
) {
    int spatial_idx  = blockIdx.x;
    int total_spatial = B * H_out * W_out;
    if (spatial_idx >= total_spatial) return;

    int b_idx  = spatial_idx / (H_out * W_out);
    int hw_idx = spatial_idx % (H_out * W_out);
    int h_out  = hw_idx / W_out;
    int w_out  = hw_idx % W_out;

    int inner_size = C_in * kH * kW;

    // Each thread handles one or more output channels
    for (int c_out = threadIdx.x; c_out < C_out; c_out += blockDim.x) {
        float do_val = dO[b_idx * (C_out * H_out * W_out) +
                          c_out  * (H_out * W_out) +
                          h_out  * W_out +
                          w_out];

        // db: sum dO over (B, H_out, W_out)
        atomicAdd(&db[c_out], do_val);

        // dY_hat and dW: loop over inner (C_in, kH, kW)
        for (int inner = 0; inner < inner_size; inner++) {
            int c_in  = inner / (kH * kW);
            int rem   = inner % (kH * kW);
            int kh    = rem / kW;
            int kw_i  = rem % kW;

            int h_in = h_out * stride - padding + kh;
            int w_in = w_out * stride - padding + kw_i;

            float w_val = W_tilde[c_out * (C_in * kH * kW) +
                                  c_in  * (kH * kW) +
                                  kh    * kW +
                                  kw_i];

            // dY_hat[b, c_in, h_in, w_in] += dO[b, c_out, h_out, w_out] * w_val
            if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                atomicAdd(
                    &dY_hat[b_idx * (C_in * H_in * W_in) +
                             c_in  * (H_in * W_in) +
                             h_in  * W_in +
                             w_in],
                    do_val * w_val
                );

                // dW[c_out, c_in, kh, kw] += dO[b, c_out, h_out, w_out] * Y_hat[b, c_in, h_in, w_in]
                float yi = Y_hat[b_idx * (C_in * H_in * W_in) +
                                 c_in  * (H_in * W_in) +
                                 h_in  * W_in +
                                 w_in];
                atomicAdd(
                    &dW[c_out * (C_in * kH * kW) +
                        c_in  * (kH * kW) +
                        kh    * kW +
                        kw_i],
                    do_val * yi
                );
            }
        }
    }
}


// ─────────────────────────────────────────────────────────────────
// BACKWARD KERNEL 2: rms_norm_bwd for Conv2d
// Computes dX from dY_hat, X, mu, var_, r per spatial position
//
// Grid: (B * H_out * W_out,) — one block per output spatial position
// Reduction over C_in (same dimension as forward RMSNorm)
// ─────────────────────────────────────────────────────────────────
__global__ void mmf_conv2d_backward_rmsnorm_kernel(
    const float* __restrict__ dY_hat,  // [B, C_in, H_in, W_in]
    const float* __restrict__ X,       // [B, C_in, H_in, W_in]
    const float* __restrict__ mu,      // [B, H_out, W_out]
    const float* __restrict__ var_,    // [B, H_out, W_out]
    const float* __restrict__ r,       // [B, H_out, W_out]
    float* __restrict__ dX,            // [B, C_in, H_in, W_in]
    int B, int C_in,
    int H_in, int W_in,
    int H_out, int W_out,
    int kH, int kW,
    int stride, int padding
) {
    int spatial_idx   = blockIdx.x;
    int total_spatial = B * H_out * W_out;
    if (spatial_idx >= total_spatial) return;

    int b_idx  = spatial_idx / (H_out * W_out);
    int hw_idx = spatial_idx % (H_out * W_out);
    int h_out  = hw_idx / W_out;
    int w_out  = hw_idx % W_out;

    float row_mu  = mu[b_idx * (H_out * W_out) + hw_idx];
    float row_r   = r [b_idx * (H_out * W_out) + hw_idx];
    float row_r3  = row_r * row_r * row_r;

    extern __shared__ float smem[];
    float* scratch = smem;

    // ── Pass 1a: dσ² = sum(dY × (X - μ)) × -0.5 × r³ ──
    // Sum over C_in at the center input position
    float sum_dsigma = 0.0f;
    for (int c = threadIdx.x; c < C_in; c += blockDim.x) {
        int h_in_c = h_out * stride - padding + kH / 2;
        int w_in_c = w_out * stride - padding + kW / 2;
        if (h_in_c >= 0 && h_in_c < H_in && w_in_c >= 0 && w_in_c < W_in) {
            int idx = b_idx * (C_in * H_in * W_in) +
                      c     * (H_in * W_in) +
                      h_in_c * W_in + w_in_c;
            sum_dsigma += dY_hat[idx] * (X[idx] - row_mu);
        }
    }
    sum_dsigma = block_reduce_sum(sum_dsigma, scratch);
    __syncthreads();

    __shared__ float row_dsigma2, row_dmu;
    if (threadIdx.x == 0)
        row_dsigma2 = sum_dsigma * (-0.5f) * row_r3;
    __syncthreads();

    // ── Pass 1b: dμ ──
    float sum_dmu_a = 0.0f, sum_xmu = 0.0f;
    for (int c = threadIdx.x; c < C_in; c += blockDim.x) {
        int h_in_c = h_out * stride - padding + kH / 2;
        int w_in_c = w_out * stride - padding + kW / 2;
        if (h_in_c >= 0 && h_in_c < H_in && w_in_c >= 0 && w_in_c < W_in) {
            int idx = b_idx * (C_in * H_in * W_in) +
                      c     * (H_in * W_in) +
                      h_in_c * W_in + w_in_c;
            sum_dmu_a += -row_r * dY_hat[idx];
            sum_xmu   += X[idx] - row_mu;
        }
    }
    sum_dmu_a = block_reduce_sum(sum_dmu_a, scratch);
    __syncthreads();
    sum_xmu   = block_reduce_sum(sum_xmu,   scratch);
    __syncthreads();

    if (threadIdx.x == 0)
        row_dmu = sum_dmu_a + row_dsigma2 * (sum_xmu / C_in) * 2.0f;
    __syncthreads();

    // ── Pass 2: dX over full receptive field (C_in, kH, kW) ──
    for (int c = threadIdx.x; c < C_in; c += blockDim.x) {
        for (int kh = 0; kh < kH; kh++) {
            for (int kw_i = 0; kw_i < kW; kw_i++) {
                int h_in = h_out * stride - padding + kh;
                int w_in = w_out * stride - padding + kw_i;
                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    int idx = b_idx * (C_in * H_in * W_in) +
                              c     * (H_in * W_in) +
                              h_in  * W_in +
                              w_in;
                    float x_val = X[idx];
                    float dy    = dY_hat[idx];
                    atomicAdd(
                        &dX[idx],
                        row_r * dy
                        + 2.0f * row_dsigma2 * (x_val - row_mu) / C_in
                        + row_dmu / C_in
                    );
                }
            }
        }
    }
}


// ─────────────────────────────────────────────────────────────────
// C++ ENTRY POINTS
// ─────────────────────────────────────────────────────────────────
std::vector<torch::Tensor> mmf_conv2d_forward(
    torch::Tensor X,       // [B, C_in, H_in, W_in]
    torch::Tensor W,       // [C_out, C_in, kH, kW]
    torch::Tensor b,       // [C_out]
    int stride, int padding, float eps
) {
    int B    = X.size(0), C_in  = X.size(1);
    int H_in = X.size(2), W_in  = X.size(3);
    int C_out = W.size(0);
    int kH   = W.size(2), kW    = W.size(3);
    int H_out = (H_in + 2*padding - kH) / stride + 1;
    int W_out = (W_in + 2*padding - kW) / stride + 1;

    auto opts    = X.options();
    auto O       = torch::empty({B, C_out, H_out, W_out}, opts);
    auto Y_hat   = torch::zeros({B, C_in,  H_in,  W_in},  opts);
    auto W_tilde = torch::empty({C_out, C_in, kH, kW},    opts);
    auto mu      = torch::empty({B, H_out, W_out},         opts);
    auto var_    = torch::empty({B, H_out, W_out},         opts);
    auto r       = torch::empty({B, H_out, W_out},         opts);

    float s_w      = 1.0f / W.abs().mean().item<float>();
    int   threads  = 256;
    int   grid     = B * H_out * W_out;
    size_t smem    = sizeof(float) * (threads / 32);

    mmf_conv2d_fused_forward_kernel<<<grid, threads, smem>>>(
        X.data_ptr<float>(), W.data_ptr<float>(), b.data_ptr<float>(),
        O.data_ptr<float>(), Y_hat.data_ptr<float>(), W_tilde.data_ptr<float>(),
        mu.data_ptr<float>(), var_.data_ptr<float>(), r.data_ptr<float>(),
        s_w, B, C_in, C_out, H_in, W_in, H_out, W_out, kH, kW,
        stride, padding, eps
    );

    return {O, Y_hat, W_tilde, mu, var_, r};
}

std::vector<torch::Tensor> mmf_conv2d_backward(
    torch::Tensor dO,      // [B, C_out, H_out, W_out]
    torch::Tensor W_tilde, // [C_out, C_in, kH, kW]
    torch::Tensor Y_hat,   // [B, C_in, H_in, W_in]
    torch::Tensor X,       // [B, C_in, H_in, W_in]
    torch::Tensor mu,      // [B, H_out, W_out]
    torch::Tensor var_,    // [B, H_out, W_out]
    torch::Tensor r        // [B, H_out, W_out]
) {
    int B     = X.size(0), C_in  = X.size(1);
    int H_in  = X.size(2), W_in  = X.size(3);
    int C_out = dO.size(1);
    int H_out = dO.size(2), W_out = dO.size(3);
    int kH    = W_tilde.size(2), kW = W_tilde.size(3);
    int stride  = (H_in + 2*(kH/2) - kH) / (H_out - 1);  // recover stride
    int padding = kH / 2;                                   // recover padding

    auto opts   = dO.options();
    auto dY_hat = torch::zeros({B, C_in, H_in, W_in},        opts);
    auto dW     = torch::zeros({C_out, C_in, kH, kW},        opts);
    auto db     = torch::zeros({C_out},                       opts);
    auto dX     = torch::zeros({B, C_in, H_in, W_in},        opts);

    int  threads = 256;
    int  grid    = B * H_out * W_out;
    size_t smem  = sizeof(float) * (threads / 32);

    // Backward kernel 1: conv linear backward
    mmf_conv2d_backward_conv_kernel<<<grid, threads>>>(
        dO.data_ptr<float>(), W_tilde.data_ptr<float>(), Y_hat.data_ptr<float>(),
        dY_hat.data_ptr<float>(), dW.data_ptr<float>(), db.data_ptr<float>(),
        B, C_in, C_out, H_in, W_in, H_out, W_out, kH, kW,
        stride, padding
    );

    // Backward kernel 2: rms_norm_bwd
    mmf_conv2d_backward_rmsnorm_kernel<<<grid, threads, smem>>>(
        dY_hat.data_ptr<float>(), X.data_ptr<float>(),
        mu.data_ptr<float>(), var_.data_ptr<float>(), r.data_ptr<float>(),
        dX.data_ptr<float>(),
        B, C_in, H_in, W_in, H_out, W_out, kH, kW,
        stride, padding
    );

    return {dX, dW, db};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mmf_conv2d_forward",  &mmf_conv2d_forward,  "MMF Conv2d fused forward (CUDA)");
    m.def("mmf_conv2d_backward", &mmf_conv2d_backward, "MMF Conv2d fused backward (CUDA)");
}