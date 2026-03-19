#include <torch/extension.h>

// Each thread computes one output element: out[b, c_out, h_out, w_out]
// x:      [B, C_in, H_in, W_in]
// weight: [C_out, C_in, kH, kW]  (ternary: -1, 0, +1)
// bias:   [C_out]
// out:    [B, C_out, H_out, W_out]
__global__ void mmf_conv2d_kernel(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    float scale,
    int B,
    int C_in, int C_out, int H_in, int W_in,
    int H_out, int W_out,
    int kH, int kW,
    int stride,
    int padding
) {
    // Map threads to output dimensions
    // threadIdx.x / blockIdx.x → spatial position (h_out * W_out + w_out)
    // threadIdx.y / blockIdx.y → output channel (c_out)
    // threadIdx.z / blockIdx.z → batch (b)
    int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;  // flat spatial index
    int c_out       = blockIdx.y * blockDim.y + threadIdx.y;  // output channel
    int b           = blockIdx.z * blockDim.z + threadIdx.z;  // batch

    if (b >= B || c_out >= C_out || spatial_idx >= H_out * W_out) return;

    // Recover 2D spatial position from flat index
    int h_out = spatial_idx / W_out;
    int w_out = spatial_idx % W_out;

    float acc = 0.0f;

    // Loop over input channels and kernel spatial positions
    for (int c_in = 0; c_in < C_in; c_in++) {
        for (int kh = 0; kh < kH; kh++) {
            for (int kw = 0; kw < kW; kw++) {

                // Corresponding input position (with stride and padding)
                int h_in = h_out * stride - padding + kh;
                int w_in = w_out * stride - padding + kw;

                // Skip if outside input bounds (implicit zero padding)
                if (h_in < 0 || h_in >= H_in || w_in < 0 || w_in >= W_in) continue;

                // Ternary weight for this (c_out, c_in, kh, kw)
                float w = weight[c_out * (C_in * kH * kW) +
                                 c_in  * (kH * kW) +
                                 kh    * kW +
                                 kw];

                // Input value
                float xi = x[b * (C_in * H_in * W_in) +
                              c_in * (H_in * W_in) +
                              h_in * W_in +
                              w_in];

                // No multiplication: branch on ternary value
                if      (w ==  1.0f) acc += xi;
                else if (w == -1.0f) acc -= xi;
            }
        }
    }

    // Write output: scale * acc + bias
    out[b    * (C_out * H_out * W_out) +
        c_out * (H_out * W_out) +
        h_out * W_out +
        w_out] = scale * acc + bias[c_out];
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

    // 3D thread grid:
    // x-dim → spatial (H_out * W_out)
    // y-dim → output channels (C_out)
    // z-dim → batch (B)
    dim3 threads(32, 8, 1); //32 * 8 = 256
    dim3 blocks(
        (H_out * W_out + threads.x - 1) / threads.x,
        (C_out         + threads.y - 1) / threads.y,
        (B             + threads.z - 1) / threads.z
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
    m.def("mmf_conv2d", &mmf_conv2d, "MMF Conv2d forward (CUDA)");
}