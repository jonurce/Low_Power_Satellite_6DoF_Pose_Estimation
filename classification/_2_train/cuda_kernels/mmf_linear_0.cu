#include <torch/extension.h>

// One thread computes one output element: out[b, o]
// x:      [B, in_features]
// weight: [out_features, in_features]  (ternary: -1, 0, +1)
// bias:   [out_features]
// out:    [B, out_features]
__global__ void mmf_linear_kernel(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    float scale,
    int B,
    int in_features,
    int out_features
) {
    // Each thread is responsible for one (batch, output) pair
    int b = blockIdx.x * blockDim.x + threadIdx.x;  // batch index
    int o = blockIdx.y * blockDim.y + threadIdx.y;  // output index

    if (b >= B || o >= out_features) return;

    float acc = 0.0f;

    for (int i = 0; i < in_features; i++) {
        float w = weight[o * in_features + i];  // ternary weight
        float xi = x[b * in_features + i];

        // No multiplication: branch on ternary value
        if (w == 1.0f)       acc += xi;
        else if (w == -1.0f) acc -= xi;
    }

    out[b * out_features + o] = scale * acc + bias[o];
}

torch::Tensor mmf_linear(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    float scale
) {
    int B           = x.size(0);
    int in_features = x.size(1);
    int out_features = weight.size(0);

    auto out = torch::empty({B, out_features}, x.options());

    // 2D thread block: (tx, ty) map to (batch, output)
    dim3 threads(16, 16);  // 16 * 16 = 256 threads per block
    dim3 blocks(
        (B           + threads.x - 1) / threads.x,
        (out_features + threads.y - 1) / threads.y
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
    m.def("mmf_linear", &mmf_linear, "MMF Linear forward (CUDA)");
}