#include <torch/extension.h>

// This function runs on the GPU, once per thread
// Each thread handles one element of the output
__global__ void vector_add_kernel(
    const float* a,    // input vector A
    const float* b,    // input vector B
    float* out,        // output vector
    int n              // total number of elements
) {
    // Which element is THIS thread responsible for? 
    // gives every thread a unique global index i
    // i = (block index * block dim) + thread index
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Guard: don't go out of bounds
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

// This is the C++ function PyTorch will call
torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
    int n = a.numel();
    auto out = torch::empty_like(a);

    // How many threads per block? 256 is a standard default -> warp
    int threads = 256;

    // How many blocks do we need to cover all n elements?
    int blocks = (n + threads - 1) / threads;

    // Launch the kernel: <<<blocks, threads>>>
    vector_add_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        n
    );

    return out;
}

// Expose to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add", &vector_add, "Vector addition (CUDA)");
}