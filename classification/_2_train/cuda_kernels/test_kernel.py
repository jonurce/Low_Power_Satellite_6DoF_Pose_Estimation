

from torch.utils.cpp_extension import load
import torch

# JIT-compiles the .cu file the first time, cached after that
vector_add_ext = load(
    name="vector_add_ext",
    sources=["classification/_2_train/cuda_kernels/vector_add.cu"],
    verbose=False
)

a = torch.tensor([1.0, 2.0, 3.0], device="cuda")
b = torch.tensor([10.0, 20.0, 30.0], device="cuda")

out = vector_add_ext.vector_add(a, b)
print(out)  # tensor([11., 22., 33.], device='cuda:0')