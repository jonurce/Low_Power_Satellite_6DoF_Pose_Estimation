import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load


################ Load MMF Linear Kernel ################
mmf_linear_ext = load(
    name="mmf_linear_ext",
    sources=["classification/_2_train/cuda_kernels/mmf_linear.cu"],
    verbose=False
)

################ MMF Linear Function ################
class MMFLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, b, eps):
        O, Y_hat, W_tilde, mu, var_, r = mmf_linear_ext.mmf_linear_forward(X, W, b, eps)
        ctx.save_for_backward(W_tilde, Y_hat, X, mu, var_, r)
        ctx.N = X.size(1)
        return O

    @staticmethod
    def backward(ctx, dO):
        W_tilde, Y_hat, X, mu, var_, r = ctx.saved_tensors
        dX, dW, db = mmf_linear_ext.mmf_linear_backward(dO.contiguous(), W_tilde, Y_hat, X, mu, var_, r)
        return dX, dW, db, None  # None for eps

################ MMF Linear Layer ################
class MMFLinear(nn.Module):
    def __init__(self, in_features, out_features, eps=1e-8):
        super().__init__()
        # Ternary weights: -1, 0, +1 [C_out, C_in]
        self.W   = nn.Parameter(torch.randn(in_features, out_features) * 0.02)

        # Bias [C_out]
        self.b   = nn.Parameter(torch.zeros(out_features))

        self.eps = eps

    def forward(self, X):
        return MMFLinearFunction.apply(X, self.W, self.b, self.eps)







################ Load MMF Conv2d Kernel ################
mmf_conv2d_ext = load(
    name="mmf_conv2d_ext",
    sources=["classification/_2_train/cuda_kernels/mmf_conv2d.cu"],
    verbose=False
)

################ MMF Conv2d Function ################
class MMFConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, b, stride, padding, eps):
        O, Y_hat, W_tilde, mu, var_, r = mmf_conv2d_ext.mmf_conv2d_forward(X, W, b, stride, padding, eps)
        ctx.save_for_backward(W_tilde, Y_hat, X, mu, var_, r)
        ctx.stride  = stride
        ctx.padding = padding
        return O

    @staticmethod
    def backward(ctx, dO):
        W_tilde, Y_hat, X, mu, var_, r = ctx.saved_tensors
        dX, dW, db = mmf_conv2d_ext.mmf_conv2d_backward(dO.contiguous(), W_tilde, Y_hat, X, mu, var_, r)
        return dX, dW, db, None, None, None  # None for stride, padding, eps

    
################ MMF Conv2d Layer ################
class MMFConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, eps=1e-8):
        super().__init__()
        # Weights: -1, 0, +1 [C_out, C_in, kH, kW]
        self.W = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.02)

        # Bias [C_out]
        self.b = nn.Parameter(torch.zeros(out_channels))

        # Stride and padding from Conv2d
        self.stride = stride
        self.padding = padding
        self.eps     = eps

    def forward(self, X):
        return MMFConv2dFunction.apply(X, self.W, self.b, self.stride, self.padding, self.eps)







################ Tests ################
if __name__ == "__main__":
    print("Testing MMFLinear...")
    x = torch.randn(32, 128, device="cuda", requires_grad=True)  # batch 32, dim 128
    lin_mmf = MMFLinear(128, 64).cuda()
    out_mmf = lin_mmf(x)
    print("MMF output shape:", out_mmf.shape) # [32, 64]
    out_mmf.sum().backward()
    print("dX shape:", x.grad.shape)    # [32, 128]
    print("dW shape:", lin_mmf.W.grad.shape)  # [128, 64]

    print("\n") 

    print("Testing MMFConv2d...")
    x = torch.randn(16, 3, 32, 32, device="cuda", requires_grad=True)  # batch 16, 3 channels, 32×32
    conv_mmf = MMFConv2d(3, 64, kernel_size=3, stride=1, padding=1).cuda()
    out_mmf = conv_mmf(x)
    print("MMF output shape:", out_mmf.shape)  # [16, 64, 32, 32]
    out_mmf.sum().backward()
    print("dX shape:", x.grad.shape)       # [16, 3, 32, 32]
    print("dW shape:", conv_mmf.W.grad.shape) # [64, 3, 3, 3]
    print("db shape:", conv_mmf.b.grad.shape) # [64]