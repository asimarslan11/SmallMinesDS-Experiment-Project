"""Check that PyTorch can see and compute on the GPU.

Usage: .venv\\Scripts\\python scripts\\check_gpu.py
"""
import sys

# Importing torch can take several seconds on Windows; show that the script is running
print("Loading PyTorch...", flush=True)

import torch
import torchvision


def main() -> int:
    print("torch", torch.__version__, "| torchvision", torchvision.__version__,
          "| CUDA", torch.version.cuda, "| cuDNN", torch.backends.cudnn.version())

    available = torch.cuda.is_available()
    print("cuda available:", available)
    if not available:
        print("FAIL: PyTorch cannot use the GPU (CPU-only build or driver problem).")
        return 1

    d = torch.cuda.get_device_properties(0)
    print(d.name, round(d.total_memory / 2**30, 1), "GB, compute", f"{d.major}.{d.minor}")

    # Matrix multiply on the GPU, compared against the CPU result
    x = torch.randn(4096, 4096, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    matmul_ok = torch.allclose(y[:64, :64].cpu(), (x.cpu() @ x.cpu())[:64, :64], atol=1e-1)
    print("GPU matmul ok, allclose CPU:", matmul_ok)

    # A convolution layer on the GPU
    c = torch.nn.Conv2d(3, 16, 3).cuda()
    print("conv ok", c(torch.randn(2, 3, 64, 64, device="cuda")).shape)

    if not matmul_ok:
        print("FAIL: GPU result does not match CPU.")
        return 1
    print("OK: GPU works with PyTorch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
