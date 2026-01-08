import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Number of GPUs:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("GPU Memory:", torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, "GB")

    # Test GPU computation
    x = torch.rand(1000, 1000).cuda()
    y = torch.rand(1000, 1000).cuda()
    z = x @ y
    print("\n✅ GPU computation test successful!")
else:
    print("\n❌ CUDA not available. Check installation.")