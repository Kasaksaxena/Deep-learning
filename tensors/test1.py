import torch
print(torch.__version__)

# if torch.cuda.is_available():
#     print("GPU IS AVAILABLE")
#     print(f" Using gpu { torch.cuda.is_device_name(0)}")

# else :
#     print("Gpu not available using CPU")


if torch.cuda.is_available():
    device = "cuda" # Use NVIDIA GPU (if available)
elif torch.backends.mps.is_available():
    device = "mps" # Use Apple Silicon GPU (if available)
else:
    device = "cpu" # Default to CPU if no GPU is available
     
# Create tensor (default on CPU)
tensor = torch.tensor([1, 2, 3])

# Tensor not on GPU
print(tensor, tensor.device)

# Move tensor to GPU (if available)
tensor_on_gpu = tensor.to(device)
tensor_on_gpu