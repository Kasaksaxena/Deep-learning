import torch

#Scaler
scaler= torch.tensor(7)
print(scaler)
print(scaler.ndim)
print(scaler.item())

#Vector 
vector = torch.tensor([7,7])
print(vector)
print(vector.ndim)
print(vector.shape)

# Matrix

MATRIX = torch.tensor([[7, 8], 
                       [9, 10]])
print(MATRIX)
print(MATRIX.ndim)
print(MATRIX.shape)


# Tensor
TENSOR = torch.tensor([[[1, 2, 3],
                        [3, 6, 9],
                        [2, 4, 5]]])

print(TENSOR)
print(TENSOR.ndim)
print(TENSOR.shape)

# Create a random tensor of size (3, 4)
RANDOM = torch.rand( size= (3,4))
print(RANDOM)
print(RANDOM.dtype)

# Create a random tensor of size (224, 224, 3)
random_image_size_tensor = torch.rand(size=(224, 224, 3))
print(random_image_size_tensor.shape)
print(random_image_size_tensor.ndim)

# Create a tensor of all zeros
zeros = torch.zeros(size=(3, 4))
print(zeros.dtype)

# Create a tensor of all ones
ones = torch.ones(size=(3, 4))
print(ones)
print(ones.dtype)

# Create a range of values 0 to 10
zero_to_ten = torch.arange(start=0, end=10, step=1)
print(zero_to_ten)

# Default datatype for tensors is float32
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None, # defaults to None, which is torch.float32 or whatever datatype is passed
                               device=None, # defaults to None, which uses the default tensor type
                               requires_grad=False) # if True, operations performed on the tensor are recorded 

print(float_32_tensor.shape) 
print(float_32_tensor.dtype) 
print(float_32_tensor.device)

float_16_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=torch.float16) # torch.half would also work

print(float_16_tensor.dtype)

# Create a tensor
some_tensor = torch.rand(3, 4)

# Find out details about it
print(some_tensor)
print(f"Shape of tensor: {some_tensor.shape}")
print(f"Datatype of tensor: {some_tensor.dtype}")
print(f"Device tensor is stored on: {some_tensor.device}") # will default to CPU