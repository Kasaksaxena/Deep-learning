import torch

# Create a tensor
y = torch.arange(0, 100, 10)
print(y)

print(f"Minimum: {y.min()}")
print(f"Maximum: {y.max()}")
# print(f"Mean: {y.mean()}") # this will error
print(f"Mean: {y.type(torch.float32).mean()}") # won't work without float datatype
print(f"Sum: {y.sum()}")

# Create a tensor
tensor = torch.arange(10, 100, 10)
print(f"Tensor: {tensor}")

# Returns index of max and min values
print(f"Index where max value occurs: {tensor.argmax()}")
print(f"Index where min value occurs: {tensor.argmin()}")

print(tensor.dtype)

x = torch.arange(1., 8.)
print(x)
print(x.shape)
# Add an extra dimension
x_reshaped = x.reshape(1, 7)
print(x_reshaped)  
print(x_reshaped.shape)

# Change view (keeps same data as original but changes view)

z = x.view(1, 7)
print(z) 
print(z.shape)

# Changing z changes x
z[:, 0] = 5
print(z) 
print(x)

# Stack tensors on top of each other
x_stacked = torch.stack([x, x, x, x], dim=0) # try changing dim to dim=1 and see what happens
print(x_stacked)

x_stacked1 = torch.stack([x, x, x, x], dim=1) # try changing dim to dim=1 and see what happens
print(x_stacked1)

print(f"Previous tensor: {x_reshaped}")
print(f"Previous shape: {x_reshaped.shape}")

# Remove extra dimension from x_reshaped
x_squeezed = x_reshaped.squeeze()
print(f"\nNew tensor: {x_squeezed}")
print(f"New shape: {x_squeezed.shape}")

# Create tensor with specific shape
x_original = torch.rand(size=(224, 224, 3))

# Permute the original tensor to rearrange the axis order
x_permuted = x_original.permute(2, 0, 1) # shifts axis 0->1, 1->2, 2->0

print(f"Previous shape: {x_original.shape}")
print(f"New shape: {x_permuted.shape}")
