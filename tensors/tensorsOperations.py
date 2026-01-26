import torch

# Create a tensor of values and add a number to it
tensor = torch.tensor([1, 2, 3])
print(tensor + 10)

# Multiply it by 10
print(tensor * 10)

# Tensors don't change unless reassigned
print(tensor)


# Subtract and reassign
tensor = tensor - 10
print(tensor)

# Can also use torch functions
print(torch.multiply(tensor, 10))

# Element-wise multiplication (each element multiplies its equivalent, index 0->0, 1->1, 2->2)
print(tensor, "*", tensor)
print("Equals:", tensor * tensor)

# Matrix multiplication
print(torch.matmul(tensor, tensor))

# Shapes need to be in the right way  
tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]], dtype=torch.float32)

tensor_B = torch.tensor([[7, 10],
                         [8, 11], 
                         [9, 12]], dtype=torch.float32)

#torch.matmul(tensor_A, tensor_B) # (this will error)
#RuntimeError: mat1 and mat2 shapes cannot be multiplied (3x2 and 3x2)

# View tensor_A and tensor_B
print(tensor_A)
print(tensor_B)

# View tensor_A and tensor_B.T
print(tensor_A)
print(tensor_B.T) #tensor.T - where tensor is the desired tensor to transpose.


# The operation works when tensor_B is transposed
print(f"Original shapes: tensor_A = {tensor_A.shape}, tensor_B = {tensor_B.shape}\n")
print(f"New shapes: tensor_A = {tensor_A.shape} (same as above), tensor_B.T = {tensor_B.T.shape}\n")
print(f"Multiplying: {tensor_A.shape} * {tensor_B.T.shape} <- inner dimensions match\n")
print("Output:\n")
output = torch.matmul(tensor_A, tensor_B.T)
print(output) 
print(f"\nOutput shape: {output.shape}")

# torch.mm is a shortcut for matmul
print(torch.mm(tensor_A, tensor_B.T))