import torch

"""
    Create the following tensors:
        1. 3D tensor of shape 20x30x40 with all values = 0
        2. 1D tensor containing the even numbers between 10 and 100
"""
print("___________1______________________")
tensor_1 = torch.zeros(20,30,40)
tensor_2 = torch.tensor([2*i for i in range(5, 50)])

print(f"1.___ {tensor_1.shape}")
print(f"2.___{tensor_2}")
"""
    x = torch.rand(4, 6)
    Calculate:
        1. Sum of all elements of x
        2. Sum of the columns of x  (result is a 6-element tensor)
        3. Sum of the rows of x   (result is a 4-element tensor)
"""
print("___________2______________________")
x = torch.rand(4,6)
sum_x = torch.sum(x)
print(f"1.___{sum_x}")
sum_column = torch.sum(x, dim=0)
print(f"2.___{sum_column.shape}")
sum_row = torch.sum(x, dim=1)
print(f"3.___{sum_row.shape}")

"""
    Calculate cosine similarity between 2 1D tensor:
    x = torch.tensor([0.1, 0.3, 2.3, 0.45])
    y = torch.tensor([0.13, 0.23, 2.33, 0.45])
"""
print("___________3______________________")
x = torch.tensor([0.1, 0.3, 2.3, 0.45])
y = torch.tensor([0.13, 0.23, 2.33, 0.45])
cosine_sim = torch.cosine_similarity(x, y, dim=0)
print(f"Consine similarity 1d: {cosine_sim}")

"""
    Calculate cosine similarity between 2 2D tensor:
    x = torch.tensor([[ 0.2714, 1.1430, 1.3997, 0.8788],
                      [-2.2268, 1.9799, 1.5682, 0.5850],
                      [ 1.2289, 0.5043, -0.1625, 1.1403]])
    y = torch.tensor([[-0.3299, 0.6360, -0.2014, 0.5989],
                      [-0.6679, 0.0793, -2.5842, -1.5123],
                      [ 1.1110, -0.1212, 0.0324, 1.1277]])
"""
print("___________4______________________")
x = torch.tensor([[ 0.2714, 1.1430, 1.3997, 0.8788],
                    [-2.2268, 1.9799, 1.5682, 0.5850],
                    [ 1.2289, 0.5043, -0.1625, 1.1403]])
y = torch.tensor([[-0.3299, 0.6360, -0.2014, 0.5989],
                    [-0.6679, 0.0793, -2.5842, -1.5123],
                    [ 1.1110, -0.1212, 0.0324, 1.1277]])
cosine_sim_2 = torch.cosine_similarity(x.flatten(), y.flatten(), dim=0)
print(f"Cosine similarity 2d: {cosine_sim_2}")
"""
    x = torch.tensor([[ 0,  1],
                      [ 2,  3],
                      [ 4,  5],
                      [ 6,  7],
                      [ 8,  9],
                      [10, 11]])
    Make x become 1D tensor
    Then, make that 1D tensor become 3x4 2D tensor 
"""
print("___________5______________________")
x = torch.tensor([[ 0,  1],
                [ 2,  3],
                [ 4,  5],
                [ 6,  7],
                [ 8,  9],
                [10, 11]])
x_1d = x.view(-1)
print(f"X to 1d tensor: {x_1d}, shape: {x_1d.shape}")
x_2d = x_1d.view(3,4)
print(f"X to 2d tensor: {x_2d}, shape: {x_2d.shape}")
"""
    x = torch.rand(3, 1080, 1920)
    y = torch.rand(3, 720, 1280)
    Do the following tasks:
        1. Make x become 1x3x1080x1920 4D tensor
        2. Make y become 1x3x720x1280 4D tensor
        3. Resize y to make it have the same size as x
        4. Join them to become 2x3x1080x1920 tensor
"""
print("___________6______________________")
x = torch.rand(3, 1080, 1920)
y = torch.rand(3, 720, 1280)

x_unsqueeze = x.unsqueeze(dim=0)
print(f"1____x: {x_unsqueeze.shape}")
y_unsqueeze = y.unsqueeze(dim=0)
print(f"2____y: {y_unsqueeze.shape}")

from torch.nn.functional import interpolate

target_size = x.shape[1:]

y_resized = interpolate(y.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False)
print(f"3____y resized: {y_resized.shape}")
xy_combined = torch.cat([x_unsqueeze, y_resized])
print(f"4____x, y combined: {xy_combined.shape}")