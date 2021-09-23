import numpy as np
import torch
# train_index = [0]


# a = [[3,5,2,3],[3,4,5,1]]
# b = train_index + a


# print(b)

# a = torch.tensor([[[[1,2,3,4],[2,3,5,6]],
#                 [[1,2,3,4],[2,3,5,6]]],
#                 [[[1,2,3,4],[2,3,5,6]],
#                 [[1,2,3,4],[2,3,5,6]]]])

# b = torch.tensor([[[[2,4,5,2],[1,5,4,5]],
#                 [[2,4,5,2],[1,5,4,5]]],
#                 [[[2,4,5,2],[1,5,4,5]],
#                 [[2,4,5,2],[1,5,4,5]]]])


# print(a.shape)

# c = torch.cat((a,b), dim=2)
# # dim이 그냥 배열 합치는거랑 똑같음
# # dim 0에서 합치고싶으면 dim 0 dim1에서 합치면 dim=1
# print(c)

# cell_indices = torch.arange(7).repeat(2, 7, 1).unsqueeze(-1)
# print(cell_indices)
# print(cell_indices.argmax(-1).shape)
# print("꺅")
# print(cell_indices.permute(0,2,1,3).shape)

a = [1,2,3,4]

print(a[1:2])