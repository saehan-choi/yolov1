import numpy as np
import torch

# a = torch.tensor([
#                 [
#                 [
#                 [111, 112],
#                 [121, 122],
#                 [131, 132]
#                 ],

#                 [
#                 [211, 110],
#                 [221, 222],
#                 [231, 232]
#                 ],

#                 [
#                 [424, 112],
#                 [245, 122],
#                 [644, 132]
#                 ],

#                 [
#                 [211, 110],
#                 [221, 222],
#                 [231, 232]
#                 ]
#                 ],
#                 #######################
#                 [
#                 [
#                 [111, 112],
#                 [121, 122],
#                 [131, 132]
#                 ],

#                 [
#                 [212, 110],
#                 [222, 222],
#                 [232, 232]
#                 ],

#                 [
#                 [424, 112],
#                 [245, 122],
#                 [644, 132]
#                 ],
#                 [
#                 [211, 110],
#                 [221, 222],
#                 [231, 232]
#                 ]
#                 ]

#                 ])
# print(a.shape)
# print(a[...,1:2])
# print("꺆")
# print(a[...,1])

# print(a.shape)
# # torch.Size([2, 3, 2])
# print(a.shape[0])
# # 2
# print(a.argmax(0))

# print(a.argmax(0).size())

# b = torch.tensor([1,2,3])

# print(b)



# print(b.repeat(4,2,1))
# print(b.repeat(4,2,1).size())


print(torch.arange(7).repeat(16,7,2).unsqueeze(-1).size())
print(torch.arange(7).repeat(16,7,1).size())

# best_box = torch.tensor([[2,3,1],
#                         [2,3,5],
#                         [2,4,5],
#                         [3,5,1]])

# best_boxes =(1-best_box)
# print(best_boxes)


# a = torch.tensor([1,2])
# b = torch.tensor([2,4])

# print(a*b)