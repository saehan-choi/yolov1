import torch
# (N,S,S,number of channel)

a = torch.randn(2,4,4,20)
# (2,4,4,20)

print(a.shape)
# torch.Size([2, 4, 4, 20])

print(torch.flatten(a,start_dim=2,end_dim=-2).shape)

# print(torch.flatten(a,start_dim=1).shape)
# # torch.Size([2, 320])
# print(torch.flatten(a,start_dim=2).shape)
# # torch.Size([2, 4, 80])
# print(torch.flatten(a,start_dim=3).shape)
# # torch.Size([2, 4, 4, 20])



# print(torch.flatten(a,end_dim=-1).shape)
# # torch.Size([640])
# print(torch.flatten(a,end_dim=-2).shape)
# # torch.Size([32, 20])
# print(torch.flatten(a,end_dim=-3).shape)
# # torch.Size([8, 4, 20])