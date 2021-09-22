import torch

a = torch.Tensor([
    [
        [1,2,3],
        [2,3,4],
        [3,4,5]
    ],

    [
        [4,5,6],
        [7,8,9],
        [10,11,12]
    ]
])

b = torch.Tensor([
    [
        [5,4,3],
        [3,6,4],
        [5,1,2]
    ],

    [
        [3,5,4],
        [3,3,4],
        [6,21,12]
    ]
])

print(a.size())
print(b.size())

c = torch.cat((a,b),dim=0)
# dim 0 -> array[0] 
# dim 1 -> array[1] 
# dim 2 -> array[2] 이런식으로 감 배열끼리 합침
print(c)