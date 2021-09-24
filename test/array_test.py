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


# print(torch.arange(7).repeat(16,7,2).unsqueeze(-1).size())
# print(torch.arange(7).repeat(16,7,1).size())
# print(torch.arange(7).repeat(16,7,1).unsqueeze(-1).permute(0,2,1,3))

# best_box = torch.tensor([[2,3,1],
#                         [2,3,5],
#                         [2,4,5],
#                         [3,5,1]])
# cell_indices = torch.arange(4).repeat(2, 4, 1).unsqueeze(-1)
# print(best_box.shape)
# print(best_box)
# best_box = best_box[...,0:1]
# best_box = best_box[...,0:1] + cell_indices
# print(best_box)

# best_boxes =(1-best_box)
# print(best_boxes)


# a = torch.tensor([1,2])
# b = torch.tensor([2,4])

# print(a*b)


# x = torch.tensor([[[[2,3,1],
#                 [2,3,5],
#                 [2,4,5],
#                 [3,5,1]],

#                 [[3,4,2],
#                 [2,4,6],
#                 [2,4,5],
#                 [3,5,1]]],

#                 [[[3,4,5],
#                 [5,4,3],
#                 [4,5,6],
#                 [3,5,1]],

#                 [[3,4,2],
#                 [2,4,6],
#                 [2,4,5],
#                 [3,5,1]]]])
# print(x.shape)
# # print(x.argmax(0))

# S = 7
# out = torch.randn(16,S,S,30)

# def convert_cellboxes(predictions, S=7):
#     """
#     Converts bounding boxes output from Yolo with
#     an image split size of S into entire image ratios
#     rather than relative to cell ratios. Tried to do this
#     vectorized, but this resulted in quite difficult to read
#     code... Use as a black box? Or implement a more intuitive,
#     using 2 for loops iterating range(S) and convert them one
#     by one, resulting in a slower but more readable implementation.
#     """

#     predictions = predictions.to("cpu")
#     batch_size = predictions.shape[0]
#     predictions = predictions.reshape(batch_size, 7, 7, 30)
    
#     # (N,S,S,4)
#     bboxes1 = predictions[..., 21:25]
#     bboxes2 = predictions[..., 26:30]
#     scores = torch.cat(
#         (predictions[...,20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim = 0
#     )

#     # argmax 하고나면 이전에 unsqueeze로 생성된 텐서는 죽어버림
#     # 그리고 argmax는 인덱스값을 반환하기 때문에 1-best_box를 하므로서 
#     # 최댓값 텐서의 인덱스텐서들이 살아남음 
#     # ex) [[1,1,1],[1,1,1]] or [[0,0,0],[0,0,0]]
#     best_box = scores.argmax(0).unsqueeze(-1)
#     # best_box -> (N,S,S)
#     best_boxes = bboxes1 * (1 - best_box) + bboxes2 * best_box
#     # best_boxes -> (N,S,S,4) * (N,S,S)  (좌표값 텐서 * obj_pred_score텐서)
#     cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
#     # 내생각엔 torch.zeros(7) 이 되야할 것 같은데..
#     # cell_indices.size() -> (batch_size, [0,1,2,...6], number of row(7), [0,1,2,...6]을 몇번 반복할건지)
#     # torch.arange(7) -> tensor([0,1,2,3,4,5,6])
#     # (16,7,7,1) 7x1 이 7개 있는게 16개 있음
#     x = 1 / S * (best_boxes[..., :1] + cell_indices)
#     y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0,2,1,3))
#     w_y = 1 / S * best_boxes[..., 2:4]

#     converted_bboxes = torch.cat((x, y, w_y), dim=-1)
#     # converted_bboxes -> (16,7,7,4)
#     # print(f"converted_bboxes.shape is {converted_bboxes.shape}")

#     predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
#     # predicted_class -> (16,7,7,1)
#     # print(f'predicted_class is {predicted_class.shape}')

#     best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(-1)
#     # best_confidence -> (16,7,7,1)
#     # print(f'best_confidence is {best_confidence.shape}')

#     converted_preds = torch.cat(
#         (predicted_class, best_confidence, converted_bboxes), dim=-1
#     )

#     # converted_preds -> (16,7,7,25)
#     return converted_preds

# converted_pred = convert_cellboxes(out)

# print(converted_pred.shape)

# a = torch.tensor([1])
# b = torch.tensor([[[1,2,3,4],
#                 [1,2,3,4]],
#                 [[1,2,3,4],
#                 [1,2,3,4]]])
# print(b.shape)
# print(torch.flatten(b, end_dim=-2).shape)
N = 16
S = 7
a = torch.randn([N,S,S,4])
b = torch.randn([N,S,S,1])
c = torch.randn([N,S,S])
print(torch.flatten(a,start_dim=1).shape)
# c = a+b
# print(c.shape)
# print(b+c)