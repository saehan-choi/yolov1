"""
Implementation of Yolo Loss Function from the original yolo paper
"""

import torch
import torch.nn as nn
from utils import intersection_over_union


class YoloLoss(nn.Module):
    """
    Calculate the loss for yolo (v1) model
    """

    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

        """
        S is split size of image (in paper 7),
        B is number of boxes (in paper 2),
        C is number of classes (in paper and VOC dataset is 20),
        """
        self.S = S
        self.B = B
        self.C = C

        # These are from Yolo paper, signifying how much we should
        # pay loss for no object (noobj) and the box coordinates (coord)
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        
        # predictions are shaped (BATCH_SIZE, S*S(C+B*5) when inputted
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # Calculate IoU for the two predicted bounding boxes with target bbox
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        # Take the box with highest IoU out of the two prediction
        # Note that bestbox will be indices of 0, 1 for which bbox was best
        iou_maxes, bestbox = torch.max(ious, dim=0)
        # torch.max는 앞은 실제 max value를 반환, 두번째는 max의 index를 반환
        # 따라서 앞쪽의 max가 높다면, 
        exists_box = target[..., 20].unsqueeze(3)  # in paper this is Iobj_i

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        # Set boxes with no object in them to 0. We only take out one of the two 
        # predictions, which is the one with highest Iou calculated previously.
        box_predictions = exists_box * (
            (
                bestbox * predictions[..., 26:30]
                + (1 - bestbox) * predictions[..., 21:25]
            )
        )
        # (N,S,S,1)*(N,S,S,4) = (N,S,S,4)

        box_target = exists_box * target[..., 21:25]

        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4])
            #  여기에 1e-6 이거 있었는데 안더했음 어차피 abs해서 필요없을거 같음.
        )

        box_target[..., 2:4] = torch.sqrt(box_target[..., 2:4])

        # (N,S,S,4) -> (N*S*S,4)
        box_loss = self.mse(
            # flatten end_dim 이 약간 헷갈림.
            # 4차원 텐서를 2차원 텐서로 만든다고 생각하면 편할듯.
            torch.flatten(box_target, end_dim=-2),
            torch.flatten(box_predictions, end_dim=-2)
        )
        # result == (N*S*S,2)
        # 여기는 mean_squared_error기 때문에 둘의 차원만 똑같으면 결과값은 단일 숫자값으로 나옴

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        pred_box = (bestbox) * predictions[...,25:26] + (1-bestbox) * predictions[...,20:21]
        # bestbox 가 0이되면 앞의 predictions[...,20:21]의 것 만 나옴 따라서 순서맞음

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[...,20:21])
        )
        # result == (N*S*S)

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        no_object_loss = self.mse(
            torch.flatten((1-exists_box) * predictions[...,20:21], start_dim=1),
            torch.flatten((1-exists_box) * target[...,20:21], start_dim=1)
        )
        
        no_object_loss += self.mse(
            torch.flatten((1-exists_box) * predictions[...,25:26], start_dim=1),
            torch.flatten((1-exists_box) * target[...,25:26], start_dim=1)
        )
        # (N,S,S,1) -> flatten start_dim = 1
        # result == (N,S*S)

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== 
        # #

        # (N,S,S,20) -> (N*S*S,20)
        class_loss = self.mse(
            torch.flatten(predictions[..., :20], end_dim=-2),
            torch.flatten(target[..., :20], end_dim=-2)
        )

        loss = (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )

        return loss