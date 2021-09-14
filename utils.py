import torch

def intersction_over_union(pred_boxes,label_boxes,box_format="corners"):
    """
    Calculates intersection over union
    Parameters:
        pred_boxes (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        label_boxes (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint"
        pred_x1 = pred_boxes[...,0:1] - pred_boxes[...,2:3] / 2 # pred_box[...,2:3] is width
        pred_y1 = pred_boxes[...,1:2] - pred_boxes[...,3:4] / 2 # pred_box[...,3:4] is height
        pred_x2 = pred_boxes[...,0:1] + pred_boxes[...,2:3] / 2
        pred_y2 = pred_boxes[...,1:2] + pred_boxes[...,3:4] / 2

        label_x1 = label_boxes[...,0:1] - label_boxes[2:3] / 2
        label_y1 = label_boxes[...,1:2] - label_boxes[3:4] / 2
        label_x2 = label_boxes[...,0:1] + label_boxes[2:3] / 2
        label_y2 = label_boxes[...,1:2] + label_boxes[3:4] / 2

# 영상에서는 if box_format 이던데 안되면 여기 볼 것
    elif box_format == "corners"
        pred_x1 = pred_boxes[...,0:1]  # maintain for tensor if you don't do it, tensor will be gone
        # this result is (N,1) but pred_boxes[...,0] result is (N)
        pred_y1 = pred_boxes[...,1:2]
        pred_x2 = pred_boxes[...,2:3]
        pred_y2 = pred_boxes[...,3:4]

        label_x1 = label_boxes[...,0:1]
        label_y1 = label_boxes[...,1:2]
        label_x2 = label_boxes[...,2:3]
        label_y2 = label_boxes[...,3:4]

    x1 = torch.max(pred_x1,label_x1)
    y1 = torch.max(pred_y1,label_y1)
    x2 = torch.min(pred_x2,label_x2)
    y2 = torch.min(pred_y2,label_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    # clamp -> minimum value = 0
    # if box doesn't exist box value is - value

    pred_area = abs((pred_x2 - pred_x1) * (pred_y2 - pred_y1))
    label_area = abs((label_x2 - label_x1) * (label_y2 - label_y1))

    return intersection / (pred_area + label_area - intersection)