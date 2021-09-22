from numpy.lib.npyio import load
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter

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

    if box_format == "midpoint":
        pred_x1 = pred_boxes[...,0:1] - pred_boxes[...,2:3] / 2 # pred_box[...,2:3] is width
        pred_y1 = pred_boxes[...,1:2] - pred_boxes[...,3:4] / 2 # pred_box[...,3:4] is height
        pred_x2 = pred_boxes[...,0:1] + pred_boxes[...,2:3] / 2
        pred_y2 = pred_boxes[...,1:2] + pred_boxes[...,3:4] / 2

        label_x1 = label_boxes[...,0:1] - label_boxes[2:3] / 2
        label_y1 = label_boxes[...,1:2] - label_boxes[3:4] / 2
        label_x2 = label_boxes[...,0:1] + label_boxes[2:3] / 2
        label_y2 = label_boxes[...,1:2] + label_boxes[3:4] / 2

    # 영상에서는 if box_format 이던데 안되면 여기 볼 것
    elif box_format == "corners":
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

def non_max_suppression(bboxes, iou_threshold ,threshold, box_format="corners"):
    assert type(bboxes) == list
    # if not bboxes is list it's gonna error
    bboxes = [box for box in bboxes if box[1] > threshold]
    # threshold -> prob_threshold
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    # ex) 
    # bboxes = [[1,0.9,...],[2,0.9,...],[1,0.8,...],[2,0.8,...]...]
    # after 1
    # bboxes = [[2,0.9,...],[2,0.8,...]...]
    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
        box for box in bboxes
        if box[0] != chosen_box[0]
        or
        intersction_over_union(
            torch.tensor(bboxes[2:]),
            torch.tensor(chosen_box[2:]),
            box_format=box_format
        ) < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

def mAP():
    pass


def plot_image(image,boxes):
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplot(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle

    for box in boxes:
        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth = 1,
            edgecolor = "r",
            facecolor = "none"
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
    
    plt.show()

def get_boxes(
    loader,
    model,
    iou_threshold,
    threshold,
    pred_format="cells",
    box_format="midpoint",
    device="cuda"
):

    all_pred_boxes = []
    all_true_boxes = []
    
    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)
        
        batch_size = x.shape[0]
        ############################################
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)
        # cellboxes_to_boxes 함수 만들것
        ############################################
        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)
            
            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)
            
            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes

def convert_cellboxes(predictions, S=7):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios. Tried to do this
    vectorized, but this resulted in quite difficult to read
    code... Use as a black box? Or implement a more intuitive,
    using 2 for loops iterating range(S) and convert them one
    by one, resulting in a slower but more readable implementation.
    """

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, 30)
    
    # (N,S,S,4)
    bboxes1 = predictions[..., 21:25]
    bboxes2 = predictions[..., 26:30]
    scores = torch.cat(
        (predictions[...,20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim = 0
    )

    # argmax 하고나면 이전에 unsqueeze로 생성된 텐서는 죽어버림
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)

    # torch.arrange(7) -> tensor([0,1,2,3,4,5,6])
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes)
    