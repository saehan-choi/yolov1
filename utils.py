from numpy.lib.npyio import load
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter

def intersection_over_union(pred_boxes,label_boxes,box_format="corners"):
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
        intersection_over_union(
            torch.tensor(bboxes[2:]),
            torch.tensor(chosen_box[2:]),
            box_format=box_format
        ) < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):
    """
    Calculates mean average precision 
    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones 
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes
    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)
        
        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)


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

def get_bboxes(
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
    # 그리고 argmax는 인덱스값을 반환하기 때문에 1-best_box를 하므로서 
    # 최댓값 텐서의 인덱스텐서들이 살아남음 
    # ex) [[1,1,1],[1,1,1]] or [[0,0,0],[0,0,0]]
    best_box = scores.argmax(0).unsqueeze(-1)
    # best_box -> (N,S,S)
    best_boxes = bboxes1 * (1 - best_box) + bboxes2 * best_box
    # best_boxes -> (N,S,S,4) * (N,S,S)  (좌표값 텐서 * obj_pred_score텐서)
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    # 내생각엔 torch.zeros(7) 이 되야할 것 같은데..
    # cell_indices.size() -> (batch_size, [0,1,2,...6], number of row(7), [0,1,2,...6]을 몇번 반복할건지)
    # torch.arange(7) -> tensor([0,1,2,3,4,5,6])
    # (16,7,7,1) 7x1 이 7개 있는게 16개 있음
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0,2,1,3))
    w_y = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    # converted_bboxes -> (16,7,7,22)
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(-1)

    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )
    # converted_preds -> (16,7,7,25)
    return converted_preds

def cellboxes_to_boxes(out, S=7):
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])