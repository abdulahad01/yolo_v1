#!/usr/bin/env python3
'''
@ author: Aladdin Persson
'''

import numpy as np
import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def cell_to_image(pred_boxes, S=7):
    '''
    Converts the predictions from cell cordinates to Image cordinates
    '''

    batch_size = pred_boxes.shape[0]
    pred_boxes = pred_boxes.reshape(-1, S, S, 30)
    boxes_1 = pred_boxes[...,21:25]
    boxes_2 = pred_boxes[..., 26:30]
    scores_1 = pred_boxes[...,20:21]
    scores_2 = pred_boxes[...,25:26]
    best_confidence = torch.max(scores_1, scores_2)
    best_score = torch.cat((scores_1, scores_2), dim = -1).argmax(-1).unsqueeze(-1)
    # print(best_score.shape, boxes_1.shape)
    # print(((best_score) * boxes_2).shape)
    best_class = pred_boxes[...,:20].argmax(-1).unsqueeze(-1)
    # print(best_class.shape)

    best_box = (1-best_score)*boxes_1 + (best_score * boxes_2)
    # print(best_box.shape)
    for batch in range(batch_size):
        for i in range(S):
            for j in range(S):
                x = (best_box[...,:1]+j)*1/S
                y = (best_box[...,1:2]+i)*1/S
                w_h = (best_box[...,2:4])*1/S
                bb_concat = torch.cat((x,y,w_h),-1)
    corrected_box = torch.cat((best_class, best_confidence, bb_concat), dim = -1)
    return corrected_box


def get_bboxes(
    loader,
    model,
    iou_threshold,
    threshold,
    pred_format="cells",
    box_format="midpoint",
    device="cuda",
):
    all_pred_boxes = []
    all_true_boxes = []

    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
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


def cellboxes_to_boxes(out, S=7):
    converted_pred = (out).reshape(out.shape[0], S * S, -1)
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


def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle potch
    for box in boxes:
        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()