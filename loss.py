#!/usr/bin/env python3

'''
@author : abdulahad01

Loss function for yolo
'''

class YOLO_LOSS(nn.Module):
  """
  The yolo loss function :
  Hyperparameter : Lambda coord, lambda no obj
  1. A mean squared loss over box cordinates ( x,y,w,h). Takes loss over the best boxes (IOU scores) and ignores boxes without objects
  2. A mean squared loss over objectness score
  3. A mean squared loss over no object (inverse of objectness)
  4. A mean squared loss over the class probabilities
  """
  def __init__(self, S=7, B=2, C=20):
    super().__init__()
    self.S = S
    self.B = B
    self.C = C
    self.mse = nn.MSELoss(reduction='sum')
    self.lambda_noobj = 0.5
    self.lambda_coord = 5
  
  def forward(self, predictions, target):
    predictions = predictions.reshape(-1, self.S, self.S, self.C+ self.B*5)
    # 0:20 : Class scores 
    # 20 : Probability scores
    # 21:25 : Box 1
    # 25 : Probability scores
    # 26:30 : Box 2
    iou_b1 = intersection_over_union(predictions[..., 21:25], target[...,21:25])
    iou_b2 = intersection_over_union(predictions[...,26:30], target[...,21:25])
    ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim = 0)
    iou_maxes, best_box = torch.max(ious, dim=0)  # best box = 0 or 1
    # Target shape :  (batch, num_boxes, 25--> obj, x,y,w,h,20 classes)
    exists_box = target[..., 20].unsqueeze(3) #Iobj_i , 3 Dim tensor

    ###                           ###
    # Box cordinates : ( x, y, w, h)#
    ###                           ###
    # 4 Dim -> 3 Dim
    # print(exists_box.shape, best_box.shape, predictions.shape)
    box_predictions = exists_box * ( best_box * predictions[...,26:30] +
                                    (1 - best_box) * predictions[..., 21:25])
    box_targets = exists_box * target[..., 21:25]
    # for width and height
    box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * \
                                            torch.sqrt( torch.abs(box_predictions[...,2:4] + 1e-6))
    box_targets[...,2:4] = torch.sqrt(box_targets[..., 2:4])

    box_loss = self.mse(
        torch.flatten(box_predictions, end_dim=-2),
        torch.flatten(box_targets, end_dim = -2)
    )

    ###              ##
    # For object loss #
    ###              ##
    pred_box = (
        best_box * predictions[...,25:26] + ( 1- best_box) * predictions[..., 20:21]
    )

    object_loss = self.mse(
        torch.flatten(exists_box * pred_box),
        torch.flatten(exists_box * target[..., 20:21])
    )

    # For no object
    no_object_loss = self.mse(
        torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim =1),
        torch.flatten((1 - exists_box) * target[..., 20:21], start_dim =1)
    )
    no_object_loss += self.mse(
        torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim =1),
        torch.flatten((1 - exists_box) * target[..., 20:21], start_dim =1)
    )

    ###
    # Class loss
    ###

    class_loss = self.mse(
        torch.flatten( exists_box * predictions[..., :20], end_dim = -2),
        torch.flatten( exists_box * target[..., :20], end_dim = -2)
    )

    # From paper 
    loss = (
        self.lambda_coord * box_loss +
        object_loss +
        self.lambda_noobj * no_object_loss
        +class_loss
    )

    return loss