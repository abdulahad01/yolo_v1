#!/usr/bin/env python3

'''
@author : abdulahad01
Program containing various evaluation metrics and functions for object detection
'''

def intersection_over_union(bb_1, bb_2):
  """
  Get two bounding boxes and calculate their IOU
  i/p format : (batch_Size, S, S, box_cordinates[5])
  box_cordinates : (c_x,c_y,w,h) 
  """
  bb_1_x_left = (bb_1[...,0:1] - bb_1[...,2:3])/2
  bb_1_y_top = (bb_1[...,1:2] - bb_1[...,3:4])/2
  bb_1_x_right = (bb_1[...,0:1] + bb_1[...,2:3])/2
  bb_1_y_bottom = (bb_1[...,1:2] + bb_1[...,3:4])/2
  # print(bb_1_x_left, bb_1_x_right)
  # print(bb_1_y_top, bb_1_y_bottom)
  
  bb_2_x_left = (bb_2[...,0:1] - bb_2[...,2:3])/2
  bb_2_y_top =  (bb_2[...,1:2] - bb_2[...,3:4])/2
  bb_2_x_right = (bb_2[...,0:1] + bb_2[...,2:3])/2
  bb_2_y_bottom = (bb_2[...,1:2] + bb_2[...,3:4])/2

  x_left = torch.max(bb_1_x_left, bb_2_x_left)
  y_top = torch.max(bb_1_y_top, bb_2_y_top)

  # print(x_left, y_top)

  x_right = torch.min(bb_1_x_right, bb_2_x_right)
  y_bottom = torch.min(bb_1_y_bottom, bb_2_y_bottom)

  # print(x_right, y_bottom)

  inter_width = x_right - x_left
  inter_height = y_bottom - y_top

  # print(inter_width, inter_height)

  inter_area = inter_width * inter_height

  total_area = (bb_1[...,2:3]* bb_1[...,3:4]) + (bb_2[...,2:3]* bb_2[...,3:4])

  union = total_area- inter_area

  iou = inter_area/ union

  return iou

def mean_average_precision(preds, labels, iou_threshold= 0.5, NUM_CLASSES= 20):
    '''
    Calculates the mean average precision scores
    pred =[[idx,class,obj,x,y,w,h],...]
    label =[[idx,class,oj,x,y,w,h],...]
    '''
    avg_prec = []
    # For each class 
    for c in range(NUM_CLASSES):
        # variables to store the predictions and true labels for the class
        predictions = []
        true_label = []
        # print(predictions, true_label)

        # Filter out predicitons and true lables for the class
        predictions = [pred for pred in preds if pred[1] == c]
        true_label = [label for label in labels if label[1] == c]

        if len(true_label) == 0:
            continue

        # Number of ground truths per image
        boxes_per_image = Counter([bbox[0] for bbox in true_label])
        # print(boxes_per_image)
        for key, val in boxes_per_image.items():
            boxes_per_image[key] = torch.zeros(val)
        

        # variables for calucating precision and recall
        true_pos = torch.tensor([0]*len(predictions))
        false_pos = torch.tensor([0]*len(predictions))

        # Sort in order of highest objectness score 
        # predictions.sort(key=lambda x:x[2], reverse= True)
        for id_pred, pred in enumerate(predictions):
            image_boxes = [bbox for bbox in true_label if bbox[0] == pred[0]]
            best_iou = 0
            for gt_idx, gt in enumerate(image_boxes):
                iou = intersection_over_union(
                    torch.tensor(pred[3:]),
                    torch.tensor(gt[3:])
                    )
                # print(iou)
                if best_iou < iou:
                    best_iou = iou 
                    best_idx = id_pred
                    true_box = gt[0]
                    true_box_id = gt_idx

            if best_iou > iou_threshold:
                if boxes_per_image[true_box][true_box_id] == 0:
                    true_pos[best_idx] = 1
                    boxes_per_image[true_box][true_box_id] = 1
                else:
                    false_pos[id_pred] = 1
            else:
                false_pos[id_pred] = 1
        
        tp_cumsum = torch.cumsum(true_pos, dim = 0)
        fp_cumsum = torch.cumsum(false_pos, dim = 0)

        # print(true_pos, false_pos)
        # print(tp_cumsum, fp_cumsum)

        precision = tp_cumsum/(tp_cumsum+fp_cumsum + 1e-6)
        recall = tp_cumsum/(len(true_label)+1e-6)

        precision = torch.cat((torch.tensor([1]), precision))
        recall = torch.cat((torch.tensor([0]), recall))

        # print(precision, recall)
        # Using trapezoidal sum approximation
        avg_prec.append(torch.trapz(precision,recall))
        # print(avg_prec)

    
    return sum(avg_prec)/len(avg_prec)

def non_max_suppression(predictions, iou_threshold = 0.5):
    '''
    Algorithm to suppress multiple boxes for the same detection
    '''
    best_boxes = []
    predictions = sorted(predictions, key = lambda x:x[1], reverse=True)
    print(predictions)
    assert type(predictions) == list

    while(predictions):
        best_box = predictions.pop(0)
        predictions = [box for box in predictions if intersection_over_union(torch.tensor(box[2:]) , torch.tensor(best_box[2:])) < iou_threshold or best_box[0] != box[0]]

        best_boxes.append(best_box)
    
    return best_boxes