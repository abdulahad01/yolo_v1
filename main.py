#!/usr/bin/env python3

'''
@author : abdulahad01

Main function for YOLO object detection algorithm
'''

# Imports 
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torchvision.transforms.functional as FT

import os
from PIL import Image
import pandas as pd

from collections import Counter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOAD_MODEL = False
EPOCHS = 1000
model = YOLO_V1(split_size = 7, bb_no =2, no_classes = 20).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr = 2e-6)
loss_fn = YOLO_LOSS()

def train(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []
    model.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss = loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")


if LOAD_MODEL:
      load_checkpoint(torch.load("/content/output.pth.tar"), model, optimizer)
for epoch in range(EPOCHS):

        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4
        )

        mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5)
        print(f"Train mAP: {mean_avg_prec}")

        if mean_avg_prec > 0.9:
           checkpoint = {
               "state_dict": model.state_dict(),
               "optimizer": optimizer.state_dict(),
           }
           save_checkpoint(checkpoint, filename="/content/output.pth.tar")
           import time
           time.sleep(10)

        train(train_loader, model, optimizer, loss_fn)