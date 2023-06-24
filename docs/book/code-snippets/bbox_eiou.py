def bbox_iou(box1, box2, x1y1x2y2=True, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2


    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    x0 = torch.min(b1_x1,b2_x1) # topleftest of top left of the boxes
    y0 = torch.min(b1_y1,b2_y1)
    x1 = torch.max(b1_x1,b2_x1) # bottomrightest of toplefts
    y1 = torch.max(b1_y1,b2_y1)
    x2 = torch.min(b1_x2,b2_x2) # topleftest of bottomrights
    y2 = torch.min(b1_y2,b2_y2)

    xmin = torch.min(x1,x2)
    ymin = torch.min(y1,y2)
    xmax = torch.max(x1,x2)
    ymax = torch.max(y1,y2)

    inter = (x2-x0)*(y2-y0) + \
            (xmin-x0)*(ymin-y0) - \
            (x1-x0)*(ymax-y0) - \
            (xmax-x0)*(y1-y0) 

    union = w1 * h1 + w2 * h2 - inter + eps
    return inter/union # using this just for covexication
    #return torch.min(inter/union,torch.Tensor([1.0]).cuda()) # use this return for EIoU without covexication
