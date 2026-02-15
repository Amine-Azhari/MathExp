import cv2
import numpy as np
import torch

CLASSES = ['0','1','2','3','4','5','6','7','8','9','+','-','*','/']

def find_symbol_boxes(gray_img: np.ndarray):
    # 1. blur + treshold : we blur to have smoother result
    blur = cv2.GaussianBlur(gray_img, (3,3), 0)

    # ink is white, background is black
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. Morph close to join small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 3. Connected components
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)

    boxes = []
    for i in range(1, n):  # skip background
        x, y, w, h, area = stats[i] # initial x, init y, final x, final y, surface

        # filter noise
        if area < 40:
            continue
        if w < 3 or h < 3:
            continue

        boxes.append((x, y, w, h))

    # 4) Sort left-to-right
    boxes.sort(key=lambda b: b[0])
    return bw, boxes


def crop_to_mnist_tensor(bw: np.ndarray, box, out_size=28, pad=6):
    x, y, w, h = box
    crop = bw[y:y+h, x:x+w]  # still 0/255

    # pad
    crop = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

    # make square
    hh, ww = crop.shape
    s = max(hh, ww)
    square = np.zeros((s, s), dtype=np.uint8)
    y0 = (s - hh)//2
    x0 = (s - ww)//2
    square[y0:y0+hh, x0:x0+ww] = crop

    # resize to 28x28
    resized = cv2.resize(square, (out_size, out_size), interpolation=cv2.INTER_AREA)

    # convert to float tensor in [0,1], then normalize to [-1,1]
    t = torch.from_numpy(resized).float() / 255.0  
    t = t.unsqueeze(0).unsqueeze(0)               
    t = (t - 0.5) / 0.5  # Normalize((0.5,),(0.5,))
    return t


@torch.no_grad()
def predict_expression(gray_img: np.ndarray, model, device):
    bw, boxes = find_symbol_boxes(gray_img)
    if not boxes:
        return "", [], []

    tensors = [crop_to_mnist_tensor(bw, b) for b in boxes]
    batch = torch.cat(tensors, dim=0).to(device)

    logits = model(batch)
    preds = logits.argmax(dim=1).cpu().tolist()
    symbols = [CLASSES[i] for i in preds]

    expr = "".join(symbols)
    # convert tuples -> lists so jsonify is happy
    boxes_json = [[int(x), int(y), int(w), int(h)] for (x,y,w,h) in boxes]
    
    print("expr:", expr, "symbols:", symbols, "boxes:", boxes)
    
    return expr, boxes_json, symbols
