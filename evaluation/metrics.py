import cv2
import torch
from PIL import Image
import math
import numpy as np
from torch import Tensor
import ipyvolume as ipv

def tf_fn_draw(input: Tensor, output: Tensor, target: Tensor) -> Image:
    output = output > 0.5
    target = target > 0
    
    img = np.zeros((output.shape[1], output.shape[2], 3))

    if torch.max(output) == 0 and torch.max(target) == 0:    
        img[:, :, 0] = input[0, :, :].cpu().numpy() * 255
        img[:, :, 1] = input[0, :, :].cpu().numpy() * 255
        img[:, :, 2] = input[0, :, :].cpu().numpy() * 255
        PIL_image = Image.fromarray(img.astype('uint8'), 'RGB')
        return PIL_image

    for i in range(target.shape[1]):
        for j in range(target.shape[2]):
            if target[0, i, j] == 0 and output[0, i, j] == 1:
                img[i, j, 0] = 255
            elif target[0, i, j] == 1 and output[0, i, j] == 0:
                img[i, j, 2] = 255
            elif target[0, i, j] == 1 and output[0, i, j] == 1:
                img[i, j, 1] = 255
            else:
                img[i, j, 0] = input[0, i, j].cpu().numpy() * 255
                img[i, j, 1] = input[0, i, j].cpu().numpy() * 255
                img[i, j, 2] = input[0, i, j].cpu().numpy() * 255

    PIL_image = Image.fromarray(img.astype('uint8'), 'RGB')
    return PIL_image


def draw_center_of_canal(input: Tensor, outputX: list, outputY: list, targetX: list, targetY: list) -> Image:
    img = np.zeros((input.shape[1], input.shape[2], 3))
    
    img[:, :, 0] = input[0, :, :].cpu().numpy() * 255
    img[:, :, 1] = input[0, :, :].cpu().numpy() * 255
    img[:, :, 2] = input[0, :, :].cpu().numpy() * 255

    for i in range(len(outputX)):
        img[outputY[i], outputX[i], 0] = 255
    
    for i in range(len(targetX)):
        if img[targetY[i], targetX[i], 0] == 255 and img[targetY[i], targetX[i], 1] != 255:
            img[targetY[i], targetX[i], 0] = 0
            img[targetY[i], targetX[i], 1] = 255
        else:
            img[targetY[i], targetX[i], 2] = 255

    PIL_image = Image.fromarray(img.astype('uint8'), 'RGB')
    return PIL_image


def find_center(contour: np.ndarray) -> tuple[int, int]:
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return cX, cY
    else:
        return None, None


def center_of_canal(tensor: Tensor) -> tuple[list, list]:
    tensor = torch.squeeze(tensor, 0).numpy().transpose(1, 2, 0).astype(np.uint8)*255
    contours, _ = cv2.findContours(tensor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = [find_center(contour) for contour in contours]
    X = [center[0] for center in centers if center[0] is not None]
    Y = [center[1] for center in centers if center[1] is not None]

    return X, Y


def centers_of_canals(output_tensor: Tensor, target_tensor: Tensor) -> float:
    output_x, output_y = center_of_canal(output_tensor)
    target_x, target_y = center_of_canal(target_tensor)

    if len(output_x) != 0 and len(target_x) != 0:
        distances = find_min_dist(output_x, target_x, output_y, target_y)
    else:
        distances = 'nan'

    return distances, output_x, output_y, target_x, target_y


def find_min_dist(output_x: list, target_x: list, output_y: list, target_y: list) -> float:
    distances = []
    shorter_x = min(output_x, target_x, key=len)
    longer_x = output_x if shorter_x == target_x else target_x
    shorter_y = min(output_y, target_y, key=len)
    longer_y = output_y if shorter_y == target_y else target_y

    for i in range(len(shorter_x)):
        min_difference = -1
        for j in range(len(longer_x)):
            dist = math.sqrt((shorter_x[i] - longer_x[j])**2 + (shorter_y[i] - longer_y[j])**2)
            if min_difference == -1 or dist < min_difference:
                min_difference = dist
        distances.append(min_difference)

    return np.sum(distances) / len(distances)