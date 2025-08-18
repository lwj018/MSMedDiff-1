import numpy as np
import math

import torch
from shapely.geometry import Polygon
from shapely.ops import unary_union
from scipy.spatial.distance import directed_hausdorff
import cv2
import numpy as np
from torch import nn


def extract_contours(mask):
    # 确保mask是0和1之间的浮点数
    assert mask.min() >= 0 and mask.max() <= 1, "Mask should be normalized between 0 and 1"

    # 将mask转换为8位无符号整数的NumPy数组
    mask_8bit = (mask.squeeze().cpu().numpy() * 255).astype(np.uint8)

    # 如果mask是三维的（例如，带有通道维），则只保留前两个维度
    if len(mask_8bit.shape) == 3:
        mask_8bit = mask_8bit[:, :, 0]

    # 确保mask的维度是二维的
    assert len(mask_8bit.shape) == 2, "Mask should have two dimensions (height, width)"

    # 找到mask中的轮廓
    contours, _ = cv2.findContours(mask_8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def simplify_contours(contours, threshold):
    simplified_contours = []
    for contour in contours:
        # 应用道格拉斯-普克算法简化轮廓
        epsilon = threshold * cv2.arcLength(contour, True)
        simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
        simplified_contours.append(simplified_contour)
    return simplified_contours


def contours_to_mask(contours, shape):
    # 创建一个空白的mask
    mask = np.zeros(shape, dtype=np.uint8)
    # 绘制轮廓到mask
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    return mask / 255  # 归一化到0-1范围


def polygon_simplification_loss(pred_mask, true_mask, threshold=1.0):
    # 提取预测和真实mask的轮廓
    pred_contours = extract_contours(pred_mask)
    true_contours = extract_contours(true_mask)

    # 简化轮廓
    simplified_pred_contours = simplify_contours(pred_contours, threshold)
    simplified_true_contours = simplify_contours(true_contours, threshold)

    # 将简化后的轮廓转换回mask
    simplified_pred_mask = contours_to_mask(simplified_pred_contours, pred_mask.shape)
    simplified_true_mask = contours_to_mask(simplified_true_contours, true_mask.shape)

    # 计算IoU
    intersection = np.logical_and(simplified_pred_mask, simplified_true_mask)
    union = np.logical_or(simplified_pred_mask, simplified_true_mask)
    iou = np.sum(intersection) / np.sum(union)

    # 可以添加更多的损失项，如顶点惩罚等
    return iou


# 假设pred_mask和true_mask是形状为B,C,H,W的mask
# 现在你可以调用polygon_simplification_loss函数


# 用于计算IoU
def iou(poly1, poly2):
    union = unary_union([poly1, poly2]).area
    intersection = poly1.intersection(poly2).area
    return intersection / union if union > 0 else 0.0


def combined_loss(pred_mask, true_mask, alpha=1.0, beta=0.01):
    # MAE Loss
    mae_criterion = nn.L1Loss()
    mae_loss = mae_criterion(pred_mask, true_mask)

    # Polygon Simplification Loss
    polygon_loss = polygon_simplification_loss(pred_mask, true_mask)

    # 结合两种损失
    total_loss = alpha * mae_loss + beta * polygon_loss
    return total_loss

if __name__ == '__main__':
    t1 = torch.rand((2, 1, 256, 256))  # 随机生成一个mask
    t2 = torch.rand((2, 1, 256, 256))  # 随机生成另一个mask

    print(combined_loss(t1, t2))
