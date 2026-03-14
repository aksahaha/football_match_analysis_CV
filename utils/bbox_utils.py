"""
Bounding box utility functions for YOLO football project.
All bounding boxes are expected in the format (x1, y1, x2, y2).
"""

from typing import Tuple

def get_center_of_bbox(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """
    Calculate the center (x, y) of a bounding box.
    Args:
        bbox: (x1, y1, x2, y2)
    Returns:
        (center_x, center_y): Tuple[int, int]
    """
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return center_x, center_y

def get_bbox_width(bbox: Tuple[int, int, int, int]) -> int:
    """
    Calculate the width of a bounding box.
    Args:
        bbox: (x1, y1, x2, y2)
    Returns:
        width: int
    """
    x1, _, x2, _ = bbox
    return x2 - x1

def measure_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """
    Calculate the Euclidean distance between two points.
    Args:
        p1: (x1, y1)
        p2: (x2, y2)
    Returns:
        distance: float
    """
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def measure_xy_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> Tuple[int, int]:
    """
    Calculate the difference in x and y between two points.
    Args:
        p1: (x1, y1)
        p2: (x2, y2)
    Returns:
        (dx, dy): Tuple[int, int]
    """
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx, dy

def get_foot_position(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """
    Get the foot (bottom center) position of a bounding box.
    Args:
        bbox: (x1, y1, x2, y2)
    Returns:
        (foot_x, foot_y): Tuple[int, int]
    """
    x1, _, x2, y2 = bbox
    foot_x = int((x1 + x2) / 2)
    foot_y = int(y2)
    return foot_x, foot_y