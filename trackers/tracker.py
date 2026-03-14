"""Tracker utilities for detecting and annotating players, referees and ball.

This module provides a `Tracker` class that wraps an Ultralytics YOLO model
and a ByteTrack tracker (via `supervision`) to extract and render
object tracks across video frames.

Notes:
- BBoxes are expected in (x1, y1, x2, y2) format.
"""

from typing import Any, Dict, List, Optional, Tuple

import os
import pickle

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import supervision as sv

from utils import get_center_of_bbox, get_bbox_width, get_foot_position


class Tracker:
    """High-level tracker for extracting and annotating object tracks.

    Attributes:
        model: Ultralytics YOLO model instance.
        tracker: supervision.ByteTrack instance for data association.
    """

    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(self, tracks: Dict[str, List[Dict[int, Dict[str, Any]]]]) -> None:
        """Add a `position` (point) to each track entry based on bbox.

        For the ball we use the bbox center, for players/referees we use
        the bottom-center (foot position).
        """
        for obj_type, frames_list in tracks.items():
            for frame_idx, frame_tracks in enumerate(frames_list):
                for track_id, track_info in frame_tracks.items():
                    bbox = track_info.get("bbox")
                    if not bbox:
                        continue
                    if obj_type == "ball":
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[obj_type][frame_idx][track_id]["position"] = position

    def interpolate_ball_positions(self, ball_positions: List[Dict[int, Dict[str, List[int]]]]) -> List[Dict[int, Dict[str, List[int]]]]:
        """Interpolate missing ball bboxes across frames.

        Args:
            ball_positions: list where each entry is a dict like {1: {"bbox": [x1,y1,x2,y2]}}
        Returns:
            A new list with interpolated bbox values where possible.
        """
        # Extract bbox lists (or None) and build a DataFrame for interpolation
        rows = [entry.get(1, {}).get("bbox", [None, None, None, None]) for entry in ball_positions]
        df = pd.DataFrame(rows, columns=["x1", "y1", "x2", "y2"])

        df = df.interpolate().bfill().ffill()

        interpolated = [{1: {"bbox": row}} for row in df.to_numpy().tolist()]
        return interpolated

    def detect_frames(self, frames: List[np.ndarray], batch_size: int = 20, conf: float = 0.1) -> List[Any]:
        """Run model predictions on frames in batches.

        Returns a list of model result objects (one per frame).
        """
        detections: List[Any] = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i : i + batch_size]
            results = self.model.predict(batch, conf=conf)
            detections.extend(results)
        return detections

    def get_object_tracks(
        self,
        frames: List[np.ndarray],
        read_from_stub: bool = False,
        stub_path: Optional[str] = None,
    ) -> Dict[str, List[Dict[int, Dict[str, Any]]]]:
        """Detect objects and produce tracks per-frame for players/referees/ball.

        If `read_from_stub` is True and `stub_path` exists, loads tracks from
        that pickle file instead of re-running detection.
        """
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        detections = self.detect_frames(frames)

        tracks: Dict[str, List[Dict[int, Dict[str, Any]]]] = {
            "players": [],
            "referees": [],
            "ball": [],
        }

        for frame_idx, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Convert to supervision Detections
            dets = sv.Detections.from_ultralytics(detection)

            # Map goalkeeper to player class id, if present
            for i, class_id in enumerate(dets.class_id):
                if cls_names[class_id] == "goalkeeper":
                    dets.class_id[i] = cls_names_inv.get("player", class_id)

            # Update tracker with detections (data association)
            tracked = self.tracker.update_with_detections(dets)

            # Ensure frame entries exist
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            # tracked entries: each entry typically contains bbox, score, cls_id, track_id
            for t in tracked:
                bbox = t[0].tolist()
                cls_id = t[3]
                track_id = t[4]

                if cls_id == cls_names_inv.get("player"):
                    tracks["players"][frame_idx][track_id] = {"bbox": bbox}
                elif cls_id == cls_names_inv.get("referee"):
                    tracks["referees"][frame_idx][track_id] = {"bbox": bbox}

            # add any ball detections from raw detections (class id may be different)
            for det in dets:
                bbox = det[0].tolist()
                cls_id = det[3]
                if cls_id == cls_names_inv.get("ball"):
                    # ball uses a fixed internal id of 1 per original code
                    tracks["ball"][frame_idx][1] = {"bbox": bbox}

        if stub_path:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame: np.ndarray, bbox: List[float], color: Tuple[int, int, int], track_id: Optional[int] = None) -> np.ndarray:
        """Draw an ellipse centered at the bottom-center of `bbox`.

        Optionally draws a filled rectangle with `track_id` text above the
        ellipse when `track_id` is provided.
        """
        y_bottom = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = max(1, get_bbox_width(bbox))

        cv2.ellipse(
            frame,
            center=(x_center, y_bottom),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4,
        )

        if track_id is not None:
            rectangle_width = 40
            rectangle_height = 20
            x1_rect = int(x_center - rectangle_width // 2)
            x2_rect = int(x_center + rectangle_width // 2)
            y1_rect = int((y_bottom - rectangle_height // 2) + 15)
            y2_rect = int((y_bottom + rectangle_height // 2) + 15)

            cv2.rectangle(frame, (x1_rect, y1_rect), (x2_rect, y2_rect), color, cv2.FILLED)

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(frame, f"{track_id}", (x1_text, y1_rect + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return frame

    def draw_triangle(self, frame: np.ndarray, bbox: List[float], color: Tuple[int, int, int]) -> np.ndarray:
        """Draw a small triangle marker above the bbox top-center.
        (Used to indicate ball or possession.)
        """
        y_top = int(bbox[1])
        x_center, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([[x_center, y_top], [x_center - 10, y_top - 20], [x_center + 10, y_top - 20]])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)
        return frame

    def draw_team_ball_control(self, frame: np.ndarray, frame_num: int, team_ball_control: np.ndarray) -> np.ndarray:
        """Render a small summary box showing ball-control percentages.

        `team_ball_control` is expected to be a 1D array-like where values
        indicate which team had control on each frame (e.g., 1 or 2).
        """
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        upto = team_ball_control[: frame_num + 1]
        team_1_count = int((upto == 1).sum())
        team_2_count = int((upto == 2).sum())
        total = team_1_count + team_2_count
        if total == 0:
            team_1_pct = team_2_pct = 0.0
        else:
            team_1_pct = team_1_count / total
            team_2_pct = team_2_count / total

        cv2.putText(frame, f"Team 1 Ball Control: {team_1_pct*100:.2f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2_pct*100:.2f}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        return frame

    def draw_annotations(self, video_frames: List[np.ndarray], tracks: Dict[str, List[Dict[int, Dict[str, Any]]]], team_ball_control: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """Draw annotations for all frames using the provided `tracks` structure.

        Returns a new list of annotated frames.
        """
        output_frames: List[np.ndarray] = []
        for frame_idx, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_idx]
            ball_dict = tracks["ball"][frame_idx]
            referee_dict = tracks["referees"][frame_idx]

            # Draw players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)
                if player.get("has_ball", False):
                    frame = self.draw_triangle(frame, player["bbox"], (0, 0, 255))

            # Draw referees
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            # Draw ball markers
            for _, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

            if team_ball_control is not None:
                frame = self.draw_team_ball_control(frame, frame_idx, team_ball_control)

            output_frames.append(frame)

        return output_frames