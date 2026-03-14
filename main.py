"""Entry point for running the detection + tracking pipeline.

This script reads an input video, runs detection + tracking, draws
annotations, and writes an output video. It is intentionally small and
delegates work to `utils` and `trackers` modules.
"""

from typing import Optional
import argparse
import logging
import sys

from utils import read_video, save_video
from trackers import Tracker


def run_pipeline(input_path: str, output_path: str, model_path: str = "best.pt") -> None:
    """Run detect->track->annotate pipeline for a single video.

    Args:
        input_path: path to input video file.
        output_path: path to save annotated video.
        model_path: path to YOLO model checkpoint.
    """
    frames = read_video(input_path)
    tracker = Tracker(model_path)
    tracks = tracker.get_object_tracks(frames)
    annotated = tracker.draw_annotations(frames, tracks)
    save_video(annotated, output_path)


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Run YOLO football tracking pipeline")
    parser.add_argument("input", help="Input video path", nargs="?", default="input_videos/sample.mp4")
    parser.add_argument("output", help="Output video path", nargs="?", default="output_videos/output_video.avi")
    parser.add_argument("--model", help="YOLO model path", default="best.pt")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        logging.info("Starting pipeline: %s -> %s", args.input, args.output)
        run_pipeline(args.input, args.output, args.model)
        logging.info("Finished: output saved to %s", args.output)
    except Exception:
        logging.exception("Pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()