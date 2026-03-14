"""Small utility to run the YOLO model on a source and print/save results.

This module exposes `run_prediction()` which can be imported or executed
from the command line.
"""

from typing import Optional
import argparse
import logging

from ultralytics import YOLO


def run_prediction(model_path: str, source: str, project: str = "runs", name: str = "predict", save: bool = True, show: bool = False):
    """Run YOLO prediction on `source`.

    Args:
        model_path: path to YOLO model.
        source: input source (file path or camera index).
        project: run output folder.
        name: run name.
        save: whether to save results.
        show: whether to display frames.
    Returns:
        results object from Ultralytics API.
    """
    model = YOLO(model_path)
    results = model.predict(
        source=source,
        save=save,
        show=show,
        project=project,
        name=name,
    )
    return results


def main():
    parser = argparse.ArgumentParser(description="Run YOLO predictions")
    parser.add_argument("--model", default="best.pt")
    parser.add_argument("--source", default="input_videos/sample.mp4")
    parser.add_argument("--project", default="runs")
    parser.add_argument("--name", default="predict")
    parser.add_argument("--no-save", action="store_true", help="Do not save results")
    parser.add_argument("--show", action="store_true", help="Show results live")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info("Running model: %s on %s", args.model, args.source)
    results = run_prediction(args.model, args.source, project=args.project, name=args.name, save=not args.no_save, show=args.show)

    if results:
        logging.info("Result summary: %s", results[0])
        for box in results[0].boxes:
            print(box)


if __name__ == "__main__":
    main()