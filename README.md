# YOLO Football

This repository contains utilities for detecting and tracking football players, referees and the ball using Ultralytics YOLO and ByteTrack (via `supervision`).

Structure
- `main.py`, `yolo_interface.py`: top-level scripts to run the pipeline.
- `utils/`: helper utilities (video I/O, bbox helpers).
- `trackers/`: tracking and annotation functionality.
- `training/`: datasets and training artefacts.

Quickstart
1. Create and activate a Python virtual environment (Windows):

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the main script (example):

```powershell
py main.py
```

Notes
- The tracker uses a YOLO model checkpoint; change the model path in `main.py` or when creating a `Tracker` instance.
- `requirements.txt` lists the primary dependencies used in the project.

If you'd like a full project reorganization (move modules to `src/` or add packaging), tell me and I will scaffold it.
