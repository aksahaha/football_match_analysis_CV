# YOLO Football

This project detects and tracks **football players, referees, and the ball** in match footage using **Ultralytics YOLO** for object detection and **ByteTrack (via Supervision)** for multi-object tracking. The system processes video frames, detects objects, assigns consistent tracking IDs, and generates an annotated output video.

---

## Project Structure

yolo_football  
├── main.py  
├── yolo_interface.py  
├── utils/  (Video reading and saving utilities)  
├── trackers/  (Tracking logic and training notebook)  
│   └── training_notebook.ipynb  
├── training/  (Dataset and training artifacts)  
├── input_videos/  
│   └── sample.mp4  
├── output_videos/  
└── models/  
  └── best.pt (Trained YOLO model)

---

## Dataset

The model was trained using an annotated dataset from Roboflow:

https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/1

The dataset contains labeled bounding boxes for:

- Players  
- Referees  
- Ball  

This allows the model to distinguish **players and referees from other people in the scene**.

---

## Model Training

Inside the **trackers** folder there is a notebook used for training the YOLO model.  
The notebook loads the annotated dataset, converts it into YOLO format, trains the model, and exports the best trained weights.

The trained model is saved as:

models/best.pt

This trained model is then used in the detection and tracking pipeline.

---

## Pipeline

Video  
→ Frame Extraction  
→ YOLO Detection  
→ Convert to Supervision Detection format  
→ ByteTrack Multi-Object Tracking  
→ Annotated Output Video  

The tracker assigns consistent IDs to players and referees across frames.

Example:

Player #1  
Player #5  
Referee #2  
Ball #9  

---

## Setup

Create and activate a virtual environment:

python -m venv .venv  
.\.venv\Scripts\Activate.ps1  
pip install -r requirements.txt  

---

## Run the Project

python main.py

The processed video will be saved to:

output_videos/output_video.avi

---

## Notes

- The model checkpoint path can be modified in `main.py`.
- All dependencies required for the project are listed in `requirements.txt`.
- The tracker combines YOLO detection with ByteTrack tracking to maintain object IDs across frames.
