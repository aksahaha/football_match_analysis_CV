# ⚽ YOLO Football Player Detection and Tracking

This project implements a **computer vision pipeline for football match analysis** that detects and tracks **players, referees, and the ball** in match footage.  
The system uses **Ultralytics YOLO for object detection** and **ByteTrack (via the Supervision library)** for **multi-object tracking** to maintain consistent identities of players across frames.

The pipeline processes a football video, performs detection and tracking on every frame, assigns **persistent tracking IDs**, and generates an **annotated output video** showing detected objects and player identities.

---

# 📌 Key Features

- Detect **football players, referees, and the ball**
- Perform **multi-object tracking across frames**
- Assign **consistent tracking IDs**
- Automatically **cluster jersey colors to assign players to teams**
- Generate **annotated video output**
- Modular and extendable pipeline for sports analytics

---

# 🧠 System Overview

The pipeline processes a football video using the following sequence of steps:

```
Input Video
    │
Frame Extraction
    │
YOLO Object Detection
    │
Convert detections to Supervision format
    │
ByteTrack Multi-Object Tracking
    │
Team Assignment (Jersey Color Clustering)
    │
Annotated Video Output
```

Each detected object is assigned a **unique tracking ID** that remains consistent across frames.

Example:

```
Player #3 (Team 1)
Player #8 (Team 2)
Referee #1
Ball #10
```

This allows tracking player movement and analyzing gameplay.

---

# 📂 Project Structure

```
yolo_football
│
├── main.py
│   Main pipeline that runs detection, tracking, and team assignment
│
├── yolo_interface.py
│   Interface for loading and running the YOLO detection model
│
├── utils/
│   Utilities for reading and saving videos
│
├── trackers/
│   Tracking logic and training notebook
│   ├── tracker.py
│   └── training_notebook.ipynb
│
├── training/
│   Dataset and training artifacts
│
├── input_videos/
│   Sample input videos
│   └── sample.mp4
│
├── output_videos/
│   Generated output videos and images
│
├── models/
│   Trained YOLO model weights
│   └── best.pt
│
├── requirements.txt
│
└── README.md
```

---

# 📊 Dataset

The model was trained using an annotated dataset from **Roboflow Universe**.

Dataset link:

https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/1

### Dataset Classes

The dataset contains bounding box annotations for:

- Player
- Referee
- Ball

This enables the model to differentiate **players and referees from other people in the scene**.

---

# 🏋️ Model Training

Model training is performed using the notebook located in:

```
trackers/training_notebook.ipynb
```

### Training Workflow

1. Load annotated dataset from Roboflow
2. Convert annotations to YOLO format
3. Configure training parameters
4. Train YOLO detection model
5. Export the best trained weights

The best trained model is saved as:

```
models/best.pt
```

This model is then used for **object detection during inference**.

---

# 🔍 Object Detection

Object detection is performed using **Ultralytics YOLO**.

The model predicts bounding boxes for:

- Players
- Referees
- Ball

Each detection includes:

- Bounding box
- Class label
- Confidence score

Example detection output:

```
Player — Confidence: 0.92
Referee — Confidence: 0.87
Ball — Confidence: 0.81
```

---

# 🎯 Multi-Object Tracking

Tracking is implemented using **ByteTrack**, integrated through the **Supervision library**.

Tracking ensures that each detected object receives a **persistent ID** across frames.

Example:

```
Frame 1 → Player ID 7
Frame 2 → Player ID 7
Frame 3 → Player ID 7
```

This allows tracking **player movement over time**.

---

# 👕 Team Assignment

The system automatically assigns players to teams based on **jersey color clustering**.

### Method

1. Extract player bounding box
2. Sample jersey colors from the region
3. Apply clustering to separate teams
4. Assign team label to each player

Example:

```
Team 1 → Red
Team 2 → Blue
```

These colors are used to visualize team membership in the output video.

---

# 🎥 Output Visualization

The output video contains annotated detections including:

- Player bounding boxes
- Player tracking IDs
- Team colors
- Ball detection
- Referee detection

Players are visualized using **elliptical markers** with their **tracking ID**.

Example annotation:

```
Player #4  (Team 1 → Red)
Player #9  (Team 2 → Blue)
Referee #1
Ball
```

---

# 📷 Output Files

The pipeline generates the following outputs:

```
output_videos/output_video.avi
output_videos/ss_output.png
output_videos/image.png
```

### Output Video

The generated video includes:

- Object detections
- Tracking IDs
- Team color annotations

---

# ⚙️ Installation

## 1. Clone the Repository

```
git clone https://github.com/your-username/yolo_football.git
cd yolo_football
```

---

## 2. Create Virtual Environment

```
python -m venv .venv
```

---

## 3. Activate the Environment

### Windows PowerShell

```
.\.venv\Scripts\Activate.ps1
```

### Mac / Linux

```
source .venv/bin/activate
```

---

## 4. Install Dependencies

```
pip install -r requirements.txt
```

Main dependencies include:

- ultralytics
- supervision
- opencv-python
- numpy
- torch

---

# ▶️ Running the Project

Run the main pipeline:

```
python main.py
```

The system will:

1. Load the trained YOLO model
2. Process the input football video
3. Detect players, referees, and the ball
4. Track objects using ByteTrack
5. Assign teams based on jersey colors
6. Generate an annotated output video

The final video will be saved in:

```
output_videos/output_video.avi
```

---

# 🔧 Files Modified in This Version

### trackers/tracker.py

Updated to:

- Draw players using **team-specific colors**
- Use `team_color` as fallback if team label is missing

---

### main.py

Updated to run **team assignment logic** using `TeamAssigner`.

New fields added to player tracks:

```
team
team_color
```

These fields are used for **visualizing team membership**.

---

# 🚀 Future Improvements

Possible enhancements for the system:

- Player pose estimation
- Ball trajectory prediction
- Pass detection
- Player heatmaps
- Tactical formation analysis
- Expected goals (xG) modeling
- Real-time inference for live matches

---

# 🧪 Technologies Used

- Python
- Ultralytics YOLO
- ByteTrack
- Supervision
- OpenCV
- NumPy
- PyTorch

---

# 📜 License

This project is intended for **educational and research purposes**.

---

# 👨‍💻 Author

Abhishek Kumar
