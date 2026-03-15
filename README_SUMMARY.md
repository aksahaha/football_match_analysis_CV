## Brief summary of changes and output

- What the code does: Detects players, referees and the ball using a YOLO model and performs multi-object tracking (ByteTrack via supervision). It assigns consistent track IDs and attempts to cluster player shirt colors to assign each player to a team.
- Output: an annotated video `output_videos/output_video.avi` and a screenshot `output_videos/ss_output.png` (sample frame). Players are annotated with an ellipse and their track id; players assigned to team 1 are drawn red, team 2 drawn blue.

Files changed in this update:
- `trackers/tracker.py` — draw colors based on `team` field and fallback to `team_color`.
- `main.py` — run team assignment using `team_assigner.TeamAssigner` and attach `team` and `team_color` fields to player tracks before drawing.

How to reproduce locally:
1. Activate the venv: `& .venv\Scripts\Activate.ps1`
2. Ensure dependencies are installed (PyTorch + others in `requirements.txt`).
3. Run: `python main.py`

Location of screenshot and output video:
- `output_videos/output_video.avi`
- `output_videos/image.png` (existing sample frame)
