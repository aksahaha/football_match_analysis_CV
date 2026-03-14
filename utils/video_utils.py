import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)

    cap.release()

    if not frames:
        raise ValueError(f"No frames were read from video: {video_path}")

    return frames


def save_video(output_video_frames, output_video_path):
    if not output_video_frames:
        raise ValueError("No frames to save. 'output_video_frames' is empty.")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(
        output_video_path,
        fourcc,
        24,
        (output_video_frames[0].shape[1], output_video_frames[0].shape[0])
    )

    if not out.isOpened():
        raise OSError(f"Could not create output video file: {output_video_path}")

    for frame in output_video_frames:
        out.write(frame)

    out.release()