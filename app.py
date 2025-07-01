from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
import numpy as np
from moviepy.editor import VideoFileClip

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploaded_videos'
OUTPUT_FOLDER = 'static/output_videos'
WEB_FOLDER = 'static/output_web'
MODEL_PATH = 'model/yolov8n.pt'

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(WEB_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['WEB_FOLDER'] = WEB_FOLDER

model = YOLO(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return 'No file uploaded.'

    video = request.files['video']
    if video.filename == '':
        return 'No selected file.'

    filename = secure_filename(video.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video.save(input_path)

    output_path = os.path.join(app.config['OUTPUT_FOLDER'], 'output_' + filename)
    final_output_path = process_video(input_path, output_path)

    return render_template('result.html', video_url=final_output_path)

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    prev_positions = {}
    scale_factor = 0.05

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        for result in results[0].boxes:
            x1, y1, x2, y2 = result.xyxy[0].cpu().numpy()
            class_id = int(result.cls.cpu().item())

            if class_id == 2:  # Car class
                car_id = str(int(x1))
                curr_bbox = (x1, y1, x2, y2)

                if car_id in prev_positions:
                    speed = estimate_speed(prev_positions[car_id], curr_bbox, fps, scale_factor)
                    cv2.putText(frame, f"Speed: {speed:.2f} km/h", (int(x1), int(y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                prev_positions[car_id] = curr_bbox

        out.write(frame)

    cap.release()
    out.release()

    return convert_to_web_compatible(output_path)

def convert_to_web_compatible(input_path):
    filename = os.path.basename(input_path)
    output_path = os.path.join(app.config['WEB_FOLDER'], 'web_' + filename)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    video_clip = VideoFileClip(input_path)
    video_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

    return output_path

def estimate_speed(prev_bbox, curr_bbox, fps, scale_factor):
    prev_center = ((prev_bbox[0] + prev_bbox[2]) / 2, (prev_bbox[1] + prev_bbox[3]) / 2)
    curr_center = ((curr_bbox[0] + curr_bbox[2]) / 2, (curr_bbox[1] + curr_bbox[3]) / 2)
    distance = np.sqrt((curr_center[0] - prev_center[0]) ** 2 + (curr_center[1] - prev_center[1]) ** 2)
    return (distance * scale_factor) * fps * 3.6

if __name__ == '__main__':
    app.run(debug=True)
