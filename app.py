from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import tensorflow as tf
import json
import os
import mediapipe as mp

app = Flask(_name_)

# Load trained model
model = tf.keras.models.load_model("model/gesture_model.h5")

# Load label map
with open("model/label_map.json", "r", encoding="utf-8") as f:
    label_map = json.load(f)
label_map = {int(k): v for k, v in label_map.items()}  # Convert keys to int

# Global variables
cap = None
current_prediction = "Waiting..."

# MediaPipe
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def update_prediction(label):
    global current_prediction
    current_prediction = label
    print("🔍 Prediction updated to:", label)

def preprocess(frame):
    img = cv2.resize(frame, (64, 64))
    img = img / 255.0
    img = img.reshape(1, 64, 64, 3)
    return img

def gen_frames():
    global cap
    while True:
        if cap is None or not cap.isOpened():
            break

        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                h, w, _ = frame.shape
                x_min, y_min, x_max, y_max = w, h, 0, 0

                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)

                padding = 20
                x_min = max(x_min - padding, 0)
                y_min = max(y_min - padding, 0)
                x_max = min(x_max + padding, w)
                y_max = min(y_max + padding, h)

                roi = frame[y_min:y_max, x_min:x_max]
                if roi.size == 0:
                    continue

                try:
                    img = preprocess(roi)
                    prediction = model.predict(img)
                    class_id = int(np.argmax(prediction))
                    label = label_map.get(class_id, "Unknown")
                    update_prediction(label)

                    # Overlay on webcam frame
                    cv2.putText(frame, f"Prediction: {label}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                except Exception as e:
                    print("⚠ Error:", e)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start')
def start_camera():
    global cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        print("📷 Camera started.")
    return jsonify({'status': 'started'})

@app.route('/stop')
def stop_camera():
    global cap
    if cap:
        cap.release()
        cap = None
        print("🛑 Camera stopped.")
    update_prediction("Stopped ❌")
    return jsonify({'status': 'stopped'})

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction():
    return jsonify({'label': current_prediction})

if _name_ == '_main_':
    app.run(debug=True)