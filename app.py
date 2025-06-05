from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO('best.pt')  # Load model YOLOv8

cap = cv2.VideoCapture(0)  # Mở webcam

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1) 
            # Nhận diện bằng YOLOv8
            results = model.predict(source=frame, imgsz=640, conf=0.5, verbose=False)
            annotated_frame = results[0].plot()

            # Mã hóa frame dưới dạng JPEG để stream lên web
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()

            # Trả frame dưới dạng chuỗi multipart
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
