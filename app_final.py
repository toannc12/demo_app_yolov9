import os
import cv2
from flask import Flask, render_template, Response, request, redirect
from ultralytics import YOLO
from PIL import Image
from werkzeug.utils import secure_filename
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'
# Load the YOLOv8 model
# model = YOLO("best_yolov9_2.pt")
# model = YOLO("best_yolov9_1.pt")
# model = YOLO("best_yolov8.pt")
# model = YOLO("best_yolov5.pt")

# Export to tensorRT (.engine file)
# model.export(format="engine", half = True)
tensorrt_model = YOLO("best_yolov9_2.engine", task="detect")

@app.route('/')
def home():
    return render_template('index_final.html')

def detect_objects_from_webcam():
    count=0
    cap = cv2.VideoCapture(0)  # 0 for the default webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1  # Increment the global count
        if count % 2 != 0:
           continue
        # Resize the frame to (1020, 600)
        frame = cv2.resize(frame, (1020, 600))
        
        # Run YOLOv9 tracking on the frame
        results = tensorrt_model.track(frame, persist=True)

        annotated_frame = results[0].plot()
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/webcam_feed')
def webcam_feed():
    return Response(detect_objects_from_webcam(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# # =========================== VIDEO ===========================
@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    # Save the uploaded file to the uploads folder
    # if not os.path.exists(app.config['UPLOAD_FOLDER']):
    #     os.makedirs(app.config['UPLOAD_FOLDER'])
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploads', file.filename)
    file.save(file_path)

    # Redirect to the video playback page after upload
    return render_template('index_final.html', filename=file.filename)

def detect_objects_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'results', f"result_tensorrt1_{os.path.basename(video_path)}")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = 720
    height = 480
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    total_fps = 0
    total_time = 0
    count=0
    color_map = {
        0: (235, 51, 0),    # D00: Longitudinal cracks - Green
        1: (241, 212, 21),    # D10: Transverse cracks - Red
        2: (238, 233, 234),    # D20: Alligator cracks - Blue
        3: (190, 222, 12),  # D30: Repaired cracks - Yellow
        4: (108, 29, 19),  # D40: Potholes - Cyan
        5: (221, 114, 244),  # D50: Pedestrian crossing blurs - Orange
        6: (79, 64, 255),  # D60: Lane line blurs - Magenta
        7: (2, 237, 204),  # D70: Manhole covers - Purple
        8: (54, 251, 0)   # D80: Patchy road sections - Gold
    }

    while cap.isOpened():
        ret, frame = cap.read()
        # check the code may attempt to process None frames after the video ends, causing errors
        if not ret:
            break
        count += 1
        if frame is None: 
            break
        
        # Resize the frame to (1020, 600)  
        frame = cv2.resize(frame, (720, 480))
        # start_time = time.time()

        # Run YOLOv9 tracking on the frame
        # results = model.predict(frame)
        results = tensorrt_model.predict(frame, device = 0)

        annotated_frame = frame.copy()

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            conf = box.conf[0]  # Confidence score
            cls = int(box.cls[0])  # Class index
            label = f'D{cls}0 {conf:.2f}'  # Label without ID

            # Get color for the class, default to white if class not in map
            color = color_map.get(cls, (255, 255, 255))    

            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            # Add label text
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # compute fps
        # current_time = time.time()
        # fpss = 1.0/(current_time - start_time)
        # start_time = current_time
        # total_fps = total_fps + fpss
        # avg_fps = total_fps/count

        # annotated_frame = results[0].plot()
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        out.write(annotated_frame)    
    
    # print("Average fps: " + str(int(avg_fps)))

@app.route('/video_feed/<filename>')
def video_feed(filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploads',filename)
    return Response(detect_objects_from_video(video_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# =========================== IMAGE ===========================
@app.route('/imgpred', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'image' not in request.files:
            return redirect(request.url)

        file = request.files['image']

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return redirect(request.url)

        if file:
            # Save the uploaded image to a temporary location
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploads', filename)
            file.save(image_path)
            # Run inference on the uploaded image
            results = tensorrt_model(image_path)  # results list

            # Visualize the results
            for i, r in enumerate(results):
                # Plot results image
                im_bgr = r.plot()  # BGR-order numpy array
                im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

                # Save the processed image
                result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'results', f"result_{os.path.basename(image_path)}")

                im_rgb.save(result_image_path)    

            # Remove the uploaded image
            # os.remove(image_path)

            # Render the HTML template with the result image path
            return render_template('index_final.html', result_image_path=result_image_path, image_path=image_path)
    
    # If no file is uploaded or GET request, render the form
    return render_template('index_final.html', image_path=None)

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=False, port=8080, threaded = True)