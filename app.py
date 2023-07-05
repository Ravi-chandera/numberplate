import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from sort.sort import *
from util import get_car, read_license_plate, write_csv
from io import BytesIO
import base64


# Function to process video frames and generate results
def process_video(video_file):
    results = {}
    mot_tracker = Sort()

    # Load models
    coco_model = YOLO('yolov8n.pt')
    license_plate_detector = YOLO('./models/license_plate_detector.pt')

    # Open video file
    cap = cv2.VideoCapture(video_file.name)  # Use the name attribute of the video_file object
    frame_nmr = -1

    # Rest of the code...

    vehicles = [2, 3, 5, 7]

# Rest of the code...
    
    # Process video frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_nmr += 1
        results[frame_nmr] = {}

        # Detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # Track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # Detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:
                # Crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # Process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # Read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}

    # Release video capture
    cap.release()

    # Write results to CSV
    write_csv(results, './test.csv')

    return results

# Streamlit app
st.title('License Plate Tracking')

# Upload video file
video_file = st.file_uploader('Upload Video', type=['mp4'])

if video_file is not None:
    # Process video and display results
    st.text('Processing video...')
    results = process_video(video_file)
    st.text('Video processing completed!')

    # Display results
    st.subheader('Results')
    st.text('Total frames processed: ' + str(len(results)))

    # Display individual frames and license plate information
    # for frame_nmr, frame_results in results.items():
        # st.subheader('Frame ' + str(frame_nmr))
        # st.text('Total vehicles detected: ' + str(len(frame_results)))

        # Display individual vehicles and license plates
        # for car_id, car_info in frame_results.items():
        #     st.text('Car ID: ' + str(car_id))
        #     st.text('License Plate: ' + car_info['license_plate']['text'])
        #     st.text('License Plate Score: ' + str(car_info['license_plate']['text_score']))
        #     st.text('Vehicle Bounding Box: ' + str(car_info['car']['bbox']))

        # st.text('---')

with open('add_missing_data.py', 'r') as file:
    code = file.read()

exec(code)
with open('visualize.py', 'r') as file:
    code = file.read()

exec(code)

with open('showvideo.py', 'r') as file:
    code = file.read()

exec(code)





