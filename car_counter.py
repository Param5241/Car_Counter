import cv2
import cvzone
import math
from ultralytics import YOLO

# Load video and YOLO model
video_path = '/Users/apple/Downloads/Azoca/08_01_2025/testData/test1.mp4'
video = cv2.VideoCapture(video_path)
model = YOLO('/Users/apple/Downloads/Azoca/08_01_2025/YOLO Weights/yolov8n.pt')  # Adjust path to your YOLO model if necessary

# Class names for YOLO
classNames = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
              'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
              'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
              'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
              'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
              'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
              'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 
              'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 
              'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
              'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 
              'toothbrush']

# Tracking dictionaries and counts
tracked_left = {}
tracked_right = {}
left_car_count = 0
right_car_count = 0

# Define line coordinates
left_line_start = (170, 510)
left_line_end = (580, 510)
right_line_start = (725, 550)
right_line_end = (1100, 550)

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Get YOLO detections
    results = model(frame, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            # Check for 'car' with confidence > 0.3
            if classNames[cls] == 'car' and conf > 0.5:
                # Draw bounding box
                cvzone.cornerRect(frame, (x1, y1, w, h), l=9)
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                # print(center_x,center_y)
                # Left line detection
                if left_line_start[1]-4 < center_y <= left_line_end[1]+2 and left_line_start[0]-7 <= center_x <= left_line_end[0]:
                    if center_x == 305 and center_y == 510:
                        continue
                    else:
                        if center_x not in tracked_left:  # Check if not already counted
                            left_car_count += 1
                            tracked_left[center_x] = True

                # Right line detection
                if right_line_start[1]-4 < center_y <= right_line_end[1]+7 and right_line_start[0]-7 < center_x <= right_line_end[0]+7:
                    if center_x not in tracked_right:  # Check if not already counted
                        right_car_count += 1
                        tracked_right[center_x] = True

    # Draw lines
    cv2.line(frame, left_line_start, left_line_end, (0, 0, 255), 5)  # Left line
    cv2.line(frame, right_line_start, right_line_end, (0, 255, 0), 5)  # Right line

    # Display car counts
    cv2.putText(frame, f"Outgoing Cars Count : {left_car_count}", (160, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Incoming Cars Count : {right_car_count}", (740, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Total Cars Count : {right_car_count+left_car_count}", (580, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Show frame
    cv2.imshow('Car Counter', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video.release()
cv2.destroyAllWindows()