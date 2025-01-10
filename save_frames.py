import cv2
import csv
import os
from datetime import datetime
import time

classNames = []
classFile = r"C:\Users\ishan\OneDrive\Desktop\Object_Detection_Files\coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = r"C:\Users\ishan\OneDrive\Desktop\Object_Detection_Files\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = r"C:\Users\ishan\OneDrive\Desktop\Object_Detection_Files\frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Create a directory to save frames
outputDir = 'saved_frames'
if not os.path.exists(outputDir):
    os.makedirs(outputDir)

logFile = 'detections_log.csv'
with open(logFile, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Timestamp', 'Class Name', 'Confidence', 'Bounding Box'])

def logDetection(logFile, timestamp, className, confidence, box):
    with open(logFile, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, className, round(confidence * 100, 2), box])

def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    if len(objects) == 0: objects = classNames
    objectInfo = []
    
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className])
                if draw:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                # Log the detection event
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                logDetection(logFile, timestamp, className, confidence, box)

    return img, objectInfo

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3, 1920)  # Set width to 1920 (Full HD)
    cap.set(4, 1080)  # Set height to 1080 (Full HD)

    # Variables for FPS calculation
    fps = 0
    prev_time = time.time()

    # Create a window
    window_name = "Output"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Create a resizable window

    # Center the window
    screen_width = 1920  # Adjust to your screen width
    screen_height = 1080  # Adjust to your screen height
    window_width = 1600   # Adjust to your window width
    window_height = 900    # Adjust to your window height

    # Resize window to desired size
    cv2.resizeWindow(window_name, window_width, window_height)  
    
    # Calculate the position to center the window
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    cv2.moveWindow(window_name, x, y)  # Move the window to the center

    while True:
        success, img = cap.read()
        if not success:
            print("Error: Could not read image.")
            break
        
        result, objectInfo = getObjects(img, 0.45, 0.2)

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # Display FPS on the frame
        cv2.putText(img, f'FPS: {int(fps)}', (10, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow(window_name, img)
        
        # Check for key presses to save the frame or quit
        key = cv2.waitKey(1) & 0xFF  # Check for key press
        if key == ord('s'):  # Press 's' to save the frame
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            framePath = os.path.join(outputDir, f'frame_{timestamp}.jpg')
            cv2.imwrite(framePath, img)  # Save the frame
            print(f"Frame saved: {framePath}")
        elif key == ord('q'):  # Press 'q' to quit
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
