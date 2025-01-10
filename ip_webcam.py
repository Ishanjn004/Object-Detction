import cv2

# Threshold to detect object
thres = 0.45

# Load class names
classNames = []
classFile = r"C:\Users\ishan\OneDrive\Desktop\Object_Detection_Files\coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = r"C:\Users\ishan\OneDrive\Desktop\Object_Detection_Files\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = r"C:\Users\ishan\OneDrive\Desktop\Object_Detection_Files\frozen_inference_graph.pb"

# Initialize DNN model
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    if len(objects) == 0:
        objects = classNames
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

    return img, objectInfo

if __name__ == "__main__":
    # Replace with your IP webcam URL
    ip_camera_url = "http://192.168.0.101:8080/video"  # Example URL
    cap = cv2.VideoCapture(ip_camera_url)

    while True:
        success, img = cap.read()
        if not success:
            print("Error: Could not read image.")
            break
        result, objectInfo = getObjects(img, thres, 0.2)
        cv2.imshow("Output", img)
        
        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
