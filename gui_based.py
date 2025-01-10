import cv2
import csv
import os
from datetime import datetime
import time
import tkinter as tk
from tkinter import Button, Label, Scale, HORIZONTAL, filedialog, StringVar, Radiobutton, Entry, messagebox

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

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection GUI")
        self.root.geometry("275x450")
        self.cap = None
        self.running = False
        self.prev_frame_time = 0
        self.new_frame_time = 0
        self.img = None
        self.camera_type = StringVar(value="laptop")
        self.ip_address = StringVar()
        
        # Create layout using grid with padding (padx) for margins
        self.option_label = Label(root, text="Choose an option:")
        self.option_label.grid(row=0, column=0, columnspan=2, pady=10, padx=20, sticky="ew")

        self.start_camera_button = Button(root, text="Real-Time Detection", command=self.start_camera_option, width=20, bg='green', fg='white')
        self.start_camera_button.grid(row=1, column=0, columnspan=2, pady=5, padx=20, sticky="ew")

        self.upload_image_button = Button(root, text="Upload Image", command=self.upload_image_option, width=20, bg='blue', fg='white')
        self.upload_image_button.grid(row=2, column=0, columnspan=2, pady=5, padx=20, sticky="ew")

        # Camera selection
        self.camera_label = Label(root, text="Select Camera Type:")
        self.camera_label.grid(row=3, column=0, columnspan=2, padx=20, sticky="ew")
        self.laptop_camera_radio = Radiobutton(root, text="Laptop Camera", variable=self.camera_type, value="laptop")
        self.laptop_camera_radio.grid(row=4, column=0, columnspan=2, padx=20, sticky="w")
        self.ip_camera_radio = Radiobutton(root, text="IP Camera", variable=self.camera_type, value="ip")
        self.ip_camera_radio.grid(row=5, column=0, columnspan=2, padx=20, sticky="w")

        # IP Camera input field
        self.ip_label = Label(root, text="IP Address:")
        self.ip_label.grid(row=6, column=0, padx=20, sticky="w")
        self.ip_entry = Entry(root, textvariable=self.ip_address, width=20)
        self.ip_entry.grid(row=6, column=1, padx=20, sticky="e")

        # Slider for Confidence Threshold
        self.thres_label = Label(root, text="Confidence Threshold")
        self.thres_label.grid(row=7, column=0, columnspan=2, padx=20, sticky="ew")
        self.thres_slider = Scale(root, from_=0, to=1, resolution=0.01, orient=HORIZONTAL)
        self.thres_slider.set(0.45)
        self.thres_slider.grid(row=8, column=0, columnspan=2, padx=20, sticky="ew")

        # Slider for Non-Max Suppression (NMS)
        self.nms_label = Label(root, text="NMS Threshold")
        self.nms_label.grid(row=9, column=0, columnspan=2, padx=20, sticky="ew")
        self.nms_slider = Scale(root, from_=0, to=1, resolution=0.01, orient=HORIZONTAL)
        self.nms_slider.set(0.2)
        self.nms_slider.grid(row=10, column=0, columnspan=2, padx=20, sticky="ew")

        # Save Frame Button
        self.save_frame_button = Button(root, text="Save Frame", command=self.save_frame, width=20, bg='orange', fg='white')
        self.save_frame_button.grid(row=11, column=0, columnspan=2, pady=5, padx=20, sticky="ew")

        # Close Button
        self.close_button = Button(root, text="Close", command=self.close_app, width=20, bg='red', fg='white')
        self.close_button.grid(row=12, column=0, columnspan=2, pady=5, padx=20, sticky="ew")

    def start_camera_option(self):
        self.hide_options()
        self.start_camera()

    def upload_image_option(self):
        self.hide_options()
        self.upload_image()

    def hide_options(self):
        self.option_label.grid_forget()
        self.start_camera_button.grid_forget()
        self.upload_image_button.grid_forget()

    def show_options(self):
        self.option_label.grid(row=0, column=0, columnspan=2, pady=10, padx=20, sticky="ew")
        self.start_camera_button.grid(row=1, column=0, columnspan=2, pady=5, padx=20, sticky="ew")
        self.upload_image_button.grid(row=2, column=0, columnspan=2, pady=5, padx=20, sticky="ew")

    def start_camera(self):
        if not self.running:
            if self.camera_type.get() == "laptop":
                self.cap = cv2.VideoCapture(0)
            elif self.camera_type.get() == "ip":
                ip = self.ip_address.get()
                if not ip:
                    messagebox.showerror("Error", "Please enter an IP address.")
                    self.show_options()
                    return
                self.cap = cv2.VideoCapture(f"http://{ip}/video")
            else:
                messagebox.showerror("Error", "Invalid camera selection.")
                self.show_options()
                return

            self.cap.set(3, 1920)  # Set width to 1920 (Full HD)
            self.cap.set(4, 1080)  # Set height to 1080 (Full HD)
            self.running = True
            self.update_frame()

    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.show_options()  # Show options again when stopped

    def upload_image(self):
        file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            img = cv2.imread(file_path)
            if img is not None:
                # Resize the image to a fixed size (e.g., 640x480) before processing
                img = cv2.resize(img, (1920, 1080))
                thres = self.thres_slider.get()
                nms = self.nms_slider.get()
                self.img, _ = getObjects(img, thres, nms)  # Store processed image in self.img

                # Show the processed image in a fixed window size
                cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Detected Objects", 640, 480)
                cv2.imshow("Detected Objects", self.img)
    
                # Wait for a key press and close the image window
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                # Enable Save Frame button
                self.save_frame_button.config(state=tk.NORMAL)
        self.show_options()  # Show options again after processing image


    def update_frame(self):
        if self.running and self.cap.isOpened():
            success, img = self.cap.read()
            if success:
                # Resize the frame to a smaller size, e.g., 640x360
                img = cv2.resize(img, (1280, 720))
            
                thres = self.thres_slider.get()
                nms = self.nms_slider.get()
                self.img, _ = getObjects(img, thres, nms)

                # Add date and time to the top right of the frame
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                cv2.putText(self.img, current_time, (self.img.shape[1] - 350, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # Calculate and display FPS
                self.new_frame_time = time.time()
                fps = 1 / (self.new_frame_time - self.prev_frame_time)
                self.prev_frame_time = self.new_frame_time
                fps_text = f"FPS: {int(fps)}"
                cv2.putText(self.img, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                cv2.imshow("Real-Time Object Detection", self.img)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop_camera()
            self.root.after(10, self.update_frame)

    def save_frame(self):
        if self.img is not None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(outputDir, f"frame_{timestamp}.jpg")
            cv2.imwrite(save_path, self.img)
            print(f"Frame saved at: {save_path}")

    def close_app(self):
        self.stop_camera()
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()
