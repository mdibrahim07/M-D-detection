import tkinter as tk
from tkinter import Label
import cv2
from PIL import Image, ImageTk

# Load pre-trained classifiers
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
mouth_cascade = cv2.CascadeClassifier("mouth.xml")

# Create the main window
root = tk.Tk()
root.title("Mask Detection")

# Set the window size and background color
root.geometry("800x600")
root.configure(bg="black")  # Set the background color of the window to black

# Add a label for the title
title = Label(root, text="Live Mask Detection", font=("Helvetica", 16), bg="black", fg="white")
title.pack()

# Create a label to display the webcam feed
video_label = Label(root, bg="black")  # Set the background of the video label to black
video_label.pack()

# Function to update the video feed and detect masks
def update_video():
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            face_roi = gray[y:y + h, x:x + w]
            mouth_rects = mouth_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))
            
            label = "Mask"
            color = (0, 255, 0)
            
            if len(mouth_rects) > 0:
                for (mx, my, mw, mh) in mouth_rects:
                    if y + my > y + h // 2:
                        label = "No Mask"
                        color = (0, 0, 255)
                        cv2.rectangle(frame, (x + mx, y + my), (x + mx + mw, y + my + mh), color, 2)
                        break
            
            # Add the label for Mask or No Mask
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Convert the frame to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the frame to a format Tkinter understands
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img)

        # Update the image on the label
        video_label.img_tk = img_tk
        video_label.config(image=img_tk)
        
    # Update the video feed every 10 ms
    video_label.after(10, update_video)

# Start the webcam feed
cap = cv2.VideoCapture(0)

# Call the update_video function to start updating the webcam feed
update_video()

# Start the GUI loop
root.mainloop()
