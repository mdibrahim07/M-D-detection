import cv2

# Load pre-trained classifiers with the complete paths
face_cascade = cv2.CascadeClassifier("C:/Users/HP/OneDrive/Desktop/Mask Detection/haarcascade_frontalface_default.xml")
mouth_cascade = cv2.CascadeClassifier("C:/Users/HP/OneDrive/Desktop/Mask Detection/mouth.xml")


# Open a connection to the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        face_roi = gray[y:y + h, x:x + w]
        mouth_rects = mouth_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))
        
        label = "Mask"
        color = (0, 255, 0)
        
        # Check for mouth detection (No mask if mouth is uncovered)
        if len(mouth_rects) > 0:
            for (mx, my, mw, mh) in mouth_rects:
                if y + my > y + h // 2:  # Mouth is typically below the middle of the face
                    label = "No Mask"
                    color = (0, 0, 255)
                    cv2.rectangle(frame, (x + mx, y + my), (x + mx + mw, y + my + mh), color, 2)
                    break  # Exit after detecting the first uncovered mouth area

        # Display the label (Mask/No Mask) on the image
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    cv2.imshow("Live Mask Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
