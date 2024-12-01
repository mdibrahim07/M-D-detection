# import RPi.GPIO as GPIO
# import time
# from scipy.spatial import distance as dist
# from imutils.video import VideoStream
# from imutils import face_utils
# import argparse
# import imutils
# import dlib
# import cv2
# from threading import Thread

# # Buzzer setup
# buzzer_pin = 23  # GPIO pin for the buzzer
# GPIO.setmode(GPIO.BCM)
# GPIO.setup(buzzer_pin, GPIO.OUT)

# # Function to beep the buzzer
# def beep_buzzer():
#     while alarm_status:  # Continuously beep while alarm is active
#         GPIO.output(buzzer_pin, GPIO.HIGH)
#         time.sleep(0.2)  # Beep duration
#         GPIO.output(buzzer_pin, GPIO.LOW)
#         time.sleep(0.2)  # Silence duration

# # Function to calculate Eye Aspect Ratio (EAR)
# def eye_aspect_ratio(eye):
#     A = dist.euclidean(eye[1], eye[5])
#     B = dist.euclidean(eye[2], eye[4])
#     C = dist.euclidean(eye[0], eye[3])
#     ear = (A + B) / (2.0 * C)
#     return ear

# # Function to calculate EAR for both eyes
# def final_ear(shape):
#     (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
#     (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
#     leftEye = shape[lStart:lEnd]
#     rightEye = shape[rStart:rEnd]
#     leftEAR = eye_aspect_ratio(leftEye)
#     rightEAR = eye_aspect_ratio(rightEye)
#     ear = (leftEAR + rightEAR) / 2.0
#     return ear

# ap = argparse.ArgumentParser()
# ap.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam on system")
# args = vars(ap.parse_args())

# EYE_AR_THRESH = 0.25
# EYE_AR_CONSEC_FRAMES = 48  # Number of consecutive frames for detecting drowsiness
# alarm_status = False
# COUNTER = 0

# print("-> Loading the predictor and detector...")
# detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# print("-> Starting Video Stream")
# vs = VideoStream(src=args["webcam"]).start()
# time.sleep(1.0)

# try:
#     while True:
#         frame = vs.read()
#         frame = imutils.resize(frame, width=450)
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

#         for (x, y, w, h) in rects:
#             rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
#             shape = predictor(gray, rect)
#             shape = face_utils.shape_to_np(shape)

#             ear = final_ear(shape)

#             # Check for drowsiness
#             if ear < EYE_AR_THRESH:
#                 COUNTER += 1
#                 if COUNTER >= EYE_AR_CONSEC_FRAMES:
#                     if not alarm_status:
#                         alarm_status = True
#                         # Start the buzzer in a separate thread
#                         t = Thread(target=beep_buzzer)
#                         t.daemon = True
#                         t.start()
#                     cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#             else:
#                 COUNTER = 0
#                 alarm_status = False  # Stop the buzzer
#                 GPIO.output(buzzer_pin, GPIO.LOW)  # Ensure buzzer is off

#             cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#         cv2.imshow("Frame", frame)

#         key = cv2.waitKey(1) & 0xFF
#         if key == ord("q"):
#             break
# finally:
#     GPIO.cleanup()  # Clean up GPIO settings
#     cv2.destroyAllWindows()
#     vs.stop()

from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import RPi.GPIO as GPIO

# Set up GPIO for the buzzer
buzzer_pin = 23  # GPIO pin where the buzzer is connected
GPIO.setmode(GPIO.BCM)
GPIO.setup(buzzer_pin, GPIO.OUT)

# Global variables for controlling the buzzer
buzzer_active = False

def sound_buzzer():
    while buzzer_active:  # Keep beeping while the condition is true
        GPIO.output(buzzer_pin, GPIO.HIGH)
        time.sleep(0.1)
        GPIO.output(buzzer_pin, GPIO.LOW)
        time.sleep(0.1)

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))
    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))
    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)
    distance = abs(top_mean[1] - low_mean[1])
    return distance

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam on system")
args = vars(ap.parse_args())

EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 70
YAWN_THRESH = 15
EYE_SLIGHTLY_CLOSED_THRESH = 0.25
YAWN_COUNT_THRESHOLD = 3
COUNTER = 0
yawn_counter = 0

print("-> Loading the predictor and detector...")
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

print("-> Starting Video Stream")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

try:
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in rects:
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            ear, leftEye, rightEye = final_ear(shape)
            distance = lip_distance(shape)

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            lip = shape[48:60]
            cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

            # Drowsiness detection
            if ear < EYE_AR_THRESH or (ear < EYE_SLIGHTLY_CLOSED_THRESH and distance > YAWN_THRESH):
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES or yawn_counter > YAWN_COUNT_THRESHOLD:
                    if not buzzer_active:
                        buzzer_active = True
                        t = Thread(target=sound_buzzer)
                        t.daemon = True
                        t.start()
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                COUNTER = 0
                yawn_counter = 0
                buzzer_active = False

            # Yawn detection
            if ear < EYE_SLIGHTLY_CLOSED_THRESH and distance > YAWN_THRESH:
                yawn_counter += 1
                cv2.putText(frame, "YAWN ALERT!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Display EAR and Lip Distance
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

finally:
    buzzer_active = False
    GPIO.cleanup()
    cv2.destroyAllWindows()
    vs.stop()




