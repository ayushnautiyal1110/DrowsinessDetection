# <<<<<<< HEAD
# from scipy.spatial import distance as dist
# from imutils import face_utils
# import imutils
# import dlib
# import cv2
# import winsound
# frequency = 2500
# duration = 1000

# def eyeAspectRatio(eye):
#     A = dist.euclidean(eye[1], eye[5])
#     B = dist.euclidean(eye[2], eye[4])
#     C = dist.euclidean(eye[0], eye[3])
#     ear = (A + B) / (2.0 * C)
#     return ear

# count = 0
# earThresh = 0.3 #distance between vertical eye coordinate Threshold
# earFrames = 10 #consecutive frames for eye closure
# shapePredictor = "shape_predictor_68_face_landmarks.dat"

# cam = cv2.VideoCapture(0)
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(shapePredictor)

# #get the coord of left & right eye
# (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
# (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# while True:
#     _, frame = cam.read()
#     frame = imutils.resize(frame, width=1050)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     rects = detector(gray, 0)

#     for rect in rects:
#         shape = predictor(gray, rect)
#         shape = face_utils.shape_to_np(shape)

#         leftEye = shape[lStart:lEnd]
#         rightEye = shape[rStart:rEnd]
#         leftEAR = eyeAspectRatio(leftEye)
#         rightEAR = eyeAspectRatio(rightEye)

#         ear = (leftEAR + rightEAR) / 2.0

#         leftEyeHull = cv2.convexHull(leftEye)
#         rightEyeHull = cv2.convexHull(rightEye)
#         cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
#         cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)

#         if ear < earThresh:
#             count += 1
#             #print(count,earFrames)
#             if count >= earFrames:
#                 cv2.putText(frame, "DROWSINESS DETECTED", (10, 30),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#                 winsound.Beep(frequency,duration)
#         else:
#             count = 0

#     cv2.imshow("Frame", frame)
#     key = cv2.waitKey(1) & 0xFF

#     if key == ord("q"):
#         break

# cam.release()
# cv2.destroyAllWindows()

import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import winsound
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import dlib
from scipy.spatial import distance as cdist
from imutils import face_utils
import streamlit as st
import imutils
import dlib
import cv2
import winsound
# Define the threshold values for the eye aspect ratio (EAR)
earThresh = 0.3 #distance between vertical eye coordinate Threshold
earFrames = 10 #consecutive frames for eye closure

# Initialize the webrtc streamer
# Start streaming the webcam video
# streamer.start()
# # Display the video stream in the app
# st.video(streamer.video_feed)

frequency = 2500
duration = 1000
def eyeAspectRatio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear
shapePredictor = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shapePredictor)
def eyeAspectRatio(eye):
    A = cdist.euclidean(eye[1], eye[5])
    B = cdist.euclidean(eye[2], eye[4])
    C = cdist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear
def process_frame(frame):
    # Convert the frame to a NumPy array

    st.title("Drowsiness Detection")



    count = 0
    earThresh = 0.3 #distance between vertical eye coordinate Threshold
    earFrames = 30 #consecutive frames for eye closure
    shapePredictor = "shape_predictor_68_face_landmarks.dat"

    cam = cv2.VideoCapture(0)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shapePredictor)

    #get the coord of left & right eye
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    while True:
        _, frame = cam.read()
    # >>>>>>> 1a92a1c64eb719f473a8dd6d372cd67492237fa6
        frame = imutils.resize(frame, width=1050)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)
    # <<<<<<< HEAD
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    # =======

    # >>>>>>> 1a92a1c64eb719f473a8dd6d372cd67492237fa6
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eyeAspectRatio(leftEye)
            rightEAR = eyeAspectRatio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)

            if ear < earThresh:
                count += 1
                #print(count,earFrames)
                if count >= earFrames:
    # <<<<<<< HEAD
                    st.warning("Drowsiness detected!")
                    frequency=10
                    duration=5
                    # Play a sound
                    winsound.Beep(frequency, duration)
            else:
                count = 0

        


def calculate_ear(landmarks):
    # Get the left and right eye landmarks
    left_eye = landmarks[0][2:16]
    right_eye = landmarks[0][36:48]

    # Calculate the distance between the two eye centers
    horizontal_distance = distance(left_eye[0], right_eye[0])

    # Calculate the distance between the top and bottom eye centers
    vertical_distance = distance(left_eye[2], right_eye[2])

    # Calculate the EAR
    ear = horizontal_distance / vertical_distance

    return ear


def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


# Main function
if __name__ == "__main__":
    # Run the app
    streamer = webrtc_streamer(key="example", video_frame_callback=process_frame)
    # st.run()
# =======
                # cv2.putText(frame, "DROWSINESS DETECTED", (10, 30),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # winsound.Beep(frequency,duration)
        

        # cv2.imshow("Frame", frame)
        # key = cv2.waitKey(1) & 0xFF

        # if key == ord("q"):
        #     break

# cam.release()
# cv2.destroyAllWindows()


# >>>>>>> 1a92a1c64eb719f473a8dd6d372cd67492237fa6
