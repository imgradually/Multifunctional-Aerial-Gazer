from math import frexp
from os import name
from re import T
from tkinter.tix import Tree
import cv2
import numpy as np
import time
import copy as cp
import math
from djitellopy import Tello
import face_recognition
from my_head_pose_estimation import head_left_right
from gesture_recognition import gesture_recognition

# device = 0 # 0 for laptop, 1 for webcam
drone = Tello()

# Create arrays of known face encodings and their names
my_image = face_recognition.load_image_file("./0616057.jpg")
my_face_encoding = face_recognition.face_encodings(my_image)[0]
my_face_name = "Lion"

def camera_calibration(storage_path="./"):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

    # Chessboard size
    row = 6
    column = 9
    # take frame_num photos for calibration
    frame_num = 8
    i = 0
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ...., (6,5,0)
    objp = np.zeros((row * column, 3), np.float32)
    objp[:, :2] = np.mgrid[0:row, 0:column].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # cap = cv2.VideoCapture(device) # 0 is laptop, 1 is webcam
    while(True):

        frame_read = drone.get_frame_read()
        frame = frame_read.frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret2, corners = cv2.findChessboardCorners(gray_frame, (row, column), None)

        # If found, add object points, image points (after refining them)
        if ret2 == True:
            optimized_corners = cv2.cornerSubPix(gray_frame, corners, (11, 11), (-1, -1), criteria)

            objpoints.append(objp)
            imgpoints.append(optimized_corners)

            i = i + 1
            if (i == frame_num):
                break
            
            time.sleep(4)

        cv2.imshow('frame', frame)
        cv2.waitKey(100)

    # calculate the parameters
    retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_frame.shape[::-1], None, None)

    # store the parameters
    f = cv2.FileStorage(storage_path + "parameters0.xml", cv2.FILE_STORAGE_WRITE)
    f.write("intrinsic", cameraMatrix)
    f.write("distortion", distCoeffs)
    f.release()

    print('calibration completed!')

def face_detection_and_recognition(frame, intrinsic, distortion):
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]
    
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    names = ["Unknown" for i in range(len(face_locations))]
    
    if len(names) == 0:
        return frame, (None, None, None, None), 0

    face_distances = face_recognition.face_distance(face_encodings, my_face_encoding)
    best_match_index = np.argmax(face_distances)
    names[best_match_index] = my_face_name
    if np.max(face_distances) < 0.76:
        names[best_match_index] = "Unknown"


    best_coordinate = (None, None, None, None)
    best_distance = None
    for (top, right, bottom, left), name in zip(face_locations, names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        x1, y1, x2, y2 = left, top, right, bottom

        # Draw a box around the face
        if name == "Lion":
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # pose estimation
        imgpoints= np.array([(x1, y1), (x1, y2), (x2, y2), (x2, y1)], np.float32)
        objpoints = np.array([(0, 0, 0), (0, 18, 0), (18, 18, 0), (18, 0, 0)], np.float32)
        retval, rvec, tvec = cv2.solvePnP(objpoints, imgpoints, intrinsic, distortion)

        # Draw a label with a name below the face
        if name == "Lion":
            best_coordinate = (x1, y1, x2, y2)
            best_distance = round(tvec[2][0])
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
        else:
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        # cv2.putText( img, text to show, coordinate, font, font size, color, line width, line type )
        font = cv2.FONT_HERSHEY_DUPLEX
        text = name + "," + str(round(tvec[2][0], 2)) if name == "Lion" else str(round(tvec[2][0], 2))
        cv2.putText(frame, text, (x1 + 6, y2 - 6), font, 1.0, (255, 255, 255), 1)

    return frame, best_coordinate, best_distance

global_dist = 100
sleeptime = 0.3


def distancewithpeople(distance):
    global global_dist, sleeptime
    move = False
    if distance >= global_dist + 20:
        move = True
        drone.move_forward(20)
    elif distance < global_dist - 20:
        move = True
        drone.move_back(20)
    time.sleep(sleeptime)
    return move

def center(coordinate):
    global sleeptime
    move = False
    cur_x, cur_y = (coordinate[0]+coordinate[2])/2, (coordinate[1]+coordinate[3])/2
    if cur_y < 120:
        move = True
        drone.move_up(20)
    elif cur_y > 600:
        move = True
        drone.move_down(20)
    time.sleep(sleeptime)

    if cur_x > 640:
        move = True
        drone.move_left(20)
    elif cur_x < 320:
        move = True
        drone.move_right(20)
    time.sleep(sleeptime)
    return move

def movement(direction):
    print(direction)
    global sleeptime
    if direction == "Left":
        drone.rotate_counter_clockwise(30)
        time.sleep(sleeptime)
        drone.move_right(50)
        time.sleep(sleeptime)
    elif direction == "Right":
        drone.rotate_clockwise(30)
        time.sleep(sleeptime)
        drone.move_left(50)
        time.sleep(sleeptime)

def main():
    global global_dist, sleeptime

    drone.connect()
    time.sleep(sleeptime)
    drone.streamoff()
    time.sleep(sleeptime)
    drone.streamon()
    time.sleep(sleeptime)

    # calibration and read the parameters
    # camera_calibration()
    cv_file = cv2.FileStorage("./parameters0.xml", cv2.FILE_STORAGE_READ)
    intrinsic = cv_file.getNode("intrinsic").mat()
    distortion = cv_file.getNode("distortion").mat()
    cv_file.release()

    process_this_frame = True
    drone.takeoff()
    time.sleep(sleeptime)

    cur_height = drone.get_height()
    if(cur_height <= 150):
        time.sleep(sleeptime)
        drone.move_up(150-cur_height)
        time.sleep(sleeptime)
    # cap = cv2.VideoCapture(0)
    while(True):
        frame_read = drone.get_frame_read()
        frame = frame_read.frame
        
        cur_height = drone.get_height()
        time.sleep(sleeptime)

        # success, frame = cap.read()
        if process_this_frame:

            frame, hand_gesture = gesture_recognition(frame)

            if("YA" in hand_gesture): 
                cv2.imshow('frame', frame)
                key = cv2.waitKey(30) & 0xff
                continue
            elif("Back Seven" in hand_gesture):
                global_dist += 20
                print("now dist =", global_dist)
            elif("Front Seven" in hand_gesture):
                global_dist -= 20
                print("now dist =", global_dist)

            frame, coordinate, distance = face_detection_and_recognition(frame, intrinsic, distortion)

            if coordinate[0] == None:
                print("No face!")

            cv2.imshow('frame', frame)
            key = cv2.waitKey(30) & 0xff
            if key == 27:
                break

            if coordinate[0] != None:
                move1 = center(coordinate)
                move2 = distancewithpeople(distance)
                if move1 is False and move2 is False:
                    direction = head_left_right(frame, coordinate)
                    print(direction)
                    movement(direction)
        
        process_this_frame = not process_this_frame

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()