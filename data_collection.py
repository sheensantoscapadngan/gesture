import keras
import cv2
import numpy as np

def empty(e):
    pass

def createTrackBar():
    cv2.namedWindow("trackbar")
    cv2.createTrackbar("lh","trackbar",0,255,empty)
    cv2.createTrackbar("uh","trackbar",0,255,empty)
    cv2.createTrackbar("ls","trackbar",0,255,empty)
    cv2.createTrackbar("us","trackbar",0,255,empty)
    cv2.createTrackbar("lv","trackbar",0,255,empty)
    cv2.createTrackbar("uv","trackbar",0,255,empty)

def getTrackbarValues():
    lh = cv2.getTrackbarPos("lh","trackbar")
    uh = cv2.getTrackbarPos("uh", "trackbar")
    ls = cv2.getTrackbarPos("ls", "trackbar")
    us = cv2.getTrackbarPos("us", "trackbar")
    lv = cv2.getTrackbarPos("lv", "trackbar")
    uv = cv2.getTrackbarPos("uv", "trackbar")
    l_b = np.array([lh,ls,lv])
    u_b = np.array([uh,us,uv])
    return l_b,u_b


def preprocess_image(frame):
    # -------PRIOR CLEANUP OF BACKGROUND WITH THRESHOLD-------#
    # th_lb,th_ub = getThreshTrackBar()
    th_lb, th_ub = (141, 255)
    mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, th_lb, th_ub, cv2.THRESH_BINARY_INV)
    frame = cv2.bitwise_and(frame, frame, mask=mask)

    # -------FINAL EXTRACTION OF HAND FROM IMAGE---------#
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kernel = np.ones((21, 21), np.uint8)
    # l_b,u_b = getHSVTrackbarValues()
    l_b, u_b = (np.array([0, 58, 71]), np.array([83, 225, 255]))
    mask = cv2.inRange(frame, l_b, u_b)
    mask = cv2.dilate(mask, kernel)
    cv2.imshow("mask", mask)
    frame = cv2.bitwise_and(frame, frame, mask=mask)
    frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
    cv2.imshow("processed", frame)

    # -------prepare for processing-------#
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame, (15, 15), 0)
    frame = cv2.resize(frame, (64, 64))
    return frame

def save_frame(frame,file_number):
    file_number += 1
    file_name = directory + str(file_number) + ".jpg"
    cv2.imwrite(file_name, frame)
    print("SAVED",file_name)
    return file_number

cap = cv2.VideoCapture(0)
createTrackBar()
file_number = 5000

while cap.isOpened():
    directory = 'data/train/thumbs_down/'
    ret,frame = cap.read()
    pt1 = (300,180)
    pt2 = (550,420)
    frame = cv2.rectangle(frame,pt1,pt2,(255,0,0),2)
    target_frame = frame[pt1[1]:pt2[1],pt1[0]:pt2[0],:]
    processed = preprocess_image(target_frame)
    file_number = save_frame(processed,file_number)
    cv2.imshow("cam",frame)
    cv2.waitKey(1)