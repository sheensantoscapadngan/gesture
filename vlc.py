import tensorflow as tf
import keras
import cv2
import numpy as np
import pyautogui
from playsound import playsound
import time
'''
-------PROGRAM STATES-------
0 - NONE
1 - PROMPT 
2 - CHROME OPENED
------------END-------------
'''

def empty(e):
    pass

def createHSVTrackBar():
    l_b = [0,54,85]
    u_b = [255,255,255]
    cv2.namedWindow("trackbar")
    cv2.createTrackbar("lh","trackbar",l_b[0],255,empty)
    cv2.createTrackbar("uh","trackbar",u_b[0],255,empty)
    cv2.createTrackbar("ls","trackbar",l_b[1],255,empty)
    cv2.createTrackbar("us","trackbar",u_b[1],255,empty)
    cv2.createTrackbar("lv","trackbar",l_b[2],255,empty)
    cv2.createTrackbar("uv","trackbar",u_b[2],255,empty)


def getHSVTrackbarValues():
    lh = cv2.getTrackbarPos("lh","trackbar")
    uh = cv2.getTrackbarPos("uh", "trackbar")
    ls = cv2.getTrackbarPos("ls", "trackbar")
    us = cv2.getTrackbarPos("us", "trackbar")
    lv = cv2.getTrackbarPos("lv", "trackbar")
    uv = cv2.getTrackbarPos("uv", "trackbar")
    l_b = np.array([lh,ls,lv])
    u_b = np.array([uh,us,uv])
    return l_b,u_b

def createThreshTrackBar():
    l_b = 138
    cv2.namedWindow("trackbar")
    cv2.createTrackbar("lb", "trackbar", l_b, 255, empty)
    cv2.createTrackbar("ub", "trackbar", 255, 255, empty)

def getThreshTrackBar():
    lb = cv2.getTrackbarPos("lb","trackbar")
    ub = cv2.getTrackbarPos("ub","trackbar")
    return lb,ub


def executeTask(task):

    '''--------------------------
    0 - CLOSED PALM
    1 - DOWN
    2 - LEFT
    3 - RIGHT
    4 - UP
    5 - NONE
    6 - OPEN PALM
    7 - THUMBS DOWN
    --------------------------'''
    global program_state
    task = int(task)
    if task == 4: pyautogui.press("up")
    elif task == 1: pyautogui.press("down")
    elif task == 3: pyautogui.press("right")
    elif task == 2: pyautogui.press("left")
    elif task == 6: pyautogui.press("space")


def preprocess_image(frame):

    #-------PRIOR CLEANUP OF BACKGROUND WITH THRESHOLD-------#
    th_lb,th_ub = getThreshTrackBar()
    mask = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    _,mask = cv2.threshold(mask,th_lb,th_ub,cv2.THRESH_BINARY_INV)
    frame = cv2.bitwise_and(frame,frame,mask=mask)

    #-------FINAL EXTRACTION OF HAND FROM IMAGE---------#
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    kernel = np.ones((21,21),np.uint8)
    l_b, u_b = getHSVTrackbarValues()
    mask = cv2.inRange(frame,l_b,u_b)
    mask = cv2.dilate(mask,kernel)
    frame = cv2.bitwise_and(frame,frame,mask=mask)
    cv2.imshow("processed",frame)
    frame = cv2.cvtColor(frame,cv2.COLOR_HSV2BGR)

    #-------prepare for processing-------#
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow("frame",frame)
    frame = cv2.GaussianBlur(frame,(15,15),0)
    frame = cv2.resize(frame,(64,64))
    frame = np.reshape(frame,(64,64,1))
    frame = np.expand_dims(frame,axis=0)
    return frame

def process_frame(target_frame):
    input = preprocess_image(target_frame)
    output = model.predict(input)[0]
    class_result = list(output).index(max(output))
    return str(class_result)

def background_process():
    global  program_state
    prev_result = -1
    result_cnt = 0
    decision_threshold = 2

    while cap.isOpened():
        #---------EXTRACT FRAME OF INTEREST---------#
        ret,frame = cap.read()
        pt1 = (300,180)
        pt2 = (550,420)
        frame = cv2.rectangle(frame,pt1,pt2,(255,0,0),2)
        target_frame = frame[pt1[1]:pt2[1],pt1[0]:pt2[0],:]

        #---------------CALCULATE RESULT AND REACT----------------#
        result = process_frame(target_frame)
        if result == prev_result: result_cnt += 1
        else: result_cnt = 1
        prev_result = result
        if result_cnt == decision_threshold:
            print("EXECUTING TASK")
            executeTask(result)
            result_cnt = 0
        #-----------------DISPLAY CAMERA------------------#
        frame = cv2.putText(frame,result,(100,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),1)
        cv2.imshow("cam",frame)
        cv2.waitKey(1)

#-----------------------------------------PROGRAM START-----------------------------------------#
model = keras.models.load_model('models/all_model1.h5')
print("MODEL INITIALIZED")
cap = cv2.VideoCapture(0)

createHSVTrackBar()
createThreshTrackBar()

background_process()
