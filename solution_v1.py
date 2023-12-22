import gym
import os
import numpy as np
import cv2 as cv
import time as t
import pybullet as p
import config_v1
import vision_v1

os.chdir(os.path.dirname(os.getcwd()))
env = gym.make('vision-v1', 
    car_location=config_v1.CAR_LOCATION,
    balls_location=config_v1.BALLS_LOCATION,
    humanoids_location=config_v1.HUMANOIDS_LOCATION,
    visual_cam_settings=config_v1.VISUAL_CAM_SETTINGS
)

""" ENTER YOUR CODE HERE """

flag = 0

env.move(vels=[[-2,2],
               [-2,2]])

while True:
    img = env.get_image(cam_height=1 , dims=[600,600])
    hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV)
    lower_red=np.array([0, 70, 50],np.uint8)
    upper_red=np.array([10, 255, 255],np.uint8)
    kernal=np.ones((5,5),"uint8")

    mask = cv.inRange(hsv, lower_red, upper_red)

    mask=cv.dilate(mask,kernal)
    res=cv.bitwise_and(img,img,mask=mask) 
 
    contours, hierarchy = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        approx = cv.approxPolyDP(contour, 0.01*cv.arcLength(contour, True), True)
        x = approx.ravel()[0]
        y = approx.ravel()[1] - 5
        area = cv.contourArea(contour)
        print(pic, area, x)
        
        if area < 1500 and len(approx) > 12:
            cv.drawContours(img, contour, -1, (0,255,0), 2)
            cv.putText( img, "HUMANOID", (x, y), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 200, 0) )
        elif area> 4000 and (x in range(270, 330)):
            cv.putText(img, "BALL", (x,y), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 200, 0))
            env.move(vels = [[0,0],
                             [0,0]])
            env.open_grip()
            flag = 1
   
    

    if flag == 1:
        break

    cv.imshow('img',img)
   # cv.imshow('mask',mask)

    k = cv.waitKey(1)
    if k==ord('q'):
        break

if flag == 1:
    env.move(vels = [[5,5],
                     [5,5]])
    while True:
        img = env.get_image(cam_height=1 , dims=[600,600])
        hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV)
        lower_red=np.array([0, 70, 50],np.uint8)
        upper_red=np.array([10, 255, 255],np.uint8)
        kernal=np.ones((5,5),"uint8")

        mask = cv.inRange(hsv, lower_red, upper_red)

        mask=cv.dilate(mask,kernal)
        res=cv.bitwise_and(img,img,mask=mask) 
 
        contours, hierarchy = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

        cv.imshow('img',img)

        for pic, contour in enumerate(contours):
            area = cv.contourArea(contour)
            if area == 0:
                env.move(vels = [[0,0],
                                 [0,0]])
                env.close_grip()
