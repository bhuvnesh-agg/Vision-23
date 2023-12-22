import gym
import os
import cv2 as cv
import time as t
import pybullet as p
import config_v0
import vision_v0
import numpy as np

os.chdir(os.path.dirname(os.getcwd()))
env = gym.make('vision-v0', 
    car_location=config_v0.CAR_LOCATION,
    ball_location=config_v0.BALL_LOCATION,
    humanoid_location=config_v0.HUMANOID_LOCATION,
    visual_cam_settings=config_v0.VISUAL_CAM_SETTINGS
)

""" ENTER YOUR CODE HERE """

env.open_grip()

env.move(vels = [[10,10],
                 [10,10]])
t.sleep(1.4)

env.move(vels = [[0,0],
                 [0,0]])

t.sleep(0.4)
env.close_grip()
 
env.move(vels=[[5,-5],
               [5,-5]])



while True:
    img = env.get_image(cam_height=1.5, dims=[100,400])
   
   
   

    hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV)
    lower_red=np.array([0, 70, 50],np.uint8)
    upper_red=np.array([10, 255, 255],np.uint8)
    kernal=np.ones((5,5),"uint8")

    mask = cv.inRange(hsv, lower_red, upper_red)

    mask=cv.dilate(mask,kernal)
    res=cv.bitwise_and(img,img,mask=mask) 
     
    cv.imshow('img',img)
    cv.imshow('mask',mask)

    contours, hierarchy = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv.contourArea(contour) 
        
    
        if(area > 250):
                env.move(vels = [[0,0],
                 [0,0]])
                t.sleep(2)

                env.open_grip()
                env.shoot(100) 
                t.sleep(30)
    

          
    k = cv.waitKey(1)
    if k==ord('q'):
        break