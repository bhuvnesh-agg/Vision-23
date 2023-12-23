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

"""YELLOW"""
flag = 0

t.sleep(0.5)

env.move(vels=[[-5,5],
               [-5,5]])

while True:
    img = env.get_image(cam_height=1 , dims=[600,600])
    hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV)
    lower_yellow=np.array([22, 93, 0],np.uint8)
    upper_yellow=np.array([45, 255, 255],np.uint8)
    kernal=np.ones((5,5),"uint8")

    mask = cv.inRange(hsv, lower_yellow, upper_yellow)

    mask=cv.dilate(mask,kernal)
    res=cv.bitwise_and(img,img,mask=mask) 
 
    contours, hierarchy = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        approx = cv.approxPolyDP(contour, 0.01*cv.arcLength(contour, True), True)
        x = approx.ravel()[0]
        area = cv.contourArea(contour)
        
        if area < 1500 and len(approx) > 12:
            cv.drawContours(img, contour, -1, (0,255,0), 2)
        elif area> 3000 and (x in range(290, 310)):
            env.move(vels = [[0,0],
                             [0,0]])
            t.sleep(0.3)
            env.open_grip()
            flag = 1
    

    if flag == 1:
        cv.destroyAllWindows()
        break

   # cv.imshow('img',img)
   # cv.imshow('mask',mask)

    k = cv.waitKey(1)
    if k==ord('q'):
        break

if flag == 1:
    env.move(vels = [[8,8],
                     [8,8]])
    while True:
        img = env.get_image(cam_height=1 , dims=[600,600])
        hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV)
        lower_yellow=np.array([22, 93, 0],np.uint8)
        upper_yellow=np.array([45, 255, 255],np.uint8)
        kernal=np.ones((5,5),"uint8")

        mask = cv.inRange(hsv, lower_yellow, upper_yellow)

        mask=cv.dilate(mask,kernal)
        res=cv.bitwise_and(img,img,mask=mask) 
 
        contours, hierarchy = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

    #    cv.imshow('mask',img)

        cv.waitKey(1)
        prevArea = 50

        if len(contours) == 0:
            env.move(vels = [[0,0],
                            [0,0]])
            env.close_grip()
            t.sleep(0.5)
            cv.destroyAllWindows()
            break

env.move(vels=[[-5,5],
               [-5,5]])

while True:
    img = env.get_image(cam_height=1 , dims=[600,600])
    hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV)
    lower_pole=np.array([70, 25, 25],np.uint8)
    upper_pole=np.array([90, 255, 255],np.uint8)
    kernal=np.ones((5,5),"uint8")

    mask = cv.inRange(hsv, lower_pole, upper_pole)

    mask=cv.dilate(mask,kernal)
    res=cv.bitwise_and(img,img,mask=mask) 
 
    contours, hierarchy = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
#    cv.imshow('pole', img)
 #   cv.imshow('pole-mask', mask)
    cv.waitKey(1)
    for pic, contour in enumerate(contours):
        approx = cv.approxPolyDP(contour, 0.01*cv.arcLength(contour, True), True)
        x = approx.ravel()[0]
        if x in range(295, 310):
            env.move(vels = [[0,0],
                             [0,0]])
            t.sleep(0.3)
            flag = 0
    if flag == 0:
        break

if flag == 0:

    while True:
        img = env.get_image(cam_height=1 , dims=[600,600])
        hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV)
        lower_pole=np.array([70, 25, 25],np.uint8)
        upper_pole=np.array([90, 255, 255],np.uint8)
        kernal=np.ones((5,5),"uint8")

        mask = cv.inRange(hsv, lower_pole, upper_pole)

        mask=cv.dilate(mask,kernal)
        res=cv.bitwise_and(img,img,mask=mask) 
 
        contours, hierarchy = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
       # cv.imshow('pole', img)
    #    cv.imshow('pole-mask', mask)
        cv.waitKey(1)
        area = 0
        for pic, contour in enumerate(contours):
            area += cv.contourArea(contour)
        print(area)
        env.close_grip()
        if area >= 22000:
            env.move(vels = [[0,0],
                             [0,0]])
            t.sleep(0.2)
            break
        env.move(vels = [[8,8],
                        [8,8]])


env.move(vels = [[-4,4],
                 [-4,4]])
while True:
    img = env.get_image(cam_height=1 , dims=[600,600])
    hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV)
    lower_yellow=np.array([22, 93, 0],np.uint8)
    upper_yellow=np.array([45, 255, 255],np.uint8)
    kernal=np.ones((5,5),"uint8")

    mask = cv.inRange(hsv, lower_yellow, upper_yellow)

    mask=cv.dilate(mask,kernal)
    res=cv.bitwise_and(img,img,mask=mask) 

   # cv.imshow('shoot', mask)
    cv.waitKey(1)

    contours, hierarchy = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        cv.drawContours(img, contour, -1, (0,255,0), 2)
        approx = cv.approxPolyDP(contour, 0.01*cv.arcLength(contour, True), True)
        x = approx.ravel()[0]
        area = cv.contourArea(contour)
        env.close_grip()
        if area > 1000 and x in range(240, 305):
            env.move(vels = [[0,0],
                             [0,0]])
            t.sleep(0.4)
            env.open_grip()
            env.shoot(500)
            flag = 1
            break
    if flag == 1:
        break


"""RED"""

flag = 0

t.sleep(0.5)

env.move(vels=[[-5,5],
               [-5,5]])

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
        area = cv.contourArea(contour)
        
        if area < 1500 and len(approx) > 12:
            cv.drawContours(img, contour, -1, (0,255,0), 2)
        elif area> 2000 and (x in range(265, 275)):
            env.move(vels = [[0,0],
                             [0,0]])
            t.sleep(0.5)
            env.open_grip()
            flag = 1
    

    if flag == 1:
        cv.destroyAllWindows()
        break

   # cv.imshow('img',img)
   # cv.imshow('mask',mask)

    k = cv.waitKey(1)
    if k==ord('q'):
        break

if flag == 1:
    env.move(vels = [[10,10],
                     [10,10]])
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

    #    cv.imshow('mask',img)

        cv.waitKey(1)
        prevArea = 50

        if len(contours) == 0:
            env.move(vels = [[0,0],
                            [0,0]])
            env.close_grip()
            t.sleep(0.5)
            cv.destroyAllWindows()
            break

env.move(vels=[[-5,5],
               [-5,5]])

while True:
    img = env.get_image(cam_height=1 , dims=[600,600])
    hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV)
    lower_pole=np.array([70, 25, 25],np.uint8)
    upper_pole=np.array([90, 255, 255],np.uint8)
    kernal=np.ones((5,5),"uint8")

    mask = cv.inRange(hsv, lower_pole, upper_pole)

    mask=cv.dilate(mask,kernal)
    res=cv.bitwise_and(img,img,mask=mask) 
 
    contours, hierarchy = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
#    cv.imshow('pole', img)
 #   cv.imshow('pole-mask', mask)
    cv.waitKey(1)
    for pic, contour in enumerate(contours):
        approx = cv.approxPolyDP(contour, 0.01*cv.arcLength(contour, True), True)
        x = approx.ravel()[0]
        if x in range(295, 310):
            env.move(vels = [[0,0],
                             [0,0]])
            t.sleep(0.3)
            flag = 0
    if flag == 0:
        break

if flag == 0:

    while True:
        img = env.get_image(cam_height=1 , dims=[600,600])
        hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV)
        lower_pole=np.array([70, 25, 25],np.uint8)
        upper_pole=np.array([90, 255, 255],np.uint8)
        kernal=np.ones((5,5),"uint8")

        mask = cv.inRange(hsv, lower_pole, upper_pole)

        mask=cv.dilate(mask,kernal)
        res=cv.bitwise_and(img,img,mask=mask) 
 
        contours, hierarchy = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
       # cv.imshow('pole', img)
        #cv.imshow('pole-mask', mask)
        cv.waitKey(1)
        area = 0
        for pic, contour in enumerate(contours):
            area += cv.contourArea(contour)
        print(area)
        if area >= 22000:
            env.move(vels = [[0,0],
                             [0,0]])
            t.sleep(0.2)
            break
        env.move(vels = [[9,9],
                        [9,9]])


env.move(vels = [[-4,4],
                 [-4,4]])
while True:
    img = env.get_image(cam_height=1 , dims=[600,600])
    hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV)
    lower_red=np.array([0, 70, 50],np.uint8)
    upper_red=np.array([10, 255, 255],np.uint8)
    kernal=np.ones((5,5),"uint8")

    mask = cv.inRange(hsv, lower_red, upper_red)

    mask=cv.dilate(mask,kernal)
    res=cv.bitwise_and(img,img,mask=mask)

    env.close_grip()

  #  cv.imshow('shoot', mask)
    cv.waitKey(1)

    contours, hierarchy = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        cv.drawContours(img, contour, -1, (0,255,0), 2)
        approx = cv.approxPolyDP(contour, 0.01*cv.arcLength(contour, True), True)
        x = approx.ravel()[0]
        area = cv.contourArea(contour)
        
        if area > 800 and x in range(260, 305):
            env.move(vels = [[0,0],
                             [0,0]])
            t.sleep(0.4)
            env.open_grip()
            env.shoot(200)
            flag = 1
            break
    if flag == 1:
        break


"""GREEN"""

flag = 0

t.sleep(0.5)

env.move(vels=[[4,-4],
               [4,-4]])

while True:
    img = env.get_image(cam_height=1 , dims=[600,600])
    hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV)
    lower_yellow=np.array([36, 25, 25],np.uint8)
    upper_yellow=np.array([70, 255, 255],np.uint8)
    kernal=np.ones((5,5),"uint8")

    mask = cv.inRange(hsv, lower_yellow, upper_yellow)

    mask=cv.dilate(mask,kernal)
    res=cv.bitwise_and(img,img,mask=mask) 
 
    contours, hierarchy = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        approx = cv.approxPolyDP(contour, 0.01*cv.arcLength(contour, True), True)
        x = approx.ravel()[0]
        area = cv.contourArea(contour)
        
        if area < 1500 and len(approx) > 12:
            cv.drawContours(img, contour, -1, (0,255,0), 2)
        elif area> 2000 and (x in range(280,300)):
            env.move(vels = [[0,0],
                             [0,0]])
            t.sleep(0.5)
            env.open_grip()
            flag = 1
    

    if flag == 1:
        cv.destroyAllWindows()
        break

   # cv.imshow('img',img)
   # cv.imshow('mask',mask)

    k = cv.waitKey(1)
    if k==ord('q'):
        break

if flag == 1:
    env.move(vels = [[8,8],
                     [8,8]])
    while True:
        img = env.get_image(cam_height=1 , dims=[600,600])
        hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV)
        lower_yellow=np.array([36, 25, 25],np.uint8)
        upper_yellow=np.array([70, 255, 255],np.uint8)
        kernal=np.ones((5,5),"uint8")

        mask = cv.inRange(hsv, lower_yellow, upper_yellow)

        mask=cv.dilate(mask,kernal)
        res=cv.bitwise_and(img,img,mask=mask) 
 
        contours, hierarchy = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

    #    cv.imshow('mask',img)

        cv.waitKey(1)
        prevArea = 50

        if len(contours) == 0:
            env.move(vels = [[0,0],
                            [0,0]])
            env.close_grip()
            t.sleep(0.5)
            cv.destroyAllWindows()
            break

env.move(vels=[[-5,5],
               [-5,5]])

while True:
    img = env.get_image(cam_height=1 , dims=[600,600])
    hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV)
    lower_pole=np.array([70, 25, 25],np.uint8)
    upper_pole=np.array([90, 255, 255],np.uint8)
    kernal=np.ones((5,5),"uint8")

    mask = cv.inRange(hsv, lower_pole, upper_pole)

    mask=cv.dilate(mask,kernal)
    res=cv.bitwise_and(img,img,mask=mask) 
 
    contours, hierarchy = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
#    cv.imshow('pole', img)
 #   cv.imshow('pole-mask', mask)
    cv.waitKey(1)
    for pic, contour in enumerate(contours):
        approx = cv.approxPolyDP(contour, 0.01*cv.arcLength(contour, True), True)
        x = approx.ravel()[0]
        if x in range(295, 310):
            env.move(vels = [[0,0],
                             [0,0]])
            t.sleep(0.3)
            flag = 0
    if flag == 0:
        break

if flag == 0:

    while True:
        img = env.get_image(cam_height=1 , dims=[600,600])
        hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV)
        lower_pole=np.array([70, 25, 25],np.uint8)
        upper_pole=np.array([90, 255, 255],np.uint8)
        kernal=np.ones((5,5),"uint8")

        mask = cv.inRange(hsv, lower_pole, upper_pole)

        mask=cv.dilate(mask,kernal)
        res=cv.bitwise_and(img,img,mask=mask) 
 
        contours, hierarchy = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
       # cv.imshow('pole', img)
    #    cv.imshow('pole-mask', mask)
        cv.waitKey(1)
        area = 0
        for pic, contour in enumerate(contours):
            area += cv.contourArea(contour)
        print(area)
        env.close_grip()
        if area >= 22000:
            env.move(vels = [[0,0],
                             [0,0]])
            t.sleep(0.2)
            break
        env.move(vels = [[8,8],
                        [8,8]])


env.move(vels = [[4,-4],
                 [4,-4]])
while True:
    img = env.get_image(cam_height=1 , dims=[600,600])
    hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV)
    lower_yellow=np.array([36, 25, 25],np.uint8)
    upper_yellow=np.array([70, 255, 255],np.uint8)
    kernal=np.ones((5,5),"uint8")

    mask = cv.inRange(hsv, lower_yellow, upper_yellow)

    mask=cv.dilate(mask,kernal)
    res=cv.bitwise_and(img,img,mask=mask) 

   # cv.imshow('shoot', mask)
    cv.waitKey(1)

    contours, hierarchy = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        cv.drawContours(img, contour, -1, (0,255,0), 2)
        approx = cv.approxPolyDP(contour, 0.01*cv.arcLength(contour, True), True)
        x = approx.ravel()[0]
        area = cv.contourArea(contour)
        env.close_grip()
        if area > 500 and x in range(230, 350):
            env.move(vels = [[0,0],
                             [0,0]])
            t.sleep(0.4)
            env.open_grip()
            env.shoot(500)
            flag = 1
            break
   # if flag == 1:
    #    break


