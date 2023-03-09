import cv2
import numpy as np
import os

def HoughDetectBulk(input_directory, output_directory):
    for image in os.listdir(input_directory):
        HoughDetect(image, input_directory, output_directory)

def HoughDetect(input_image_name, input_directory, output_directory):
    maxCircles = 3.4 # min number of circles that can fit one next to each other
    minCircles = 5 # max number of circles that can fit one next to each other

    image = cv2.imread(input_directory + '/' + input_image_name)

    output = image.copy()
    height, width = image.shape[:2]

    maxRadius = int(1.1*(width/3.4)/2)
    minRadius = int(0.9*(width/5)/2)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dp = 1.2
    minDist = 2*minRadius
    param1 = 20
    param2 = 12

    circles = cv2.HoughCircles(image=gray,
                            method=cv2.HOUGH_GRADIENT,
                            dp=dp,
                            minDist=minDist,
                            param1=param1,
                            param2=param2,
                            minRadius=minRadius,
                            maxRadius=maxRadius
                            )

    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circlesRound = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circlesRound:
            cv2.circle(output, (x, y), r, (0, 255, 0), 1)

        cv2.imwrite(output_directory + '/' + input_image_name, output)
        print ('Image saved at: ' + output_directory + '/' + input_image_name)
    else:
        print ('No circles found')

def Script():
    imageNamePrefix = '0_97300'
    imageExt = '.png'

    for i in range(1,100):
        imageName = imageNamePrefix + str(i).zfill(2) + imageExt
        HoughDetect(imageName, 'RandomDataset/crop', 'circles')


##


def HoughDetectBatch(input_image_name, input_directory, output_directory):
    maxCircles = 3.4 # min number of circles that can fit one next to each other
    minCircles = 5 # max number of circles that can fit one next to each other

    reference = cv2.imread('RandomDataset/crop/0_9730001.png')
    height, width = reference.shape[:2]

    image = cv2.imread(input_directory + '/' + input_image_name)

    output = image.copy()

    maxRadius = int(1.1*(width/3.4)/2)
    minRadius = int(0.9*(width/5)/2)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dp = 1.2
    minDist = 2*minRadius
    param1 = 20
    param2 = 10

    circles = cv2.HoughCircles(image=gray,
                            method=cv2.HOUGH_GRADIENT,
                            dp=dp,
                            minDist=minDist,
                            param1=param1,
                            param2=param2,
                            minRadius=minRadius,
                            maxRadius=maxRadius
                            )

    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circlesRound = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circlesRound:
            cv2.circle(output, (x, y), r, (0, 255, 0), 1)

        cv2.imwrite(output_directory + '/' + input_image_name, output)
        print ('Image saved at: ' + output_directory + '/' + input_image_name)
    else:
        print ('No circles found')


def ScriptBatch():
    HoughDetectBatch('sample_batch.png', 'wgansamples', 'circles')











