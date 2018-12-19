from Tkinter import *
from PIL import Image
from PIL import ImageTk
import cv2, threading, os, time
from threading import Thread
from os import listdir
from os.path import isfile, join

import dlib
from imutils import face_utils, rotate_bound
import math
from scipy.spatial import distance as dist  
from scipy.spatial import ConvexHull 
import numpy as np 

# Draws sprite over a image
# It uses the alpha chanel to see which pixels need to be reeplaced
# Input: image, sprite: numpy arrays
# output: resulting merged image

def draw_sprite(frame, sprite, x_offset, y_offset,shift=False,down=False):
    (h,w) = (sprite.shape[0], sprite.shape[1])
    (imgH,imgW) = (frame.shape[0], frame.shape[1])

    if y_offset+h >= imgH: #if sprite gets out of image in the bottom
        sprite = sprite[0:imgH-y_offset,:,:]

    if x_offset+w >= imgW: #if sprite gets out of image to the right
        sprite = sprite[:,0:imgW-x_offset,:]

    if x_offset < 0: #if sprite gets out of image to the left
        sprite = sprite[:,abs(x_offset)::,:]
        w = sprite.shape[1]
        x_offset = 0

    #for each RGB chanel
    for c in range(3):
            #chanel 4 is alpha: 255 is not transpartne, 0 is transparent background
            frame[y_offset:y_offset+h, x_offset:x_offset+w, c] =  \
            sprite[:,:,c] * (sprite[:,:,3]/255.0) +  frame[y_offset:y_offset+h, x_offset:x_offset+w, c] * (1.0 - sprite[:,:,3]/255.0)
    return frame

#Adjust the given sprite to the head's width and position
#in case of the sprite not fitting the screen in the top, the sprite should be trimed
def adjust_sprite2head(sprite, head_width, head_ypos, ontop = True, factor = False,scaling=False):
    (h_sprite,w_sprite) = (sprite.shape[0], sprite.shape[1])
    if factor==False:
        factor = 1.0*head_width/w_sprite
    else:
        factor = 1.6*head_width/w_sprite
    sprite = cv2.resize(sprite, (0,0), fx=factor, fy=factor) # adjust to have the same width as head
    if scaling == True:
        (h_sprite,w_sprite) = (sprite.shape[0], sprite.shape[1])
        sprite=cv2.resize(sprite,(w_sprite+15,h_sprite))
    (h_sprite,w_sprite) = (sprite.shape[0], sprite.shape[1])

    y_orig =  head_ypos-h_sprite if ontop else head_ypos # adjust the position of sprite to end where the head begins
    if (y_orig < 0): #check if the head is not to close to the top of the image and the sprite would not fit in the screen
            sprite = sprite[abs(y_orig)::,:,:] #in that case, we cut the sprite
            y_orig = 0 #the sprite then begins at the top of the image
    return (sprite, y_orig)

# Applies sprite to image detected face's coordinates and adjust it to head
def apply_sprite(image, path2sprite,w,x,y, angle, ontop = True,factor= False,scaling=False,shift=False,shift_crown=False,down=False):
    sprite = cv2.imread(path2sprite,-1)
    #print sprite.shape
    sprite = rotate_bound(sprite, angle)
    (sprite, y_final) = adjust_sprite2head(sprite, w, y, ontop,factor,scaling)
    if shift==True and down==False and shift_crown==False:
        image = draw_sprite(image,sprite,x-50, y_final+5,shift)
    elif shift==False and down==True and shift_crown==False:
        image = draw_sprite(image,sprite,x-9, y_final+5,down)
    elif shift==False and down==False and shift_crown==True:
        image = draw_sprite(image,sprite,x-47, y_final+10,shift_crown)
    elif shift==False and down==False and shift_crown==False:
        image = draw_sprite(image,sprite,x, y_final)


#points are tuples in the form (x,y)
# returns angle between points in degrees
def calculate_inclination(point1, point2):
    x1,x2,y1,y2 = point1[0], point2[0], point1[1], point2[1]
    incl = 180/math.pi*math.atan((float(y2-y1))/(x2-x1))
    return incl


def calculate_boundbox(list_coordinates):
    x = min(list_coordinates[:,0])
    y = min(list_coordinates[:,1])
    w = max(list_coordinates[:,0]) - x
    h = max(list_coordinates[:,1]) - y
    return (x,y,w,h)

def get_face_boundbox(points, face_part):
    if face_part == 1:
        (x,y,w,h) = calculate_boundbox(points[17:22]) #left eyebrow
    elif face_part == 2:
        (x,y,w,h) = calculate_boundbox(points[22:27]) #right eyebrow
    elif face_part == 3:
        (x,y,w,h) = calculate_boundbox(points[36:42]) #left eye
    elif face_part == 4:
        (x,y,w,h) = calculate_boundbox(points[42:48]) #right eye
    elif face_part == 5:
        (x,y,w,h) = calculate_boundbox(points[29:36]) #nose
    elif face_part == 6:
        (x,y,w,h) = calculate_boundbox(points[48:68]) #mouth
    return (x,y,w,h)

def eye_size(eye):  
    eyeWidth = dist.euclidean(eye[0], eye[3])  
    hull = ConvexHull(eye)  
    eyeCenter = np.mean(eye[hull.vertices, :], axis=0)  
    eyeCenter = eyeCenter.astype(int)  
    return int(eyeWidth), eyeCenter  
   
def place_eye(frame, eyeCenter, eyeSize,imgEye,orig_mask,orig_mask_inv):  
    eyeSize = int(eyeSize * 1.5)  
   
    x1 = int(eyeCenter[0] - (eyeSize/2))  
    x2 = int(eyeCenter[0] + (eyeSize/2))  
    y1 = int(eyeCenter[1] - (eyeSize/2))  
    y2 = int(eyeCenter[1] + (eyeSize/2))  
   
    h, w = frame.shape[:2]  
   
    # check for clipping  
    if x1 < 0:  
        x1 = 0  
    if y1 < 0:  
        y1 = 0  
    if x2 > w:  
        x2 = w  
    if y2 > h:  
        y2 = h  
   
    # re-calculate the size to avoid clipping  
    eyeOverlayWidth = x2 - x1  
    eyeOverlayHeight = y2 - y1  
   
    # calculate the masks for the overlay  
    eyeOverlay = cv2.resize(imgEye, (eyeOverlayWidth,eyeOverlayHeight), interpolation = cv2.INTER_AREA)  
    mask = cv2.resize(orig_mask, (eyeOverlayWidth,eyeOverlayHeight), interpolation = cv2.INTER_AREA)  
    mask_inv = cv2.resize(orig_mask_inv, (eyeOverlayWidth,eyeOverlayHeight), interpolation = cv2.INTER_AREA)  
   
    # take ROI for the verlay from background, equal to size of the overlay image  
    roi = frame[y1:y2, x1:x2]  
   
    # roi_bg contains the original image only where the overlay is not, in the region that is the size of the overlay.  
    roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)  
   
    # roi_fg contains the image pixels of the overlay only where the overlay should be  
    roi_fg = cv2.bitwise_and(eyeOverlay,eyeOverlay,mask = mask)  
   
    # join the roi_bg and roi_fg  
    dst = cv2.add(roi_bg,roi_fg)  
   
    # place the joined image, saved to dst back over the original image  
    frame[y1:y2, x1:x2] = dst  
