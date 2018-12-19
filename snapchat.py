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
from utils import *


filterIndex = 1

def choose_filter(event,x,y,flags,param):
	global filterIndex
	if event == cv2.EVENT_LBUTTONDBLCLK:
		filterIndex += 1
		filterIndex %= 14
	elif event == cv2.EVENT_LBUTTONDOWN:
		filterIndex -= 1
		filterIndex %=14


(leye_Start, leye_End) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(reye_Start, reye_End) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(l_eyebrow_Start, l_eyebrow_End) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
(r_eyebrow_Start, r_eyebrow_End) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
(nose_Start, nose_End) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]


dir_ = "./images/flyes/"
flies = [f for f in listdir(dir_) if isfile(join(dir_, f))] #image of flies to make the "animation"
i = 0
video_capture = cv2.VideoCapture(0) #read from webcam
(x,y,w,h) = (0,0,10,10) #whatever initial values

cv2.namedWindow('Selfie Filters')
cv2.setMouseCallback('Selfie Filters',choose_filter)


#Filters path
detector = dlib.get_frontal_face_detector()

#Facial landmarks
print("[INFO] loading facial landmark predictor...")
model = "filters/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(model) # link to model: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

while True: #while the thread is active we loop
	ret, image = video_capture.read()
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces = detector(gray, 0)

	for face in faces: #if there are faces
		(x,y,w,h) = (face.left(), face.top(), face.width(), face.height())

		
		
		# *** Facial Landmarks detection
		shape = predictor(gray, face)
		shape = face_utils.shape_to_np(shape)
		incl = calculate_inclination(shape[17], shape[26]) #inclination based on eyebrows

		leftEyebrow = shape[l_eyebrow_Start:l_eyebrow_End]
		rightEyebrow = shape[r_eyebrow_Start:r_eyebrow_End]
		nose=shape[nose_Start:nose_End]

		# condition to see if mouth is open
		is_mouth_open = (shape[66][1] -shape[62][1]) >= 10 #y coordiantes of landmark points of lips

		#hat condition
		if filterIndex==1:
			apply_sprite(image, "./images/hat.png",w,x,y, incl,factor = True,shift=True)

		#mustache condition
		if filterIndex==2:
			(x1,y1,w1,h1) = get_face_boundbox(shape, 6)
			apply_sprite(image, "./images/mustache.png",w1,x1,y1, incl,scaling=True,down=True)

		#glasses condition
		if filterIndex==3:
			(x3,y3,_,h3) = get_face_boundbox(shape, 1)
			apply_sprite(image, "./images/glasses.png",w,x,y3, incl, ontop = False)
            
            

		#doggy condition
		(x0,y0,w0,h0) = get_face_boundbox(shape, 6) #bound box of mouth
		if filterIndex==4:
			(x3,y3,w3,h3) = get_face_boundbox(shape, 5) #nose
			apply_sprite(image, "./images/doggy_nose.png",w3,x3,y3, incl, ontop = False)

			apply_sprite(image, "./images/doggy_ears.png",w,x,y, incl)

			if is_mouth_open:
				apply_sprite(image, "./images/doggy_tongue.png",w0,x0,y0, incl, ontop = False)
		'''
		else:
			if is_mouth_open:
				apply_sprite(image, "./sprites/rainbow.png",w0,x0,y0, incl, ontop = False)
		'''

        
		#flies condition
		if filterIndex==5:
			#to make the "animation" we read each time a different image of that folder
			# the images are placed in the correct order to give the animation impresion
			apply_sprite(image, dir_+flies[i],w,x,y, incl)
			i+=1
			i = 0 if i >= len(flies) else i #when done with all images of that folder, begin again
            
		# eye filter
		if filterIndex==6:
			# Load the image to be used as our overlay  
			#imgEye = cv2.imread('PROG_SAVED_2.png',-1)
			imgEye = cv2.imread('./images/big1.jpg',-1)  
			#imgEye1 = cv2.flip(imgEye,1)
			# Create the mask from the overlay image  
			orig_mask = imgEye[:,:,2]  
			#orig_mask1 = imgEye1[:,:,2]  
			# Create the inverted mask for the overlay image  
			orig_mask_inv = cv2.bitwise_not(orig_mask)  
			#orig_mask_inv1 = cv2.bitwise_not(orig_mask1)  
			# Convert the overlay image image to BGR  
			# and save the original image size  
			imgEye = imgEye[:,:,0:3]  
			origEyeHeight, origEyeWidth = imgEye.shape[:2]  
			#imgEye1 = imgEye1[:,:,0:3]  
			#origEyeHeight1, origEyeWidth1 = imgEye1.shape[:2]  

			left_eye = shape[leye_Start:leye_End]  
			right_eye = shape[reye_Start:reye_End]

			leftEyeSize, leftEyeCenter = eye_size(left_eye)  
			rightEyeSize, rightEyeCenter = eye_size(right_eye)  
   
			place_eye(image, leftEyeCenter, leftEyeSize,imgEye,orig_mask,orig_mask_inv)  
			place_eye(image, rightEyeCenter, rightEyeSize,imgEye,orig_mask,orig_mask_inv)    


		#glasses condition
		if filterIndex==7:
			(x3,y3,_,h3) = get_face_boundbox(shape, 1)
			apply_sprite(image, "./images/sunglasses.png",w,x,y3, incl, ontop = False)

		'''

		if filterIndex==7:
			sunglasses = cv2.imread('sprites/sunglasses.png', cv2.IMREAD_UNCHANGED)
			#sunglasses = rotate_bound(sunglasses, incl)
			sunglass_width = leftEyebrow[4][0]-rightEyebrow[0][0]
			sunglass_height = nose[2][1] - rightEyebrow[4][1]
			sunglass_resized = cv2.resize(sunglasses, (sunglass_width, sunglass_height), interpolation = cv2.INTER_CUBIC)
			transparent_region = sunglass_resized[:,:,:3] != 0
			image[rightEyebrow[0][1]:(rightEyebrow[0][1]+sunglass_height),rightEyebrow[0][0]:(rightEyebrow[0][0]+sunglass_width),:][transparent_region]= sunglass_resized[:,:,:3][transparent_region]
		'''
		#heart crown condition
		if filterIndex==8:
			apply_sprite(image, "./images/crown.png",w,x,y, incl,factor=True,shift_crown=True)

		#heart crown condition
		if filterIndex==9:
			apply_sprite(image, "./images/filter2.png",w,x,y, incl,factor=True,shift_crown=True)

		#glasses condition
		if filterIndex==10:
			(x3,y3,_,h3) = get_face_boundbox(shape, 1)
			apply_sprite(image, "./images/sunglasses_2.png",w,x,y3, incl, ontop = False)

		#glasses condition
		if filterIndex==11:
			(x3,y3,_,h3) = get_face_boundbox(shape, 1)
			apply_sprite(image, "./images/sg2.png",w,x,y3, incl, ontop = False)

		#glasses condition
		if filterIndex==12:
			(x3,y3,_,h3) = get_face_boundbox(shape, 1)
			apply_sprite(image, "./images/sg3.png",w,x,y3, incl, ontop = False)

		#glasses condition
		if filterIndex==13:
			(x3,y3,_,h3) = get_face_boundbox(shape, 1)
			apply_sprite(image, "./images/sg4.png",w,x,y3, incl, ontop = False)

	cv2.imshow("Selfie Filters", image)
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

video_capture.release()
cv2.destroyAllWindows()


