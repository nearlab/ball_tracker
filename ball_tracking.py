# USAGE
# python ball_tracking.py --video ball_tracking_example.mp4
# python ball_tracking.py

# Don't forget about republish [in_transport] in:=<in_base_topic> [out_transport] out:=<out_base_topic>

# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import roslib
import rospy
from sensor_msgs.msg import Image
from pv_estimator.msg import Meas

class frame_grabber:
	def __init__(self):
		self.frame = np.zeros((256,256,3), np.uint8)	
		self.time = rospy.Time()

  # Callback for when an image is available
	def imageCallback(self, msg):
		np_arr = np.fromstring(msg.data, np.uint8)
		self.frame = cv2.imdecode(np_arr,cv2.IMREAD_COLOR)
		#destRGB = cv2.cvtColor(srcBGR, cv2.COLOR_BGR2RGB)
		self.time = msg.header.stamp

def main(args):
	rospy.init_node('ball_tracker', anonymous=True)	

	### Image Processing variable here ###
	redLower = (150, 100, 100)
	redUpper = (200, 255, 255)
	prevFrame =  np.zeros((256,256,3), np.uint8)	
	# lk_params = dict( winSize = (15,15),
	# 									maxLevel = 2,
	# 									criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, .03))


	### ROS variables ###
	fg = frame_grabber()
	sub = rospy.Subscriber("/pixelink/image",Image,fg.imageCallback,queue_size = 1)
	pub = rospy.Publisher("/tracker/meas", Meas)
	
	tLastImg = rospy.Time()
	tCurrImg = rospy.Time()
	rate = rospy.Rate(100)
			
	# keep looping
	while not rospy.is_shutdown():
		v = (0,0)
		# check if new image is available
		if(tCurrImg.toSec() <= tLastImg.toSec()):
			rate.sleep()
			rospy.spinOnce()
			continue
		
		# resize the frame, blur it, and convert it to the HSV
		# color space
		frame = imutils.resize(fg.frame, width=600)
		blurred = cv2.GaussianBlur(frame, (11, 11), 0)
		grey = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
		hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

		# construct a mask for the color "red", then perform
		# a series of dilations and erosions to remove any small
		# blobs left in the mask
		mask = cv2.inRange(hsv, redLower, redUpper)
		mask = cv2.erode(mask, None, iterations=2)
		mask = cv2.dilate(mask, None, iterations=2)

		# find contours in the mask and initialize the current
		# (x, y) center of the ball
		cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		cnts = cnts[0] if imutils.is_cv2() else cnts[1]
		center = None

		# only proceed if at least one contour was found
		if len(cnts) > 0:
			# find the largest contour in the mask, then use
			# it to compute the minimum enclosing circle and
			# centroid
			c = max(cnts, key=cv2.contourArea)
			((x, y), radius) = cv2.minEnclosingCircle(c)
			M = cv2.moments(c)
			center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

			if(tLastImg.toSec()>0):
				prevGrey = cv2.cvtColor(cv2.GaussianBlur(prevFrame,(11,11),0),cv2.COLOR_BGR2GRAY)
				pts, st, err = cv2.calcOptFlowPyrLK(prevGrey,grey,c,None)#,**lk_params)
				good_old = c[st==1]
				good_new = pts[st==1]
				s = np.sum(good_new-good_old,axis=0)
				v = s/len(good_new[:,1])
			# # only proceed if the radius meets a minimum size
			# if radius > 10:
			# 	# draw the circle and centroid on the frame,
			# 	# then update the list of tracked points
			# 	cv2.circle(frame, (int(x), int(y)), int(radius),
			# 		(0, 255, 255), 2)
			# 	cv2.circle(frame, center, 5, (0, 0, 255), -1)
		prevFrame = frame
		# show the frame to our screen
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		measMsg =Meas()
		measMsg.r[0] = center[0]
		measMsg.r[1] = center[1]
		measMsg.v[0] = v[0]
		measMsg.v[1] = v[1]
		tCurr = rospy.Time.now() 
		measMsg.tStamp = tCurr.toSec()
		pub.publsih(measMsg)

		tLastImg = tCurrImg
		rate.sleep()
		rospy.spinOnce()

if __name__ == '__main__':
	main(sys.argv)
