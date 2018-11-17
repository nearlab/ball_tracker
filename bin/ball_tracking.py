#!/usr/bin/env python

# USAGE
# python ball_tracking.py --video ball_tracking_example.mp4
# python ball_tracking.py

# Don't forget about republish [in_transport] in:=/pixelink/image [out_transport] out:=/pixelink/image_py

# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import roslib
import rospy
import traceback
import logging
import sys
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from pv_estimator.msg import Meas

class frame_grabber:
  def __init__(self):
    self.prevFrame = np.zeros((256,256,3),np.uint8)
    self.tLastImg = rospy.Time()
    self.bridge = CvBridge()
    
    self.pub = rospy.Publisher("/tracker/meas", Meas, queue_size = 2)
    self.count = 0
    print('Frame Grabber Initialized')

  # Callback for when an image is available
  def imageCallback(self, msg):
    found = False
    try:
      frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
    except Exception as e:
      print('failed to convert ')
      print(e)
      logging.error(traceback.format_exc())
    #destRGB = cv2.cvtColor(srcBGR, cv2.COLOR_BGR2RGB)
    tCurrImg = msg.header.stamp

    # Setup Image Processing Variables
    redLower1 = (150, 100, 100)
    redUpper1 = (180, 255, 255)
    redLower2 = (0, 100, 100)
    redUpper2 = (20, 255, 255)
    v = (0,0)
    
    # resize the frame, blur it, and convert it to the HSV
    # color space
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    grey = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # construct a mask for the color "red", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, redLower1, redUpper1)
    mask = mask | cv2.inRange(hsv, redLower2, redUpper2)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
      cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    center = None
    print('Processed frame, found '+str(len(cnts))+' contours')
    # only proceed if at least one contour was found
    
    if len(cnts) > 0:
      # find the largest contour in the mask, then use
      # it to compute the minimum enclosing circle and
      # centroid
      c = max(cnts, key=cv2.contourArea)
      ((x, y), radius) = cv2.minEnclosingCircle(c)
      M = cv2.moments(c)
      center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
      print('Found balloon '+ str(c.shape))
      inPts = np.transpose(np.array([c[:,0,0],c[:,0,1]],dtype=np.float32))
      found = True
      if(self.tLastImg.to_sec()>0):
        prevGrey = cv2.cvtColor(cv2.GaussianBlur(self.prevFrame,(11,11),0),cv2.COLOR_BGR2GRAY)
        pts, st, err = cv2.calcOpticalFlowPyrLK(prevGrey,grey,np.float32([inPts]),None)#,**lk_params)
        good_old = c[st==1]
        good_new = pts[np.transpose(st)==1]
        s = np.sum(good_new-good_old,axis=0)/len(good_new[:,1]) 
        v = s/(tCurrImg.to_sec() - self.tLastImg.to_sec())#This had better not be dividing by zero...
        print('avg motion is ' + str(s)+'\t and the avg vel is '+str(v))
      # # only proceed if the radius meets a minimum size
      # if radius > 10:
      # 	# draw the circle and centroid on the frame,
      # 	# then update the list of tracked points
      	
      cv2.circle(frame, (int(x), int(y)), int(radius),
        (0, 255, 255), 2)
      cv2.circle(frame, center, 5, (0, 0, 255), -1)

    # save the frame to file
    self.count = self.count + 1
    imName = "image-%04d.jpg" % self.count
    cv2.imwrite(imName,frame)
    key = cv2.waitKey(1) & 0xFF
    self.prevFrame = frame
      
    

    measMsg = Meas()
    if(found):
      measMsg.r[0] = center[0]
      measMsg.r[1] = center[1]
      measMsg.v[0] = v[0]
      measMsg.v[1] = v[1]
      tCurr = rospy.Time.now() 
      measMsg.tStamp = tCurr.to_sec()
      self.pub.publish(measMsg)

    self.tLastImg = tCurrImg

def main(args):
  rospy.init_node('ball_tracker', anonymous=True)	

  ### Image Processing variable here ###

  prevFrame =  np.zeros((256,256,3), np.uint8)	
  # lk_params = dict( winSize = (15,15),
  # 									maxLevel = 2,
  # 									criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, .03))


  ### ROS variables ###
  fg = frame_grabber()

  sub = rospy.Subscriber("/pixelink/image",Image,fg.imageCallback,queue_size = 100)
  print('ROS Initialized')
  rospy.spin()
  

if __name__ == '__main__':
  main(sys.argv)
