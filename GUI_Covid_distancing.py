import qdarkstyle
from threading import Thread
from collections import deque
from datetime import datetime
import time
import sys
import cv2
import imutils
from PyQt4 import QtCore, QtGui
import cv2,os,urllib.request
import numpy as np
from Detection_files import social_distancing_config as config
from Detection_files.detection import detect_people
from scipy.spatial import distance as dist
import argparse
import imutils
from django.conf import settings
from ctypes import *
import math
import random
import time
import darknet
from itertools import combinations
import requests
import json
import imutils,time,sys
from threading import Thread

weightPath =   "./yolov4.weights"
configPath =   "./yolov4.cfg"
metaPath =    "./coco.data"
print("[INFO] loading YOLO from disk...")
global metaMain, netMain, altNames
netMain = None
metaMain = None
altNames = None

if not os.path.exists(configPath):
	raise ValueError("Invalid config path `" +
						os.path.abspath(configPath)+"`")
if not os.path.exists(weightPath):
	raise ValueError("Invalid weight path `" +
						os.path.abspath(weightPath)+"`")
if not os.path.exists(metaPath):
	raise ValueError("Invalid data file path `" +
						os.path.abspath(metaPath)+"`")
if netMain is None:
	netMain = darknet.load_net_custom(configPath.encode(
		"ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
if metaMain is None:
	metaMain = darknet.load_meta(metaPath.encode("ascii"))
if altNames is None:
	try:
		with open(metaPath) as metaFH:
			metaContents = metaFH.read()
			import re
			match = re.search("names *= *(.*)$", metaContents,
								re.IGNORECASE | re.MULTILINE)
			if match:
				result = match.group(1)
			else:
				result = None
			try:
				if os.path.exists(result):
					with open(result) as namesFH:
						namesList = namesFH.read().strip().split("\n")
						altNames = [x.strip() for x in namesList]
			except TypeError:
				pass
	except Exception:
		pass
darknet_image = darknet.make_image(540, 960, 3)
darknet_image1= darknet.make_image(540, 960, 3)
detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
detections1 = darknet.detect_image(netMain, metaMain, darknet_image1, thresh=0.25)

class CameraWidget(QtGui.QWidget):
    """Independent camera feed
    Uses threading to grab IP camera frames in the background

    @param width - Width of the video frame
    @param height - Height of the video frame
    @param stream_link - IP/RTSP/Webcam link
    @param aspect_ratio - Whether to maintain frame aspect ratio or force into fraame
    """
    def is_close(self,p1, p2):
        dst = math.sqrt(p1**2 + p2**2)
        return dst 


    def convertBack(self,x, y, w, h): 
        #================================================================
        # 2.Purpose : Converts center coordinates to rectangle coordinates
        #================================================================  
        """
        :param:
        x, y = midpoint of bbox
        w, h = width, height of the bbox
        
        :return:
        xmin, ymin, xmax, ymax
        """
        xmin = int(round(x - (w / 2)))
        xmax = int(round(x + (w / 2)))
        ymin = int(round(y - (h / 2)))
        ymax = int(round(y + (h / 2)))
        return xmin, ymin, xmax, ymax


    def cvDrawBoxes(self,detections, img):
        """
        :param:
        detections = total detections in one frame
        img = image from detect_image method of darknet
        :return:
        img with bbox
        """
        #================================================================
        # 3.1 Purpose : Filter out Persons class from detections and get 
        #           bounding box centroid for each person detection.
        #================================================================
        if len(detections) > 0:  						# At least 1 detection in the image and check detection presence in a frame  
            centroid_dict = dict() 						# Function creates a dictionary and calls it centroid_dict
            objectId = 0								# We inialize a variable called ObjectId and set it to 0
            for detection in detections:				# In this if statement, we filter all the detections for persons only
                # Check for the only person name tag 
                name_tag = str(detection[0].decode())   # Coco file has string of all the names
                if name_tag == 'person':                
                    x, y, w, h = detection[2][0],\
                                detection[2][1],\
                                detection[2][2],\
                                detection[2][3]      	# Store the center points of the detections
                    if w < 300 and h < 450:
                        xmin, ymin, xmax, ymax = self.convertBack(float(x), float(y), float(w), float(h))   # Convert from center coordinates to rectangular coordinates, We use floats to ensure the precision of the BBox            
                        # Append center point of bbox for persons detected.
                        centroid_dict[objectId] = (int(x), int(y), xmin, ymin, xmax, ymax) # Create dictionary of tuple with 'objectId' as the index center points and bbox
                        objectId += 1 #Increment the index for each detection     
        #=================================================================#
        
        #=================================================================
        # 3.2 Purpose : Determine which person bbox are close to each other
        #=================================================================            	
            red_zone_list = [] # List containing which Object id is in under threshold distance condition. 
            red_line_list = []
            for (id1, p1), (id2, p2) in combinations(centroid_dict.items(), 2): # Get all the combinations of close detections, #List of multiple items - id1 1, points 2, 1,3
                dx, dy = p1[0] - p2[0], p1[1] - p2[1]  	# Check the difference between centroid x: 0, y :1
                distance = self.is_close(dx, dy) 			# Calculates the Euclidean distance
                if distance < config.MIN_DISTANCE:						# Set our social distance threshold - If they meet this condition then..
                    if id1 not in red_zone_list:
                        red_zone_list.append(id1)       #  Add Id to a list
                        red_line_list.append(p1[0:2])   #  Add points to the list
                    if id2 not in red_zone_list:
                        red_zone_list.append(id2)		# Same for the second id 
                        red_line_list.append(p2[0:2])
            
            for idx, box in centroid_dict.items():  # dict (1(key):red(value), 2 blue)  idx - key  box - value
                if idx in red_zone_list:   # if id is in red zone list

                    cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (0, 0, 255), 2) # Create Red bounding boxes  #starting point, ending point size of 2
                else:
                    cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (0, 255, 0), 2) # Create Green bounding boxes
            #=================================================================#

            #=================================================================
            # 3.3 Purpose : Display Risk Analytics and Show Risk Indicators
            #=================================================================        
            text = "Social Distancing Violations: %s" % str(len(red_zone_list))
            # if len(red_zone_list) < config.Count:
            #     payload = {"Camera1WarningBit":"False"}
            #     response = requests.put(url,headers = headers,json = payload)
            # else:
            #     payload = {"Camera1WarningBit":"True"}
            #     response = requests.put(url,headers = headers,json = payload)		# Count People at Risk
            location = (10,900)												# Set the location of the displayed text
            cv2.putText(img, text, location, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)  # Display Text 
            #=================================================================#
        return img

    def __init__(self, width, height, stream_link=0, aspect_ratio=False, parent=None, deque_size=1):
        super(CameraWidget, self).__init__(parent)

        # Initialize deque used to store frames read from the stream
        self.deque = deque(maxlen=deque_size)

        # Slight offset is needed since PyQt layouts have a built in padding
        # So add offset to counter the padding 
        self.offset = 16
        self.screen_width = width - self.offset
        self.screen_height = height - self.offset
        self.maintain_aspect_ratio = aspect_ratio

        self.camera_stream_link = stream_link

        # Flag to check if camera is valid/working
        self.online = False

        self.capture = None
        self.video_frame = QtGui.QLabel()

        self.load_network_stream()

        # Start background frame grabbing
        self.get_frame_thread = Thread(target=self.get_frame, args=())
        self.get_frame_thread.daemon = True
        self.get_frame_thread.start()

        # Periodically set video frame to display
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.set_frame)
        self.timer.start(.5)

        print('Started camera: {}'.format(self.camera_stream_link))

    def load_network_stream(self):
        """Verifies stream link and open new stream if valid"""

        def load_network_stream_thread():
            if self.verify_network_stream(self.camera_stream_link):
                self.capture = cv2.VideoCapture(self.camera_stream_link)
                self.online = True
        self.load_stream_thread = Thread(target=load_network_stream_thread, args=())
        self.load_stream_thread.daemon = True
        self.load_stream_thread.start()

    def verify_network_stream(self, link):
        """Attempts to receive a frame from given link"""

        cap = cv2.VideoCapture(link)
        if not cap.isOpened():
            return False
        cap.release()
        return True

    def get_frame(self):
        """Reads frame, resizes, and converts image to pixmap"""

        while True:
            try:
                if self.capture.isOpened() and self.online:
                    # Read next frame from stream and insert into deque
                    status, frame = self.capture.read()
                    if status:

                        self.deque.append(frame)
                    else:
                        self.capture.release()
                        self.online = False
                else:
                    # Attempt to reconnect
                    print('attempting to reconnect', self.camera_stream_link)
                    self.load_network_stream()
                    self.spin(2)
                self.spin(.001)
            except AttributeError:
                pass

    def spin(self, seconds):
        """Pause for set amount of seconds, replaces time.sleep so program doesnt stall"""

        time_end = time.time() + seconds
        while time.time() < time_end:
            QtGui.QApplication.processEvents()

    def set_frame(self):
        """Sets pixmap image to video frame"""

        if not self.online:
            self.spin(1)
            return

        if self.deque and self.online:
            # Grab latest frame
            frame = self.deque[-1]
            frame_resized = cv2.resize(frame,(540, 960),interpolation=cv2.INTER_LINEAR)
            darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())
            detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
            frame = self.cvDrawBoxes(detections, frame_resized)
		    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB 
            #ret, jpeg = cv2.imencode('.jpg', image)
        # Keep frame aspect ratio
        if self.maintain_aspect_ratio:
            self.frame = imutils.resize(frame, width=self.screen_width)
                # Force resize
        else:
            self.frame = cv2.resize(frame, (self.screen_width, self.screen_height))

        # Add timestamp to cameras
        cv2.rectangle(self.frame, (self.screen_width-190,0), (self.screen_width,50), color=(0,0,0), thickness=-1)
        cv2.putText(self.frame, datetime.now().strftime('%H:%M:%S'), (self.screen_width-185,37), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), lineType=cv2.LINE_AA)

        # Convert to pixmap and set to video frame
        self.img = QtGui.QImage(self.frame, self.frame.shape[1], self.frame.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.pix = QtGui.QPixmap.fromImage(self.img)
        self.video_frame.setPixmap(self.pix)

    def get_video_frame(self):
        return self.video_frame

def exit_application():
    """Exit program event handler"""

    sys.exit(1)

if __name__ == '__main__':

    # Create main application window
    app = QtGui.QApplication([])
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt())
    app.setStyle(QtGui.QStyleFactory.create("Cleanlooks"))
    mw = QtGui.QMainWindow()
    mw.setWindowTitle('Camera GUI')
    mw.setWindowFlags(QtCore.Qt.FramelessWindowHint)

    cw = QtGui.QWidget()
    ml = QtGui.QGridLayout()
    cw.setLayout(ml)
    mw.setCentralWidget(cw)
    mw.showMaximized()

    # Dynamically determine screen width/height
    screen_width = QtGui.QApplication.desktop().screenGeometry().width()
    screen_height = QtGui.QApplication.desktop().screenGeometry().height()

    # Create Camera Widgets 
    #username = 'Your camera username!'
    #password = 'Your camera password!'

    # Stream links
    camera0 = 'sample.mp4'
    camera1 = 'Sample1.mp4'
    camera2 = 'Sample2.mp4'
    #camera3 = "Sample3.mp4"
    #camera4 = "Sample4.mp4"
    #camera3 = 'rtsp://{}:{}@192.168.1.40:554/cam/realmonitor?channel=1&subtype=0'.format(username, password)
    #camera4 = 'rtsp://{}:{}@192.168.1.44:554/cam/realmonitor?channel=1&subtype=0'.format(username, password)
    #camera5 = 'rtsp://{}:{}@192.168.1.42:554/cam/realmonitor?channel=1&subtype=0'.format(username, password)
    #camera6 = 'rtsp://{}:{}@192.168.1.46:554/cam/realmonitor?channel=1&subtype=0'.format(username, password)
    #camera7 = 'rtsp://{}:{}@192.168.1.41:554/cam/realmonitor?channel=1&subtype=0'.format(username, password)

    # Create camera widgets
    print('Creating Camera Widgets...')
    zero = CameraWidget(screen_width//3, screen_height//3, camera0)
    one = CameraWidget(screen_width//3, screen_height//3, camera1)
    two = CameraWidget(screen_width//3, screen_height//3, camera2)
    #three = CameraWidget(screen_width//3, screen_height//3, camera3)
    #four = CameraWidget(screen_width//3, screen_height//3, camera4)
    #five = CameraWidget(screen_width//3, screen_height//3, camera5)
    #six = CameraWidget(screen_width//3, screen_height//3, camera6)
    #seven = CameraWidget(screen_width//3, screen_height//3, camera7)

    # Add widgets to layout
    print('Adding widgets to layout...')
    ml.addWidget(zero.get_video_frame(),0,0,1,1)
    ml.addWidget(one.get_video_frame(),0,1,1,1)
    ml.addWidget(two.get_video_frame(),0,2,1,1)
    # ml.addWidget(three.get_video_frame(),1,0,1,1)
    # ml.addWidget(four.get_video_frame(),1,1,1,1)
    #ml.addWidget(five.get_video_frame(),1,2,1,1)
    #ml.addWidget(six.get_video_frame(),2,0,1,1)
    #ml.addWidget(seven.get_video_frame(),2,1,1,1)

    print('Verifying camera credentials...')

    mw.show()

    QtGui.QShortcut(QtGui.QKeySequence('Ctrl+Q'), mw, exit_application)

    if(sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
