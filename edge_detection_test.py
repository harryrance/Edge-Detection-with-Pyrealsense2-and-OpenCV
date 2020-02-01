import cv2
import numpy as np
import pyrealsense2 as rs
from matplotlib import pyplot as plt
import math

#Start by configuring the depth and colour streams (Realsense)
pipeline =rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

#Start Streaming
pipeline.start(config)

try:
	while True:
		
		#Wait for pair of frames
		frames = pipeline.wait_for_frames()
		depth_frame = frames.get_depth_frame()
		color_frame = frames.get_color_frame()
		if not depth_frame or not color_frame:
			continue
		
		#Convert images to numpy arrays
		depth_image = np.asanyarray(depth_frame.get_data())
		color_image = np.asanyarray(color_frame.get_data())
		
		#Apply colormap on depth image (image must be converted to 8-bit)
		depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
		
		#Stack images horizontally
		#images = np.hstack((color_image, depth_colormap))
	
		
		color_image_b = cv2.GaussianBlur(color_image, (5,5), 0)
		edges = cv2.Canny(color_image_b, 100, 100)
		
		lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=30)
		if lines is not None:
			for line in lines:
				x1, y1, x2, y2 = line[0]
				if math.sqrt((y1-y2)*(y1-y2)) < 15:
					cv2.line(color_image, (x1, y1), (x2, y2), (0,0,255), 2)
		
		#Show images
		cv2.namedWindow('Realsense', cv2.WINDOW_AUTOSIZE)
		cv2.imshow('Realsense', color_image)
		cv2.waitKey(1)
		
finally:
	
	#Stop streaming
	pipeline.stop()
