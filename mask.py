import cv2
import numpy as np
def get_mask_by_type(img, type):
	# type is white (default) or black

	# Create HSV and grayscale formats
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Define variables used in masking
	value_range = 80
	if (type == "black"):
		min_value = np.amin(gray)
		max_value = min_value + value_range
	else:
		max_value = np.amax(gray)
		min_value = max_value - value_range
	min_color = np.array([0, 0, min_value])
	max_color = np.array([255, 255, max_value])

	# Create initial region mask
	reg = cv2.inRange(hsv, min_color, max_color)

	# Define morphologial transformation kernel
	kernel = np.ones((3,3),np.uint8)

	# Create edges and dilate to get better results
	edges = cv2.Canny(gray, 100, 150)
	edges = cv2.dilate(edges, kernel)
	edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

	# Intersect region and edges
	mask = reg & edges
	
	return mask

def get_mask(img):
	white = get_mask_by_type(img, "white")
	black = get_mask_by_type(img, "black")
	mask = white + black
	kernel = np.ones((3, 3),np.uint8)
	mask = cv2.dilate(mask, kernel)

	return mask