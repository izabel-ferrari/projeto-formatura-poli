import numpy as np
import cv2

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

	# Intersect region and edges
	mask = reg & edges

	return mask

def get_mask(img):
	white = get_mask_by_type(img, "white")
	black = get_mask_by_type(img, "black")
	mask = white + black
	kernel = np.ones((3, 3),np.uint8)
	mask = cv2.dilate(mask, kernel)
	areaThreshold = 400

	im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	for i, contour in enumerate(contours):
		if(cv2.contourArea(contour) < areaThreshold):
			leftmost = contour[contour[:,:,0].argmin()][0][0]
			rightmost = contour[contour[:,:,0].argmax()][0][0]
			topmost = contour[contour[:,:,1].argmin()][0][1]
			bottommost = contour[contour[:,:,1].argmax()][0][1]

			topleft = (leftmost, topmost)
			bottomright = (rightmost, bottommost)

			cv2.rectangle(mask, topleft, bottomright, 255, -1)
	return mask

def remove_eyes_from_mask(face_mask, true_eyes):
    try:
        for (ex, ey, ew, eh) in true_eyes:
            for m in face_mask[ey:(ey + eh + 1)]:
                m[ex:(ex + ew + 1)] = 0
    except:
        pass
    return
