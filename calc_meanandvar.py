from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import cv2
import math
import os

# For a new value newValue, compute the new count, new mean, the new M2.
# mean accumulates the mean of the entire dataset
# M2 aggregates the squared distance from the mean
# count aggregates the number of samples seen so far
def update(count, mean, M2, newValue):
	count += 1
	delta = newValue - mean
	mean += delta / count
	delta2 = newValue - mean
	M2 += delta * delta2
	return count, mean, M2

# Retrieve the mean, variance and sample variance from an aggregate
def finalize(count, mean, M2):
	if count < 2:
		return float("nan"), float("nan")
	else:
		(mean, variance) = (mean, M2 / count)
		return mean, variance

data_folder = 'landsat8_data'
data_list = os.listdir(data_folder)

for i in range(len(data_list)):
	file_id = data_list[i]

	im_blue = io.imread(data_folder + '/' + file_id + '/' + file_id + '_SR_B2.tif')
	im_green = io.imread(data_folder + '/' + file_id + '/' + file_id +  '_SR_B3.tif')
	im_red = io.imread(data_folder + '/' + file_id + '/' + file_id +  '_SR_B4.tif')
	im_rgb = np.zeros((im_blue.shape[0], im_blue.shape[1], 3))

	im_QA = io.imread(data_folder + '/' + file_id + '/' + file_id +  '_QA_PIXEL.tif')

	# Cloud bit is in position 5
	water_mask = (im_QA >> 4) & 0b11111

	water_mask = water_mask == 28
	water_mask = water_mask.astype(np.uint8)

	CORNER_UL_LAT_PRODUCT = 0
	CORNER_UL_LON_PRODUCT = 0
	CORNER_UR_LAT_PRODUCT = 0
	CORNER_UR_LON_PRODUCT = 0
	CORNER_LL_LAT_PRODUCT = 0
	CORNER_LL_LON_PRODUCT = 0
	CORNER_LR_LAT_PRODUCT = 0
	CORNER_LR_LON_PRODUCT = 0

	metadatafile = open(data_folder + '/' + file_id + '/' + file_id + '_MTL.txt', 'r')
	for line in metadatafile:
		if('CORNER_UL_LAT_PRODUCT' in line):
			CORNER_UL_LAT_PRODUCT = float(line.split("= ",1)[1])
		if('CORNER_UL_LON_PRODUCT' in line):
			CORNER_UL_LON_PRODUCT = float(line.split("= ",1)[1])
		if('CORNER_UR_LAT_PRODUCT' in line):
			CORNER_UR_LAT_PRODUCT = float(line.split("= ",1)[1])
		if('CORNER_UR_LON_PRODUCT' in line):
			CORNER_UR_LON_PRODUCT = float(line.split("= ",1)[1])
		if('CORNER_LL_LAT_PRODUCT' in line):
			CORNER_LL_LAT_PRODUCT = float(line.split("= ",1)[1])
		if('CORNER_LL_LON_PRODUCT' in line):
			CORNER_LL_LON_PRODUCT = float(line.split("= ",1)[1])
		if('CORNER_LR_LAT_PRODUCT' in line):
			CORNER_LR_LAT_PRODUCT = float(line.split("= ",1)[1])
		if('CORNER_LR_LON_PRODUCT' in line):
			CORNER_LR_LON_PRODUCT = float(line.split("= ",1)[1])
	metadatafile.close()

	# Lat + long of bounding box of the study area (NW corner and SE corner)
	latNW = 26.984196
	longNW = -82.513229
	latSE = 26.358022
	longSE = -81.693427

	latDiff = CORNER_LL_LAT_PRODUCT - CORNER_UL_LAT_PRODUCT
	lonDiff = CORNER_UR_LON_PRODUCT - CORNER_UL_LON_PRODUCT
	latDiffPerPixel = latDiff/im_rgb.shape[0]
	lonDiffPerPixel = lonDiff/im_rgb.shape[1]

	latPixelDiffNW = int((latNW - CORNER_UL_LAT_PRODUCT)/latDiffPerPixel)
	lonPixelDiffNW = int((longNW - CORNER_UL_LON_PRODUCT)/lonDiffPerPixel)
	latPixelDiffSE = int((latSE - CORNER_UL_LAT_PRODUCT)/latDiffPerPixel)
	lonPixelDiffSE = int((longSE - CORNER_UL_LON_PRODUCT)/lonDiffPerPixel)

	if(latPixelDiffNW < 0):
		latPixelDiffNW = 0
	if(latPixelDiffNW > im_rgb.shape[0]-1):
		latPixelDiffNW = im_rgb.shape[0]-1
	if(lonPixelDiffNW < 0):
		lonPixelDiffNW = 0
	if(lonPixelDiffNW > im_rgb.shape[1]-1):
		lonPixelDiffNW = im_rgb.shape[1]-1
	if(latPixelDiffSE < 0):
		latPixelDiffSE = 0
	if(latPixelDiffSE > im_rgb.shape[0]-1):
		latPixelDiffSE = im_rgb.shape[0]-1
	if(lonPixelDiffSE < 0):
		lonPixelDiffSE = 0
	if(lonPixelDiffSE > im_rgb.shape[1]-1):
		lonPixelDiffSE = im_rgb.shape[1]-1

	# Scale values from 1-65535 to 0-1
	im_blue = im_blue/65535
	im_green = im_green/65535
	im_red = im_red/65535

	im_rgb[:, :, 0] = im_red
	im_rgb[:, :, 1] = im_green
	im_rgb[:, :, 2] = im_blue

	water_rgb = np.zeros_like(im_rgb)
	cv2.bitwise_and(im_rgb, im_rgb, water_rgb, water_mask)

	study_area = water_rgb[latPixelDiffNW:latPixelDiffSE, lonPixelDiffNW:lonPixelDiffSE, :]

	print('file {}'.format(i))

	# First time through, initialize empty arrays
	if(i==0):
		#Keeps running count, mean, and M2 for B, G, and R bands
		runningAggregates = np.zeros((study_area.shape[0], study_area.shape[1], 9))
		for j in range(runningAggregates.shape[0]):
			for k in range(runningAggregates.shape[1]):
				if(study_area[j, k, 0]!=0 or study_area[j, k, 1]!=0 or study_area[j, k, 2]!=0):
					for band in range(3):
						runningAggregates[j, k, 3*band], runningAggregates[j, k, 3*band+1], runningAggregates[j, k, 3*band+2] = update(runningAggregates[j, k, 3*band], runningAggregates[j, k, 3*band+1], runningAggregates[j, k, 3*band+2], study_area[j, k, band])
	else:
		for j in range(study_area.shape[0]):
			for k in range(study_area.shape[1]):
				if((study_area[j, k, 0]!=0 or study_area[j, k, 1]!=0 or study_area[j, k, 2]!=0) and j<runningAggregates.shape[0] and k<runningAggregates.shape[1]):
					for band in range(3):
						runningAggregates[j, k, 3*band], runningAggregates[j, k, 3*band+1], runningAggregates[j, k, 3*band+2] = update(runningAggregates[j, k, 3*band], runningAggregates[j, k, 3*band+1], runningAggregates[j, k, 3*band+2], study_area[j, k, band])

meanAndVar = np.zeros((study_area.shape[0], study_area.shape[1], 6))

for j in range(runningAggregates.shape[0]):
	for k in range(runningAggregates.shape[1]):
		for band in range(3):
			meanAndVar[j, k, 2*band], meanAndVar[j, k, 2*band+1] = finalize(runningAggregates[j, k, 3*band], runningAggregates[j, k, 3*band+1], runningAggregates[j, k, 3*band+2])

np.save('meanAndVar.npy', meanAndVar)