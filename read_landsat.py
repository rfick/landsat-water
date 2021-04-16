from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import cv2
import math
import os

# Computes estimated chlorophyll-a concentration according to the CI algorithm
# See: https://oceancolor.gsfc.nasa.gov/atbd/chlor_a/
def OCI(Rrs_red, Rrs_green, Rrs_blue, wv_red, wv_green, wv_blue):
	return Rrs_green - [Rrs_blue + ((wv_green - wv_blue)/(wv_red - wv_blue))*(Rrs_red - Rrs_blue)]

# Computes estimated chlorophyll-a concentration according to the OCx algorithm
# See: https://oceancolor.gsfc.nasa.gov/atbd/chlor_a/
def OCX(Rrs_blue, Rrs_green, a0, a1, a2, a3, a4):
	return np.power(10, a0 + a1*(np.log10(Rrs_blue/Rrs_green)) + a2*np.power(np.log10(Rrs_blue/Rrs_green), 2) + a3*np.power(np.log10(Rrs_blue/Rrs_green), 3) + a4*np.power(np.log10(Rrs_blue/Rrs_green), 4))

data_folder = 'landsat8_data'
data_list = os.listdir(data_folder)

for i in range(len(data_list)):
	file_id = data_list[i]

	im_blue = io.imread(data_folder + '/' + file_id + '/' + file_id + '_SR_B1.tif')
	im_green = io.imread(data_folder + '/' + file_id + '/' + file_id +  '_SR_B3.tif')
	im_red = io.imread(data_folder + '/' + file_id + '/' + file_id +  '_SR_B4.tif')
	im_rgb = np.zeros((im_blue.shape[0], im_blue.shape[1], 3))

	im_QA = io.imread(data_folder + '/' + file_id + '/' + file_id +  '_QA_PIXEL.tif')

	# Cloud bit is in position 5
	water_mask = (im_QA >> 4) & 0b11111

	water_mask_hold = water_mask

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
	latNW = 27.117033
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

	plt.figure()
	plt.imshow(water_mask_hold[latPixelDiffNW:latPixelDiffSE, lonPixelDiffNW:lonPixelDiffSE])
	plt.colorbar()

	# Characteristics of Landsat 8
	# See: https://en.wikipedia.org/wiki/Landsat_8
	wv_blue = 443
	wv_green = 562.5
	wv_red = 655

	# Pulling some coefficients for the OCTS sensor from https://oceancolor.gsfc.nasa.gov/atbd/chlor_a/
	a0 = 0.2412
	a1 = -2.0546
	a2 = 1.1776
	a3 = -0.5538
	a4 = -0.4570

	#nonzero_blue = np.zeros_like(im_blue)
	#nonzero_blue[im_blue==0] = 1

	#plt.figure()
	#plt.imshow(nonzero_blue)
	#plt.show()
	#asldkj

	#blue_flatten = im_blue.flatten()
	#plt.figure()
	#plt.hist(blue_flatten, bins=100)
	#green_flatten = im_green.flatten()
	#plt.figure()
	#plt.hist(green_flatten, bins=100)
	#red_flatten = im_red.flatten()
	#plt.figure()
	#plt.hist(red_flatten, bins=100)
	#plt.show()

	# Scale values from 1-65535 to 0-1
	im_blue = im_blue/65535
	im_green = im_green/65535
	im_red = im_red/65535

	#CI = OCI(im_red, im_green, im_blue, wv_red, wv_green, wv_blue)
	#CI = np.squeeze(CI)

	#CX = OCX(im_blue, im_green, a0, a1, a2, a3, a4)

	im_rgb[:, :, 0] = im_red
	im_rgb[:, :, 1] = im_green
	im_rgb[:, :, 2] = im_blue

	im_rgb = np.sqrt(im_rgb)

	#plt.figure()
	#plt.imshow(im_rgb)

	water_rgb = np.zeros_like(im_rgb)
	cv2.bitwise_and(im_rgb, im_rgb, water_rgb, water_mask)

	plt.figure()
	plt.imshow(water_rgb[latPixelDiffNW:latPixelDiffSE, lonPixelDiffNW:lonPixelDiffSE, :])
	plt.title(file_id)

	#plt.figure()
	#plt.imshow(water_mask[latPixelDiffNW:latPixelDiffSE, lonPixelDiffNW:lonPixelDiffSE])
	#plt.colorbar()

	#plt.figure()
	#plt.imshow(CI[latPixelDiffNW:latPixelDiffSE, lonPixelDiffNW:lonPixelDiffSE])
	#plt.colorbar()

	#plt.figure()
	#plt.imshow(CX[latPixelDiffNW:latPixelDiffSE, lonPixelDiffNW:lonPixelDiffSE])
	#plt.clim(0, 10)
	#plt.colorbar()

plt.show()