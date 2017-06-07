import numpy as np
import math

def get_range_of_linear_acceleration(dataPoint,bottom,roof):
	#This function return the range of linear acceleration(x,y,z) in 4s.
	#The range can be modified by bottom and roof
	#For example, if the acceleration is 1,2,3......,100
	#If bottom is 0.01 and roof is 1, then the range is 100-1=99
	#If bottom is 0.25 and roof is 0.75 then the range is 75-25=50

	x = np.array([p[1] for p in dataPoint])
	y = np.array([p[2] for p in dataPoint])
	z = np.array([p[3] for p in dataPoint])
	x = np.sort(x)
	y = np.sort(y)
	z = np.sort(z)
	length = len(x)

	x_roof = x[int(length*roof)]
	x_bottom = x[int(length*bottom)]
	y_roof = x[int(length*roof)]
	y_bottom = x[int(length*bottom)]
	z_roof = x[int(length*roof)]
	z_bottom = x[int(length*bottom)]

	x_range = x_roof-x_bottom
	y_range = y_roof-y_bottom
	z_range = z_roof-z_bottom

	result = [x_range, y_range, z_range]
	return result


def get_integral_of_linear_acceleration(dataPoint):
	#This function return velocity from integral of linear acceleration(x,y,z) in 4s.
	x = np.array([p[1] for p in dataPoint])
	y = np.array([p[2] for p in dataPoint])
	z = np.array([p[3] for p in dataPoint])

	result = [np.sum(x), np.sum(y), np.sum(z)]
	return result
	

def get_range_of_gravity_vector(dataPoint,bottom,roof):
	#This function return the range of gravity vector(x,y,z) in 4s.
	#The range can be modified by bottom and roof
	#For example, if the gravity vector is 1,2,3......,100
	#If bottom is 0.01 and roof is 1, then the range is 100-1=99
	#If bottom is 0.25 and roof is 0.75 then the range is 75-25=50

	x = np.array([p[1] for p in dataPoint])
	y = np.array([p[2] for p in dataPoint])
	z = np.array([p[3] for p in dataPoint])
	x = np.sort(x)
	y = np.sort(y)
	z = np.sort(z)
	length = len(x)

	x_roof = x[int(length*roof)]
	x_bottom = x[int(length*bottom)]
	y_roof = x[int(length*roof)]
	y_bottom = x[int(length*bottom)]
	z_roof = x[int(length*roof)]
	z_bottom = x[int(length*bottom)]

	x_range = x_roof-x_bottom
	y_range = y_roof-y_bottom
	z_range = z_roof-z_bottom

	result = [x_range, y_range, z_range]
	return result


def get_Kurtosis_of_gravity_vector(dataPoint):
	#This function return the Kurtosis of gravity vector(x,y,z) in 4s.

	x = np.array([p[1] for p in dataPoint])
	y = np.array([p[2] for p in dataPoint])
	z = np.array([p[3] for p in dataPoint])
	length = len(x)

	x_mean = np.mean(x)
	y_mean = np.mean(y)
	z_mean = np.mean(z)

	x_m2 = 0
	x_m4 = 0
	for i in x:
		x_m2 = x_m2+(i-x_mean)**2
		x_m4 = x_m4+(i-x_mean)**4
	x_m2 = x_m2/length
	x_m4 = x_m4/length
	x_Kurtosis = x_m4/x_m2**2-3

	y_m2 = 0
	y_m4 = 0
	for i in x:
		y_m2 = y_m2+(i-y_mean)**2
		y_m4 = y_m4+(i-y_mean)**4
	y_m2 = y_m2/length
	y_m4 = y_m4/length
	y_Kurtosis = y_m4/y_m2**2-3

	z_m2 = 0
	z_m4 = 0
	for i in x:
		z_m2 = z_m2+(i-z_mean)**2
		z_m4 = z_m4+(i-z_mean)**4
	z_m2 = z_m2/length
	z_m4 = z_m4/length
	z_Kurtosis = z_m4/z_m2**2-3

	result = [x_Kurtosis, y_Kurtosis, z_Kurtosis]
	return result






