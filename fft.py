#You need to tell the program Fs and Time_length
#You also need give the signal x,y,z (1-d float array)

#X_sum_from_0_3Hz_to_15Hz is the power form 0.3Hz to 15Hz of siganl x, which is a float number
#Y_sum_from_0_3Hz_to_15Hz, Z_sum_from_0_3Hz_to_15Hz, SMV_sum_from_0_3Hz_to_15Hz are similar

#X_sum_from_0_6Hz_to_2_5Hz is the power form 0.6Hz to 2.5Hz of siganl x, c
#Y_sum_from_0_6Hz_to_2_5Hz, Z_sum_from_0_6Hz_to_2_5Hz, SMV_sum_from_0_6Hz_to_2_5Hz are similar

#X_sum_domain is the power near the domain frequency F of siganl x, which is a float number 
#the range is [F-0.1, F+0.1]. If F is smaller than 0.1Hz, the range is [0,F+0.1]
#Y_sum_domain, Z_sum_domain , SMV_sum_domain are similar

#X_ratio is the ratio between the power of domian frequency and the total power of 0.3Hz to 15Hz of signal x, which is a float number
#Y_ratio, Z_ratio, SMV_ratio are similar


import numpy as np
import math
import matplotlib.pyplot as plt

#Basic parameter of this program
#------------------------------------------------------------------------------------------------------------------------------
Fs = 100 #sample rate
Ts = 1.0/Fs #sample interval
Time_length = 4.0 #time length of the total signal
n = int(Fs*Time_length) #number of sample points, For frequency domian the number of data is the also the same
frq_interval = 1.0/Time_length #the frquency interval between each point
index_of_0_3Hz = int(math.ceil(0.3/frq_interval))
index_of_0_6Hz = int(math.ceil(0.6/frq_interval))
index_of_2_5Hz = int(math.ceil(2.5/frq_interval))
index_of_15Hz = int(math.floor(15/frq_interval))
num_of_points_in_0_1Hz  = int(0.1/frq_interval)
#------------------------------------------------------------------------------------------------------------------------------


#generate signals
#------------------------------------------------------------------------------------------------------------------------------
t = np.arange(0,Time_length,Ts) # Time vector. Only useful for creating new signal. Delete it when we have our own signal
x = np.sin(2*np.pi*5*t) #T his is the time domian siganl. Change to our signal later
y = np.sin(2*np.pi*10*t)
z = np.sin(2*np.pi*15*t)
smv = list(map(lambda x: math.sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]), zip(x, y, z))) 
#------------------------------------------------------------------------------------------------------------------------------


#calculate frequency
#------------------------------------------------------------------------------------------------------------------------------
k = np.arange(n)
frq = k/Time_length # two sides frequency range
frq = frq[range(n/2)] # one side frequency range
#------------------------------------------------------------------------------------------------------------------------------


#fft and calculate power spectrum
#------------------------------------------------------------------------------------------------------------------------------
X = np.fft.fft(x) # fft computing
X = X[range(n/2)]/Fs*2 # fft normalization
XW = list(map(lambda x: abs(x[0])*abs(x[0]/Time_length), zip(X))) # power spectrum
Y = np.fft.fft(y) # fft computing
Y = Y[range(n/2)]/Fs*2 # fft normalization
YW = list(map(lambda x: abs(x[0])*abs(x[0]/Time_length), zip(Y))) # power spectrum
Z = np.fft.fft(z) # fft computing
Z = Z[range(n/2)]/Fs*2 # fft normalization
ZW = list(map(lambda x: abs(x[0])*abs(x[0]/Time_length), zip(Z))) # power spectrum
SMV = np.fft.fft(smv) # fft computing
SMV = SMV[range(n/2)]/Fs*2 # fft normalization
SMVW = list(map(lambda x: abs(x[0])*abs(x[0]/Time_length), zip(SMV))) # power spectrum
#------------------------------------------------------------------------------------------------------------------------------


#interval of different frequency range
#------------------------------------------------------------------------------------------------------------------------------
#from 0.3Hz to 15Hz
X_sum_from_0_3Hz_to_15Hz = 0; #power form 0.3Hz to 15Hz for X
for i in range(index_of_0_3Hz,index_of_15Hz+1):
	X_sum_from_0_3Hz_to_15Hz = X_sum_from_0_3Hz_to_15Hz + XW[i];
X_sum_from_0_3Hz_to_15Hz = X_sum_from_0_3Hz_to_15Hz/Time_length

Y_sum_from_0_3Hz_to_15Hz = 0; #power form 0.3Hz to 15Hz for Y
for i in range(index_of_0_3Hz,index_of_15Hz+1):
	Y_sum_from_0_3Hz_to_15Hz = Y_sum_from_0_3Hz_to_15Hz + YW[i];
Y_sum_from_0_3Hz_to_15Hz = Y_sum_from_0_3Hz_to_15Hz/Time_length

Z_sum_from_0_3Hz_to_15Hz = 0; #power form 0.3Hz to 15Hz for Z
for i in range(index_of_0_3Hz,index_of_15Hz+1):
	Z_sum_from_0_3Hz_to_15Hz = Z_sum_from_0_3Hz_to_15Hz + ZW[i];
Z_sum_from_0_3Hz_to_15Hz = Z_sum_from_0_3Hz_to_15Hz/Time_length

SMV_sum_from_0_3Hz_to_15Hz = 0; #power form 0.3Hz to 15Hz for SMV
for i in range(index_of_0_3Hz,index_of_15Hz+1):
	SMV_sum_from_0_3Hz_to_15Hz = SMV_sum_from_0_3Hz_to_15Hz + SMVW[i];
SMV_sum_from_0_3Hz_to_15Hz = SMV_sum_from_0_3Hz_to_15Hz/Time_length
#------------------------------------------------------------------------------------------------------------------------------

#from 0.6Hz to 2.5Hz
X_sum_from_0_6Hz_to_2_5Hz = 0; #power form 0.6Hz to 2.5Hz for X
for i in range(index_of_0_6Hz,index_of_2_5Hz+1):
	X_sum_from_0_6Hz_to_2_5Hz = X_sum_from_0_6Hz_to_2_5Hz + XW[i];
X_sum_from_0_6Hz_to_2_5Hz = X_sum_from_0_6Hz_to_2_5Hz/Time_length

Y_sum_from_0_6Hz_to_2_5Hz = 0; #power form 0.6Hz to 2.5Hz for Y
for i in range(index_of_0_6Hz,index_of_2_5Hz+1):
	Y_sum_from_0_6Hz_to_2_5Hz = Y_sum_from_0_6Hz_to_2_5Hz + YW[i];
Y_sum_from_0_6Hz_to_2_5Hz = Y_sum_from_0_6Hz_to_2_5Hz/Time_length

Z_sum_from_0_6Hz_to_2_5Hz = 0; #power form 0.6Hz to 2.5Hz for Z
for i in range(index_of_0_6Hz,index_of_2_5Hz+1):
	Z_sum_from_0_6Hz_to_2_5Hz = Z_sum_from_0_6Hz_to_2_5Hz + ZW[i];
Z_sum_from_0_6Hz_to_2_5Hz = Z_sum_from_0_6Hz_to_2_5Hz/Time_length

SMV_sum_from_0_6Hz_to_2_5Hz = 0; #power form 0.6Hz to 2.5Hz for X
for i in range(index_of_0_6Hz,index_of_2_5Hz+1):
	SMV_sum_from_0_6Hz_to_2_5Hz = SMV_sum_from_0_6Hz_to_2_5Hz + SMVW[i];
SMV_sum_from_0_6Hz_to_2_5Hz = SMV_sum_from_0_6Hz_to_2_5Hz/Time_length
#------------------------------------------------------------------------------------------------------------------------------

#power of domain frequency + - 0.1Hz
#find index of domian frequency
X_domain_frq_index = XW.index(max(XW))
Y_domain_frq_index = YW.index(max(YW))
Z_domain_frq_index = ZW.index(max(ZW))
SMV_domain_frq_index = SMVW.index(max(SMVW))

X_sum_domain = 0; #power form -0.1Hz to 0.1Hz of domain frequency for X
head_index = max(0,X_domain_frq_index-num_of_points_in_0_1Hz)
tail_index = X_domain_frq_index+num_of_points_in_0_1Hz+1
for i in range(head_index,tail_index):
	X_sum_domain = X_sum_domain + XW[i];
X_sum_domain = X_sum_domain/Time_length

Y_sum_domain = 0; #power form -0.1Hz to 0.1Hz of domain frequency for X
head_index = max(0,Y_domain_frq_index-num_of_points_in_0_1Hz)
tail_index = Y_domain_frq_index+num_of_points_in_0_1Hz+1
for i in range(head_index,tail_index):
	Y_sum_domain = Y_sum_domain + YW[i];
Y_sum_domain = Y_sum_domain/Time_length

Z_sum_domain = 0; #power form -0.1Hz to 0.1Hz of domain frequency for X
head_index = max(0,Z_domain_frq_index-num_of_points_in_0_1Hz)
tail_index = Z_domain_frq_index+num_of_points_in_0_1Hz+1
for i in range(head_index,tail_index):
	Z_sum_domain = Z_sum_domain + ZW[i];
Z_sum_domain = Z_sum_domain/Time_length

SMV_sum_domain = 0; #power form -0.1Hz to 0.1Hz of domain frequency for X
head_index = max(0,SMV_domain_frq_index-num_of_points_in_0_1Hz)
tail_index = SMV_domain_frq_index+num_of_points_in_0_1Hz+1
for i in range(head_index,tail_index):
	SMV_sum_domain = SMV_sum_domain + SMVW[i];
SMV_sum_domain = SMV_sum_domain/Time_length
#------------------------------------------------------------------------------------------------------------------------------

#ratio between the power of domain frequency and the total power of 0.3Hz to 15Hz
X_ratio = X_sum_domain/X_sum_from_0_3Hz_to_15Hz
Y_ratio = Y_sum_domain/Y_sum_from_0_3Hz_to_15Hz
Z_ratio = Z_sum_domain/Z_sum_from_0_3Hz_to_15Hz
SMV_ratio = SMV_sum_domain/SMV_sum_from_0_3Hz_to_15Hz
#------------------------------------------------------------------------------------------------------------------------------


'''
#Draw and save pictures 
#------------------------------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(2, 1)
ax[0].plot(t,x) #plotting the signal in time domain
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Amplitude')
ax[1].plot(frq,XW,'r') #plotting the spectrum
ax[1].set_xlabel('Freq (Hz)')
ax[1].set_ylabel('|X(W/Hz)|')
plt.savefig('plotX.png', format = 'png')

fig, ax = plt.subplots(2, 1)
ax[0].plot(t,y) #plotting the signal in time domain
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Amplitude')
ax[1].plot(frq,YW,'r') #plotting the spectrum
ax[1].set_xlabel('Freq (Hz)')
ax[1].set_ylabel('|Y(W/Hz)|')
plt.savefig('plotY.png', format = 'png')

fig, ax = plt.subplots(2, 1)
ax[0].plot(t,z) #plotting the signal in time domain
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Amplitude')
ax[1].plot(frq,ZW,'r') #plotting the spectrum
ax[1].set_xlabel('Freq (Hz)')
ax[1].set_ylabel('|Z(W/Hz)|')
plt.savefig('plotZ.png', format = 'png')

fig, ax = plt.subplots(2, 1)
ax[0].plot(t,smv) #plotting the signal in time domain
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Amplitude')
ax[1].plot(frq,SMVW,'r') #plotting the spectrum
ax[1].set_xlabel('Freq (Hz)')
ax[1].set_ylabel('|SMV(W/Hz)|')
plt.savefig('plotSMV.png', format = 'png')
'''
