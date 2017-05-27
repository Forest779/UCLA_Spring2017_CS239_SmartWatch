import gzip
import numpy as np
import os
import csv
from scipy.interpolate import interp1d

class data_reader():
	def __init__(self):
		self.feature_final=None
		self.label_Voted=None

		self.cuttingpoint = None #To Hao: When you generate the cuttingpoint, assign this so that I can use in my function

	def __get_start_end(self,path):
		start_list=[]
		end_list=[]
		for root, dirs, files in os.walk(path):
			for filename in files:
				if filename[-3:]=='.gz' and 'sensor' in filename and 'significant_motion' not in filename and 'step' not in filename and 'light' not in filename:
					full_path=root+'/'+filename
					csvfile=gzip.open(full_path, 'rb')
					reader=csv.reader(csvfile)
					IsFirstRow=True
					for row in reader:
						num=long(row[0])
						if IsFirstRow:
							start_list.append(num)
							IsFirstRow=False
							start=num+12000
						last=num
					end_list.append(last)
		start=max(start_list)+12000 #skip the first 1200 ms
		end=min(end_list)
		if start<=end:
			return (start,end)
		else:
			return None

	def __generate_cutting_point(self,start_end_tup):
		interval=4000
		point=start_end_tup[0]
		cuttingpoints=[]
		while point+4000<start_end_tup[1]:
			cuttingpoints.append(point)
			point+=4000
		return cuttingpoints

	def __process_batch(self,batch):
		if not batch:
			print 'Warning! the batch is empty'
		a=np.array(batch)
		feature_line=[]
		if True:# All the batch has the pure label in the training set
			for i in range(1,4):
				feature_line.append(max(a[:,i].astype(np.float)))
				feature_line.append(min(a[:,i].astype(np.float)))
				feature_line.append(np.average(a[:,i].astype(np.float)))
				feature_line.append(np.std(a[:,i].astype(np.float)))
			return (feature_line,[batch[0][-1]])
		else:
			return None

	def __BM(self,line):
		major=line[0]
		count=1
		for m in line:
			if m==major:
				count+=1
			else:
				count-=1
			if count==0:
				major=m
				count=1
		count=0
		for m in line:
			if m==major:
				count+=1
		if count>len(line)/2:
			return major
		else:
			return None

	def __major_vote(self,label_toVote):
		result=[]
		for line in label_toVote:
			major_label=self.__BM(line)
			if major_label:
				result.append([major_label])
			else:
				result.append(['None'])
		return np.asarray(result)

	def __fileToArray(self,filepath,cuttingpoints):
		csvfile=gzip.open(filepath,'rb')
		reader=csv.reader(csvfile)
		first_batch=True
		batch=[]
		feature_Array=[]
		label_Array=[]
		cutting_index=0
		for row in reader:
			cur_time=long(row[0])
			if cutting_index+1>len(cuttingpoints)-1: #The last cutting_point is the boundry
				break
			if cur_time<cuttingpoints[cutting_index]:
				continue
			if cur_time>=cuttingpoints[cutting_index] and cur_time<cuttingpoints[cutting_index+1]:
				batch.append(row)
			if cur_time>cuttingpoints[cutting_index+1]:
				line=self.__process_batch(batch)
				feature_Array.append(line[0])
				label_Array.append(line[1])
				batch=[]
				batch.append(row)
				cutting_index+=1
		return (np.asarray(feature_Array),np.asarray(label_Array))

	def __same_dir_mutipleFileToArray(self,root_path,cuttingpoints):
		feature_array_list=[]
		label_array_list=[]
		for root, dirs, files in os.walk(root_path):
			for filename in files:
				if filename[-3:]=='.gz' and 'sensor' in filename and 'significant_motion' not in filename and 'step' not in filename and 'light' not in filename:
					print 'Reading... '+filename
					filepath=root_path+'/'+filename
					feature_label=self.__fileToArray(filepath,cuttingpoints)
					print 'Length of data after processing '+str(len(feature_label[1]))+'\n'
					feature_array_list.append(feature_label[0])
					label_array_list.append(feature_label[1])
		return (np.concatenate(feature_array_list,axis=1),np.concatenate(label_array_list,axis=1))

	def __multiple_dir_multiFileArray(self,person_datapath):
		feature_array_list=[]
		label_array_list=[]
		for root, dirs, files in os.walk(person_datapath):
			if 'data' in root:
				start_end_tup=self.__get_start_end(root)
				cuttingpoints=self.__generate_cutting_point(start_end_tup)

				self.cuttingpoint = cuttingpoints

				feature_label=self.__same_dir_mutipleFileToArray(root,cuttingpoints)
				feature_array_list.append(feature_label[0])
				label_array_list.append(feature_label[1])
		self.feature_final=np.concatenate(feature_array_list,axis=0)
		label_toVote=np.concatenate(label_array_list,axis=0)
		self.label_Voted=self.__major_vote(label_toVote)
		return (self.feature_final,self.label_Voted)
		#return np.concatenate([feature_final,label_Voted],axis=1)

	def read_array(self,person_datapath):
		return self.__multiple_dir_multiFileArray(person_datapath)

	def save_data(self,filename):
		np.savetxt(filename+'.feature.mydata',self.feature_final,delimiter=',')
		np.savetxt(filename+'.label.mydata',self.label_Voted, fmt='%s',delimiter=',')


	def handle_step(self,filename, cuttingpoints):

            f = gzip.open(filename, 'rb')
            data = f.readlines()
            f.close()

            times = []
            steps = []

            for i in xrange(len(data)):
                items = data[i].split(',')

                time = int(items[0])
                step = int(float(items[1]))

                times.append(time)
                steps.append(step)



            f = interp1d(times, steps)

            new_array = []

            for entry in cuttingpoints:
            	
            	new_step = f(entry)
            	new_data = [entry,new_step]
            	new_array.append(new_data)
            	new_data = []

            new_array = np.asarray(new_array)

            #print new_array

            start_time = new_array[0][0]
            start_step = new_array[0][1]

            res = []
            newline = [start_time, 0]
            res.append(newline)

            for i in xrange(1,len(new_array)):
                newline = []
                time = new_array[i][0]
                step = new_array[i][1]

                if (time-start_time) > 0:
                    speed = float(step - start_step)/(time-start_time)*1000

                #each line of return list: [time:int, speed:float, label:string]
                    newline.append(time)
                    newline.append(speed)

                    res.append(newline)

                start_time = time
                start_step = step

            return res
