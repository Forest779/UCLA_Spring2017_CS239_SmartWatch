import gzip
import numpy as np
import os
import csv

class data_reader():
	def __init__(self):
		self.feature_final=None
		self.label_Voted=None

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

	def __multiple_dir_multiFileArray(self):
		feature_array_list=[]
		label_array_list=[]
		for root, dirs, files in os.walk(os.getcwd()):
			if 'data' in root:
				start_end_tup=self.__get_start_end(root)
				cuttingpoints=self.__generate_cutting_point(start_end_tup)
				feature_label=self.__same_dir_mutipleFileToArray(root,cuttingpoints)
				feature_array_list.append(feature_label[0])
				label_array_list.append(feature_label[1])
		self.feature_final=np.concatenate(feature_array_list,axis=0)
		label_toVote=np.concatenate(label_array_list,axis=0)
		self.label_Voted=self.__major_vote(label_toVote)
		return (self.feature_final,self.label_Voted)
		#return np.concatenate([feature_final,label_Voted],axis=1)

	def read_array(self):
		return self.__multiple_dir_multiFileArray()

	def save_data(self):
		np.savetxt('feature.mydata',self.feature_final,delimiter=',')
		np.savetxt('label.mydata',self.label_Voted, fmt='%s',delimiter=',')


		
		