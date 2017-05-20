import gzip
import os

class DataProcessing:

    def __init__(self):
        self.dict = {}

    def generate_filelist(self):
        res = []
        for root,dirs,files in os.walk(os.getcwd()):
            for filename in files:
                if "65539" not in filename and "gz" in filename:
                    res.append(filename)
        return res

    #look at the list[string] files of each 4s window and decide whether size > 80% of certain threshold
    def isRight(self,entry_list, threshold):

        return len(entry_list) > 0.8 * threshold

    #look at the list[string] files and decide whether the label is pure
    def isPure(self,entry_list):

        start_label = entry_list[0].split(',')[-1].strip('\n')

        for line in entry_list:
            label = line.split(',')[-1].strip('\n')
            if label != start_label:
                return False
        return True

    #go through the test files and generate the data number threshold for each sensor
    def generate_dict(self,filenames):
        for filename in filenames:
            f = gzip.open(filename, 'rb')
            data = f.readlines()
            f.close()

            start = int(data[0].split(',')[0])
            end = int(data[-1].split(',')[0])

            num_4s = (end - start)/4000

            num_entry = len(data)

            res = num_entry/num_4s

            self.dict[filename] = res

    #transform steps to speed, itype: name of speed sensor data, rtype: list[list[]]
    def step_speed(self, filename):
        f = gzip.open(filename, 'rb')
        data = f.readlines()
        f.close()

        res = []


        start_time = int(data[0].split(',')[0])
        start_step = int(float(data[0].split(',')[1]))

        for i in xrange(1,len(data)):
            newline = []
            items = data[i].split(',')

            time = int(items[0])
            step = int(float(items[1]))
            label = items[-1].strip('\n')

            if (time-start_time) > 0:
                speed = float(step - start_step)/(time-start_time)*1000

            #each line of return list: [time:int, speed:float, label:string]
                newline.append(time)
                newline.append(speed)
                newline.append(label)

                res.append(newline)

            start_time = time
            start_step = step


        return res
