from add_method import *

test = data_reader()
test.read_array()

filepath = "Walking/14492D46823E60C_Fri_May_12_18-21_2017_PDT/data/19_android.sensor.step_counter.data.csv.gz"
cuttingpoints = test.cuttingpoint

#print cuttingpoints
res = test.handle_step(filepath,cuttingpoints)

print res
