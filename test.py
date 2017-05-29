from data_reader import *

reader = data_reader()

root = os.getcwd()

reader.read_array(root)

reader.add_step_data(root)
