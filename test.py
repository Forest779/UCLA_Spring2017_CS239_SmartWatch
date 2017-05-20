from files import *


test = DataProcessing()

filenames = test.generate_filelist()

test.generate_dict(filenames)
