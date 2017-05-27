from data_reader import *
from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier()
my_reader=data_reader()
X,Y=my_reader.read_array('/Users/HaoWu/Documents/Code/CS239/Data Set')
clf.fit(X,Y)
my_reader.save_data('test')
