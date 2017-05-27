from data_reader import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


clf=DecisionTreeClassifier()
my_reader=data_reader()
X,Y=my_reader.read_array()



test_data = data_reader()
clf.fit(X,Y)
my_reader.save_data()
