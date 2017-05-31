from data_reader import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import *

# Data Processing
Tran_reader = data_reader()
X_tran,Y_tran = Tran_reader.read_array('/Users/xiaoyan/Google Drive/CS-239/Data Set/Zhehan Li') # Training set
Test_reader = data_reader()
X_test,Y_true = Test_reader.read_array('/Users/xiaoyan/Google Drive/CS-239/Data Set/Xiao Yan')    # Test set

# Model Training
clf = RandomForestClassifier(n_estimators=500, max_depth = 100)
model = OneVsRestClassifier(clf, n_jobs=-1)
model.fit(X_tran,Y_tran)
y_test = model.predict(X_test)

print 'reading done'
# Results:
acc = accuracy_score(Y_true, y_test, normalize=True, sample_weight=None)
print "Accuracy: ",acc
#recall = recall_score(Y_true, y_test, average=None)
#print "Recall: Sitting: ", recall[0], "Standing: ", recall[1], "Walking: ", recall[2]
#precision = precision_score(Y_true, y_test, average=None)
#print "Precision: Sitting: ", precision[0], "Standing: ", precision[1], "Walking: ", precision[2]
