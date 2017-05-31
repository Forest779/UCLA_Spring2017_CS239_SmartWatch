from data_reader import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import *

# Data Processing
Tran_reader = data_reader()
X_tran,Y_tran = Tran_reader.read_array('/Users/HaoWu/Documents/Code/CS239/Data Set 2/Dashing Gu') # Training set
Test_reader = data_reader()
X_test,Y_true = Test_reader.read_array('/Users/HaoWu/Documents/Code/CS239/Data Set 2/Zhehan Li')    # Test set

# Model Training
clf = RandomForestClassifier(n_estimators=500, max_depth = 100)
model = OneVsOneClassifier(clf, n_jobs=-1)
model.fit(X_tran,Y_tran)
y_test = model.predict(X_test)

print 'reading done'
# Results:
acc = accuracy_score(Y_true[:,0], y_test[:,0], normalize=True, sample_weight=None)
print "Accuracy: ",acc
recall = recall_score(Y_true[:,0], y_test[:,0], average=None)
print "Recall: Sitting: ", recall[0], "Standing: ", recall[1], "Walking: ", recall[2]
precision = precision_score(Y_true[:,0], y_test[:,0], average=None)
print "Precision: Sitting: ", precision[0], "Standing: ", precision[1], "Walking: ", precision[2]
