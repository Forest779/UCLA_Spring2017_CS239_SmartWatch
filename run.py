from data_reader import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from sklearn.utils import shuffle

root = "/Users/xiaoyan/Google Drive/CS-239/Data Set/"
namelist = ["Dashing Gu","Zhehan Li", "Xiao Yan"]

test_name = "Test"
# Data Processing
Tran_reader = data_reader()

X_train_list = []
Y_train_list = []
for name in namelist:
    X_tran,Y_tran = Tran_reader.read_array(root+name) # Training set
    X_train_list.append(X_tran)
    Y_train_list.append(Y_tran)

X_train_total = np.concatenate(X_train_list,axis=0)
Y_train_total = np.concatenate(Y_train_list,axis=0)
Y_train_total = np.asarray(Y_train_total).reshape(len(Y_train_total),)
print len(Y_train_total)

Test_reader = data_reader()
X_test,Y_true = Test_reader.read_array(root+test_name)    # Test set

# Model Training
clf = RandomForestClassifier(n_estimators=500, max_depth = 100)

clf.fit(X_train_total,Y_train_total)
y_test = clf.predict(X_test)


print 'reading done'
# Results:
acc = accuracy_score(Y_true, y_test, normalize=True, sample_weight=None)
print "Accuracy: ",acc
#recall = recall_score(Y_true, y_test, average=None)
#print "Recall: Sitting: ", recall[0], "Standing: ", recall[1], "Walking: ", recall[2]
#precision = precision_score(Y_true, y_test, average=None)
#print "Precision: Sitting: ", precision[0], "Standing: ", precision[1], "Walking: ", precision[2]
