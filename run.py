from data_reader import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from sklearn.utils import shuffle


class data_processing(object):

    def __init__(self):

        self.root = "/Users/xiaoyan/Google Drive/CS-239/Data Set/" #Set up the route to the data
        self.namelist = ["Dashing Gu","Zhehan Li", "Xiao Yan"]  # choose the folder contains the training data
        self.test_name = "Test" # choose the folder contains the test data

        self.X_train_total = None
        self.Y_train_total = None

        self.X_test = None
        self.Y_true = None

        self.Y_pred = None

        self.shuffle = False


    def generate_training_data(self):

        Tran_reader = data_reader()

        X_train_list = []
        Y_train_list = []
        for name in self.namelist:
            X_tran,Y_tran = Tran_reader.read_array(self.root+name) # Training set
            X_train_list.append(X_tran)
            Y_train_list.append(Y_tran)

        self.X_train_total = np.concatenate(X_train_list,axis=0)
        self.Y_train_total = np.concatenate(Y_train_list,axis=0)
        self.Y_train_total = np.asarray(self.Y_train_total).reshape(len(self.Y_train_total),)
        print "Total number of training data:"
        print len(self.Y_train_total)


    def generate_test_data(self):

        Test_reader = data_reader()
        self.X_test,self.Y_true = Test_reader.read_array(self.root+self.test_name)    # Test set
        print "Total number of test data:"
        print len(self.Y_true)


    def reset_to_noshuffle(self):
        self.shuffle = False
        self.load_test_data()
        self.load_training_data()
        print "Set back to original state"

    def shuffle_data(self):
        try:
            self.X_train_total, self.Y_train_total = shuffle(self.X_train_total, self.Y_train_total,random_state =0)

            self.shuffle = True

            print "Data shuffled"
        except:
            raise ValueError, "Training data needs to be generated"

    def save_training_data(self):

        try:
            filename = "feature"
            if self.shuffle:
                filename += "_shuffle"

            np.savetxt(filename+'.mydata',self.X_train_total,delimiter=',')

            filename2 = "label"
            if self.shuffle:
                filename2 += "_shuffle"
	    np.savetxt(filename2+'.mydata',self.Y_train_total, fmt='%s',delimiter=',')

	    print "Saved: {},{}".format(filename,filename2)

	except:
	    raise ValueError,"Training data needs to be generated"


    def save_test_data(self):
        try:

            filename = "testfeature"
            if self.shuffle:
                filename += "_shuffle"

            np.savetxt(filename+'.mydata',self.X_test,delimiter=',')

            filename2 = "testlabel"
            if self.shuffle:
                filename2 += "_shuffle"
	    np.savetxt(filename2+'.mydata',self.Y_true, fmt='%s',delimiter=',')

	    print "Saved: {},{}".format(filename,filename2)

	except:
	    raise ValueError,"Test data needs to be generated"


    def load_training_data(self):
        filename = "feature"
        if self.shuffle:
            filename += "_shuffle"

        filename2 = "label"
        if self.shuffle:
            filename2 += "_shuffle"
        try:
            self.X_train_total=np.loadtxt(filename+ '.mydata',delimiter=',')

            self.Y_train_total=np.loadtxt(filename2+'.mydata',dtype='str',delimiter=',')

            print "Loaded: {},{}".format(filename,filename2)

        except:
            raise IOError,"Error loading training data"


    def load_test_data(self):

        filename = "testfeature"
        if self.shuffle:
            filename += "_shuffle"

        filename2 = "testlabel"
        if self.shuffle:
            filename2 += "_shuffle"
        try:
            self.X_test=np.loadtxt(filename+ '.mydata',delimiter=',')

            self.Y_true=np.loadtxt(filename2+'.mydata',dtype='str',delimiter=',')

            print "Loaded: {},{}".format(filename,filename2)

        except:
            raise IOError,"Error loading test data"


    def model_training(self):
        print ""
        # Model Training
        clf = RandomForestClassifier(n_estimators=500, max_depth = 100)

        clf.fit(self.X_train_total,self.Y_train_total)
        self.Y_pred = clf.predict(self.X_test)

        # Results:
        acc = accuracy_score(self.Y_true, self.Y_pred, normalize=True, sample_weight=None)

        if self.shuffle:
            print "Result after shuffle:"
        else:
            print "Result without shuffle:"

        print "Accuracy: ",acc

        print "Confusion matrix:"
        print confusion_matrix(self.Y_true, self.Y_pred)

        print "Report:"
        print classification_report(self.Y_true, self.Y_pred)


if __name__ == '__main__':

    #generate file once
    model = data_processing()
    model.generate_training_data()
    model.generate_test_data()
    model.model_training()

    #save and load multiple times

    model.save_training_data()
    model.save_test_data()
    #model.load_training_data()
    #model.load_test_data()
    #model.model_training()

    #shuffle

    #model.shuffle_data()
    #model.model_training()


    #reset to original state (if the data is saved)
    #model.reset_to_noshuffle()
    #model.model_training()

#recall = recall_score(Y_true, y_test, average=None)
#print "Recall: Sitting: ", recall[0], "Standing: ", recall[1], "Walking: ", recall[2]
#precision = precision_score(Y_true, y_test, average=None)
#print "Precision: Sitting: ", precision[0], "Standing: ", precision[1], "Walking: ", precision[2]
