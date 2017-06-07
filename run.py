from data_reader import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

class data_processing(object):

    def __init__(self):

        self.root = "/Users/xiaoyan/Google Drive/CS-239/Data Set/" #Set up the route to the data
        #self.namelist = ["Dashing Gu","Zhehan Li", "Xiao Yan", "Hao Wu"]  # choose the folder contains the training data
        self.namelist = ["Hao Wu"]

        self.test_name = "Test" # choose the folder contains the test data

        self.X_train_total = None
        self.Y_train_total = None

        self.X_test = None
        self.Y_true = None

        self.Y_pred = None

        self.shuffle = False

        self.no_shuffle_metric = []
        self.shuffle_metric = []


    def set_root(self,root_path):
        self.root = root_path
        print "Root path to data is:"
        print self.root


    def set_test_person(self,test_person):
        self.test_name = test_person

        print "Persons for training data:"
        for person in self.namelist:
            if person != self.test_name:
                print person

        print "Person for test data:"
        print self.test_name

    def generate_training_data(self):

        Tran_reader = data_reader()

        X_train_list = []
        Y_train_list = []
        for name in self.namelist:
            if name != self.test_name:
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


    def model_training(self, feature_report = False):
        print ""
        print "Test person is: " + self.test_name
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

        if feature_report:
            importance = clf.feature_importances_
            print importance

    def randomforest_tuning(self, est_range, mdep_range):
        acc_result = []
        for i in est_range:
            for j in mdep_range:
                clf = RandomForestClassifier(n_estimators=i, max_depth = j)
                clf.fit(self.X_train_total,self.Y_train_total)
                self.Y_pred = clf.predict(self.X_test)
                acc = accuracy_score(self.Y_true, self.Y_pred, normalize=True, sample_weight=None)
                acc_result.append(acc)
                print "n_estimators = ", i, " max_depth = ", j," | Accuracy: ", acc





    def draw_shuffle_figure(self):

        no_shuffle = self.no_shuffle_metric
        shuffle = self.shuffle_metric

    #list: 1)acc, 2)precision, 3)recall, 4)F1 score

        if len(no_shuffle) != 4 or len(shuffle) != 4:
            raise ValueError, "Input is wrong"

        else:
            fig, ax = plt.subplots()
            index = np.arange(len(shuffle))


        # create plot

            bar_width = 0.35
            opacity = 0.8

            rects1 = plt.bar(index, no_shuffle, bar_width,
                            alpha=opacity,
                            color='b',
                            label='without shuffle')

            rects2 = plt.bar(index + bar_width, shuffle, bar_width,
                            alpha=opacity,
                            color='g',
                            label='shuffle')

            plt.xlabel('Metrics')
            plt.ylabel('Scores')
            plt.title('Metrics without shuffle vs shuffle')
            plt.xticks(index + bar_width, ('Acc', 'Prec', 'Rec', 'F1'))
            plt.legend()

            plt.tight_layout()
            plt.savefig("shuffle.png")
            plt.show()


if __name__ == '__main__':

    #generate file once

    model = data_processing()
    # model.set_root("/path/to/Data Set/")
    model.set_root("/Users/lizhehan/UCLA/CS-239/Data Set/")
    # model.set_test_person("Xiao Yan") #choose the person to be tested
    # model.generate_training_data()
    # model.generate_test_data()
    # model.model_training()

    #save and load multiple times
    # model.save_training_data()
    # model.save_test_data()

    model.load_training_data()
    model.load_test_data()
    model.model_training(feature_report = True)
    # model.randomforest_tuning(xrange(500,2001,500), xrange(20,101,20))

    #shuffle
    model.shuffle_data()
    model.model_training()
    #model.draw_shuffle_figure()

    #reset to original state (if the data is saved)
    #model.reset_to_noshuffle()
    #model.model_training()
