from data_reader import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from operator import itemgetter

class data_processing(object):

    def __init__(self):

        self.root = "/Users/xiaoyan/Google Drive/CS-239/Data Set/" #Set up the route to the data
        self.namelist = ["Dashing Gu","Zhehan Li", "Xiao Yan", "Hao Wu"]  # choose the folder contains the training data
        #self.namelist = ["Zhehan Li"]

        self.test_name = "Test" # choose the folder contains the test data

        self.X_train_total = None
        self.Y_train_total = None

        self.X_test = None
        self.Y_true = None

        self.Y_pred = None

        self.shuffle = False

        self.no_shuffle_metric = []
        self.shuffle_metric = []

        self.feature_list = []
        self.importance_list = []
        self.feature_select = False

    def set_root(self,root_path):
        if root_path[-1] != "/":
            root_path += "/"
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
            self.importance_list = importance
            for i in xrange(len(self.feature_list)):
                self.feature_list[i].append(self.importance_list[i])

        data = precision_recall_fscore_support(self.Y_true, self.Y_pred, average='macro')

        if self.shuffle:
            self.shuffle_metric.append(acc)
            for i in range(3):
                self.shuffle_metric.append(data[i])
        else:
            self.no_shuffle_metric.append(acc)
            for i in range(3):
                self.no_shuffle_metric.append(data[i])



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
            print "Need more input"
            return

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
            plt.ylim((0.7,1.0))
            plt.title('Metrics without shuffle vs shuffle')
            plt.xticks(index + bar_width, ('Acc', 'Prec', 'Rec', 'F1'))
            plt.legend()

            plt.tight_layout()
            plt.savefig("shuffle.png")
            plt.show()

    def generate_feature_list(self):
        feature_name = ["Max", "Min", "Avg", "Std"]
        feature_axis = ["_X_", "_Y_", "_Z_"]
        feature_axis_plus = ["_X_", "_Y_", "_Z_", "_SMV_"]
        linear_acc = ["Power_0.3-15Hz", "Power_0.6-2.5Hz", "Power_Domain", "Power_Ratio(Domain/Total)"]
        feature_code = 0
        feature_list = []
        for root, dirs, files in os.walk(self.root):
            if 'data' in root:
                for filename in files:
                    if filename[-3:]=='.gz' and 'sensor' in filename and 'significant_motion' not in filename and 'step' not in filename and 'light' not in filename:
                        filename = filename[:-12]
                        filename = filename[filename.rfind('.')+1:]
                        for fn in feature_name:
                            for fa in feature_axis:
                                feature_list.append([feature_code, filename+fa+fn])
                                feature_code+=1
                        if "linear_acceleration" in filename:
                            # __fft
        				    for fn in linear_acc:
        				        for fa in feature_axis_plus:
        				            feature_list.append([feature_code, filename+fa+fn])
        				            feature_code+=1
        				    #__range_linear_acc
        				    for fa in feature_axis:
        				        feature_list.append([feature_code, filename+fa+".8-.2_Range"])
        				        feature_code+=1
        				    #__integral_linear_acc
        				    for fa in feature_axis:
        				        feature_list.append([feature_code, filename+fa+"Integral"])
        				        feature_code+=1
                        if "gravity" in filename:
        				    #__range_gravity
        				    for fa in feature_axis:
        				        feature_list.append([feature_code, filename+fa+".8-.2_Range"])
        				        feature_code+=1
        				    #__Kurtosis_gravity
        				    for fa in feature_axis:
        				        feature_list.append([feature_code, filename+fa+"Kurtosis"])
        				        feature_code+=1
                break
        feature_list.append([feature_code, "step_counter_Speed_Avg"])
        # print feature_list
        self.feature_list = feature_list

    def generate_imp_csv(self, name):
        with open(name, 'wb') as f:
            for item in self.feature_list:
                line = str(item[0]) + ',' + item[1] + ',' + str(item[2]) + '\n'
                f.write(line.encode('utf-8'))

    def feature_selection(self, num = 10):
        self.feature_select = False
        self.feature_list.sort(key = itemgetter(2), reverse = True)
        print self.feature_list
        print self.X_train_total.shape
        print self.X_test.shape
        X_train_total = self.X_train_total
        X_test = self.X_test
        feature_list = self.feature_list
        self.X_train_total = X_train_total[:,0:1]
        self.X_test = X_test[:,0:1]
        self.feature_list = []
        for i in xrange(num):
            col = feature_list[i][0]
            self.X_train_total = np.hstack((self.X_train_total, X_train_total[:,col:col+1]))
            self.X_test = np.hstack((self.X_test, X_test[:,col:col+1]))
            self.feature_list.append([col,feature_list[i][1]])
        self.X_train_total = self.X_train_total[:,1:]
        self.X_test = self.X_test[:,1:]
        print self.feature_list
        print self.X_train_total.shape
        print self.X_test.shape

if __name__ == '__main__':

    #generate file once

    model = data_processing()
    model.set_root("/Users/lizhehan/UCLA/CS-239/Data Set/")
    model.generate_feature_list()
    # model.set_test_person("Zhehan Li") #choose the person to be tested
    model.generate_training_data()
    model.generate_test_data()
    #
    model.save_test_data()
    model.save_training_data()
    #
    # model.draw_shuffle_figure()

    #save and load multiple times
    # model.save_training_data()
    # model.save_test_data()

    #model.save_training_data()
    #model.save_test_data()
    # model.load_training_data()
    # model.load_test_data()
    model.model_training(feature_report = True)
    model.generate_imp_csv('imp_origin.csv')

    #feature select
    model.feature_selection()
    model.model_training(feature_report = True)
    model.generate_imp_csv('imp_seletion.csv')

    #shuffle
    # model.shuffle_data()
    # model.model_training()
    # model.draw_shuffle_figure()

    #reset to original state (if the data is saved)
    #model.reset_to_noshuffle()
    #model.model_training()
