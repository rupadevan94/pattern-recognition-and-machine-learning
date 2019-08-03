import itertools
import numpy as np
from scipy import stats
# import pylab as pl
# from sklearn import svm, linear_model, cross_validation
import pickle
import os
from skimage.io import imread
from scipy.io import loadmat
from skimage.feature import hog
from sklearn.preprocessing import MinMaxScaler
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.metrics import accuracy_score, precision_score
import pdb
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC

# For senators
# path = 'C:/Users/rupa/Documents/Rupa/UCLA/Q1/M276A-PRML/project3_code_and_data/project4_code_and_data'
path = 'C:/Users/hsatreya/Desktop/RupaFolder/project4_code_and_data'
sen_path = '/img-elec/senator'
gov_path = '/img-elec/governor'
elec_type = 'Sen'


def scale_features_concat(X_train, X_test):
    '''
    scaler = MinMaxScaler()
    scaler.fit(X_train[:, :250000])
    hog_train_scaled = scaler.transform(X_train[:, :250000])
    hog_train_scaled[hog_train_scaled > 1] = 1
    hog_test_scaled = scaler.transform(X_test[:, :250000])

    scaler = MinMaxScaler()
    scaler.fit(X_train[:, 250000:])
    lm_train_scaled = scaler.transform(X_train[:, 250000:])
    lm_train_scaled[lm_train_scaled > 1] = 1
    lm_test_scaled = scaler.transform(X_test[:, 250000:])

    X_train_concat = np.concatenate((hog_train_scaled, lm_train_scaled), axis=1)
    X_test_concat = np.concatenate((hog_test_scaled, lm_test_scaled), axis=1)
    #pdb.set_trace()
    print("shape of train " + str(np.shape(X_train_concat)))
    print("shape of test " + str(np.shape(X_test_concat)))
    return X_train_concat, X_test_concat
    '''
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled[X_train_scaled > 1] = 1.0
    return X_train_scaled, X_test_scaled


def grid_search_concat(X_train_scaled, y_train, cost_vals):
    scorer = make_scorer(mean_squared_error, greater_is_better=False)

    start = time.clock()
    svc = LinearSVC(fit_intercept=False)
    parameters = {'C': cost_vals}
    print(parameters)
    print("Param calculation started***")
    clf = GridSearchCV(svc, parameters, cv=10, n_jobs=-1, scoring=scorer, verbose=5)
    clf.fit(X_train_scaled, y_train)
    print("Calculated**")
    print(clf.best_params_)
    print(abs(clf.best_score_))
    end = time.clock()
    print("Time taken : " + str(end - start))
    return clf.best_params_, clf.best_score_


if elec_type == 'Sen':
    data_dir = path + '/sen_concat_features_1_2.pkl'
    if not os.path.exists(data_dir):
        # if 1 == 1:
        # Read images and perform HoG
        files = os.listdir(path + sen_path)
        fimages = [os.path.join(path + sen_path, f) for f in files if f[-3:] == 'jpg']
        h = 500
        w = 500
        sen_hog_images = np.zeros((len(fimages), h, w))
        # fd = np.zeros

        for i, fi in enumerate(fimages):
            image = imread(fi)
            fd, sen_hog_images[i] = hog(image, orientations=8, pixels_per_cell=(16, 16),
                                        cells_per_block=(1, 1), visualize=True, multichannel=True)
            pdb.set_trace()

        print(sen_hog_images.shape)

        # Read landmarks
        sen_data = loadmat(path + '/stat-sen.mat')
        sen_face_landmarks = sen_data['face_landmark']
        sen_vote_diff = sen_data['vote_diff']
        print(np.shape(sen_face_landmarks))
        sen_hog_images = np.reshape(sen_hog_images, (116, 500 * 500))
        sen_concat_features = np.concatenate((sen_hog_images, sen_face_landmarks), axis=1)
        print(np.shape(sen_concat_features))
        pickle.dump(sen_concat_features, open(path + '/sen_concat_features_1_2.pkl', 'wb'))
        pickle.dump(sen_vote_diff, open(path + '/sen_vote_diff_1_2.pkl', 'wb'))

    else:
        sen_concat_features = pickle.load(open(path + '/sen_concat_features_1_2.pkl', 'rb'))
        sen_voting_diff = pickle.load(open(path + '/sen_vote_diff_1_2.pkl', 'rb'))
        print(np.shape(sen_concat_features))
        X_train, X_test, y_train, y_test = train_test_split(sen_concat_features, sen_voting_diff, test_size=0.2,
                                                            random_state=2)
        print("Printing here ")
        print(np.shape(X_train), np.shape(X_test))
        X_train_scaled, X_test_scaled = scale_features_concat(X_train, X_test)

        # print(np.shape(X_train_scaled), np.shape(X_test_scaled))
        # calculating features for rank svm
        Xp_train = []
        diff_train = []
        yp_train = []
        for i in range(0, 40, 2):
            # print(np.shape(X_train_scaled[i + 1] - X_train_scaled[i]))
            Xp_train.append(X_train_scaled[i + 1] - X_train_scaled[i])
            diff_train.append(abs(y_train[i]))
            yp_train.append(np.sign(y_train[i + 1] - y_train[i]))

        for i in range(40, 92, 2):
            print(i)
            # print(np.shape(X_train_scaled[i + 1] - X_train_scaled[i]))
            Xp_train.append(X_train_scaled[i] - X_train_scaled[i + 1])
            diff_train.append(abs(y_train[i]))
            yp_train.append(np.sign(y_train[i] - y_train[i + 1]))

        Xp_test = []
        diff_test = []
        yp_test = []
        for i in range(0, 12, 2):
            Xp_test.append(X_test_scaled[i + 1] - X_test_scaled[i])
            diff_test.append(abs(y_test[i]))
            yp_test.append(np.sign(y_test[i + 1] - y_test[i]))

        for i in range(12, 24, 2):
            Xp_test.append(X_test_scaled[i] - X_test_scaled[i + 1])
            diff_test.append(abs(y_test[i]))
            yp_test.append(np.sign(y_test[i] - y_test[i + 1]))

        print("Rank features ")
        print(np.shape(np.array(Xp_train)), np.shape(np.array(yp_train)))
        print(np.shape(Xp_test), np.shape(yp_test))
        print("Test output")
        print(yp_test)
        pdb.set_trace()
        print("Coarse search of parameters")
        cost_vals = [2 ** x for x in range(-15, 16, 5)]
        best_params, best_mse = grid_search_concat(Xp_train, np.asarray(yp_train).ravel(), cost_vals)
        best_C = np.log2(best_params['C'])

        print("Fine search of parameters")
        cost_vals = [2 ** x for x in range(int(best_C) - 5, int(best_C) + 5)]
        best_params, best_mse = grid_search_concat(Xp_train, np.asarray(yp_train).ravel(), cost_vals)
        best_C = best_params['C']

        print("testing")
        clf = LinearSVC(C=best_C, fit_intercept=False)
        clf.fit(Xp_train, np.asarray(yp_train).ravel())
        yp_test_predict = clf.predict(Xp_test)
        print(yp_test_predict)

        print("Testing Accuracy is ")
        print(accuracy_score(yp_test, yp_test_predict))
        print("Testing precision is ")
        print(precision_score(yp_test, yp_test_predict))

else:
    data_dir = path + '/gov_concat_features_1_2.pkl'
    if not os.path.exists(data_dir):
        # if 1 == 1:
        print("Reading data")

        files = os.listdir(path + gov_path)
        fimages = [os.path.join(path + gov_path, f) for f in files if f[-3:] == 'jpg']
        h = 500
        w = 500
        gov_hog_images = np.zeros((len(fimages), h, w))

        for i, fi in enumerate(fimages):
            image = imread(fi)
            fd, gov_hog_images[i] = hog(image, orientations=8, pixels_per_cell=(16, 16),
                                        cells_per_block=(1, 1), visualize=True, multichannel=True)

        print(gov_hog_images.shape)

        # Read landmarks
        gov_data = loadmat(path + '/stat-gov.mat')
        gov_face_landmarks = gov_data['face_landmark']
        gov_vote_diff = gov_data['vote_diff']
        print(np.shape(gov_face_landmarks))
        gov_hog_images = np.reshape(gov_hog_images, (112, 500 * 500))
        gov_concat_features = np.concatenate((gov_hog_images, gov_face_landmarks), axis=1)
        print(np.shape(gov_concat_features))
        pickle.dump(gov_concat_features, open(path + '/gov_concat_features_1_2.pkl', 'wb'))
        pickle.dump(gov_vote_diff, open(path + '/gov_vote_diff_1_2.pkl', 'wb'))

    else:
        gov_concat_features = pickle.load(open(path + '/gov_concat_features_1_2.pkl', 'rb'))
        gov_voting_diff = pickle.load(open(path + '/gov_vote_diff_1_2.pkl', 'rb'))
        print(np.shape(gov_concat_features))

        Xp = []
        diff = []
        yp = []
        for i in range(0, 56, 2):
            # print(np.shape(X_train_scaled[i + 1] - X_train_scaled[i]))
            Xp.append(gov_concat_features[i + 1] - gov_concat_features[i])
            diff.append(abs(gov_voting_diff[i]))
            yp.append(np.sign(gov_voting_diff[i + 1] - gov_voting_diff[i]))

        for i in range(56, 112, 2):
            print(i)
            # print(np.shape(X_train_scaled[i + 1] - X_train_scaled[i]))
            Xp.append(gov_concat_features[i] - gov_concat_features[i + 1])
            diff.append(abs(gov_voting_diff[i]))
            yp.append(np.sign(gov_voting_diff[i] - gov_voting_diff[i + 1]))

        Xp = np.array(Xp)
        yp = np.array(yp)
        Xp_train, Xp_test, yp_train, yp_test = train_test_split(Xp, yp, test_size=0.2, random_state=2)

        # X_train, X_test, y_train, y_test = train_test_split(gov_concat_features, gov_voting_diff, test_size=0.19, random_state=10)
        print("Printing here ")
        print(np.shape(Xp_train), np.shape(Xp_test))
        Xp_train_scaled, Xp_test_scaled = scale_features_concat(Xp_train, Xp_test)
        '''
        #print(np.shape(X_train_scaled), np.shape(X_test_scaled))
        #calculating features for rank svm
        Xp_train = []
        diff_train = []
        yp_train = []
        for i in range(0, 45, 2):
            #print(np.shape(X_train_scaled[i + 1] - X_train_scaled[i]))
            Xp_train.append(X_train_scaled[i + 1] - X_train_scaled[i])
            diff_train.append(abs(y_train[i]))
            yp_train.append(np.sign(y_train[i + 1] - y_train[i]))

        for i in range(45, 89, 2):
            print(i)
            #print(np.shape(X_train_scaled[i + 1] - X_train_scaled[i]))
            Xp_train.append(X_train_scaled[i] - X_train_scaled[i + 1])
            diff_train.append(abs(y_train[i]))
            yp_train.append(np.sign(y_train[i] - y_train[i + 1]))

        Xp_test = []
        diff_test = []
        yp_test = []
        for i in range(0, 11, 2):
            Xp_test.append(X_test_scaled[i + 1] - X_test_scaled[i])
            diff_test.append(abs(y_test[i]))
            yp_test.append(np.sign(y_test[i + 1] - y_test[i]))

        for i in range(11, 21, 2):
            Xp_test.append(X_test_scaled[i] - X_test_scaled[i + 1])
            diff_test.append(abs(y_test[i]))
            yp_test.append(np.sign(y_test[i] - y_test[i + 1]))
        '''
        print("Rank features ")
        print(np.shape(np.array(Xp_train_scaled)), np.shape(np.array(yp_train)))
        print(np.shape(Xp_test_scaled), np.shape(yp_test))
        print("Test output")
        print(yp_test)

        # pdb.set_trace()
        print("Coarse search of parameters")
        cost_vals = [2 ** x for x in range(-15, 16, 3)]
        best_params, best_mse = grid_search_concat(Xp_train_scaled, np.asarray(yp_train).ravel(), cost_vals)
        best_C = np.log2(best_params['C'])

        print("Fine search of parameters")
        cost_vals = [2 ** x for x in range(int(best_C) - 5, int(best_C) + 5)]
        best_params, best_mse = grid_search_concat(Xp_train_scaled, np.asarray(yp_train).ravel(), cost_vals)
        best_C = best_params['C']

        # best_C = 0.0009765625
        # Xp_train = np.array(Xp_train)
        # Xp_test = np.array(Xp_test)
        # yp_train = np.array(yp_train)
        # yp_test = np.array(yp_test)
        # Train accuracy
        print("Calculating train accuracy")
        kf = KFold(n_splits=10)
        accuracy = []
        precision = []
        for train_ind, val_ind in kf.split(Xp_train_scaled):
            print(train_ind, val_ind)
            X_val_train, X_val = Xp_train_scaled[train_ind], Xp_train_scaled[val_ind]
            y_val_train, y_val = yp_train[train_ind].ravel(), yp_train[val_ind].ravel()
            svc = LinearSVC(C=best_C, fit_intercept=False)
            svc.fit(X_val_train, y_val_train)
            y_val_predict = svc.predict(X_val)
            # y_val_labels = np.sign(y_val)
            # y_val_predict_labels = np.sign(y_val_predict)
            accuracy.append(accuracy_score(y_val, y_val_predict))
            precision.append(precision_score(y_val, y_val_predict))
        print("Training accuracy and precision ")
        print(len(accuracy))
        print(np.mean(np.array(accuracy)), np.mean(np.array(precision)))

        # Testing accuracy
        print("testing")
        clf = LinearSVC(C=best_C, fit_intercept=False)
        clf.fit(Xp_train_scaled, np.asarray(yp_train).ravel())
        yp_test_predict = clf.predict(Xp_test_scaled)
        print(yp_test_predict)

        print("Testing Accuracy is ")
        print(accuracy_score(yp_test, yp_test_predict))
        print("Testing precision is ")
        print(precision_score(yp_test, yp_test_predict))
