from skimage.feature import  hog
from skimage.io import imread
from utils import *
import os
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import pdb
from scipy.io import loadmat

#path = 'C:/Users/rupa/Documents/Rupa/UCLA/Q1/M276A-PRML/project3_code_and_data/project4_code_and_data'
path = 'C:/Users/hsatreya/Desktop/RupaFolder/project4_code_and_data'
load_dir = path + '/concat_features_1_2.pkl'
print(load_dir)

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

def grid_search_concat(X_train_scaled, y_train, gamma_vals, cost_vals, epsilon_vals):
    scorer = make_scorer(mean_squared_error, greater_is_better= False)

    start = time.clock()
    svr = SVR(kernel= 'rbf')
    parameters = {'gamma' : gamma_vals, 'C' : cost_vals, 'epsilon' : epsilon_vals}
    print(parameters)
    print("Param calculation started***")
    clf = GridSearchCV(svr, parameters, cv = 10, n_jobs=-1, scoring=scorer, verbose = 1)
    clf.fit(X_train_scaled, y_train)
    print("Calculated**")
    print(clf.best_params_)
    print(abs(clf.best_score_))
    end = time.clock()
    print("Time taken : " + str(end - start))
    return clf.best_params_, clf.best_score_

if not os.path.exists(load_dir):
    print("Dir not there yet")
    face_landmarks, train_annotations = load_data(path)
    h = 500
    w = 500
    files = os.listdir(path + '/img')
    fimages = [os.path.join(path + '/img', f) for f in files if f[-3:] == 'jpg']
    #hog_images = np.zeros((len(fimages), h, w))
    fd = np.zeros((len(fimages), 7688))

    for i, fi in enumerate(fimages):
        image = imread(fi)
        fd[i], hog_images = hog(image, orientations=8, pixels_per_cell=(16, 16),
                            cells_per_block=(1, 1), visualize=True, multichannel=True)
        #pdb.set_trace()

    print(fd.shape)
    print(np.shape(face_landmarks))
    #hog_images = np.reshape(hog_images, (491, 500*500))
    concat_features = np.concatenate((fd, face_landmarks), axis = 1)
    print(np.shape(concat_features))
    pickle.dump(concat_features, open(path + '/concat_features_1_2.pkl', 'wb'))
    pickle.dump(train_annotations, open(path + '/y_1_2.pkl', 'wb'))

else:
    i = 0
    concat_features = pickle.load(open(path + '/concat_features_1_2.pkl', 'rb'))
    #concat_features = loadmat(path + '/hogfeatures.mat')
    #concat_features = concat_features['hogfeats']
    #concat_features = np.reshape(concat_features, ((491, 6272)))
    train_annotations = pickle.load(open(path + '/y_1_2.pkl', 'rb'))
    print("Loaded valuse**")
    #X_train, X_test, y_train, y_test = split_data(i, concat_features, train_annotations)
    #X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    #path = 'C:/Users/rupa/Documents/Rupa/UCLA/Q1/M276A-PRML/project3_code_and_data/project4_code_and_data'
    load_dir = path + '/best_params_1_2.pkl'
    face_landmarks, train_annotations = load_data(path)
    if not os.path.exists(load_dir):
        params = {}
        mse = {}
        for i in range(14):
            print("Iteration " + str(i))
            X_train, X_test, y_train, y_test = split_data(i, concat_features, train_annotations)
            X_train_scaled, X_test_scaled = scale_features_concat(X_train, X_test)
            #pdb.set_trace()
            print(np.sum(np.where(X_train_scaled > 1)))
            print("Computed Scaled values")
            '''
            gamma_vals = [2 ** x for x in range(-19, -6, 1)]
            cost_vals = [2 ** x for x in range(-3, 9, 1)]
            epsilon_vals = [2 ** x for x in range(-12, 0, 1)]
            '''
            gamma_vals = np.logspace(-19, -5, 15, base=2)
            cost_vals = np.logspace(4, 13, 10, base=2)
            epsilon_vals = np.logspace(-9, 1, 11, base=2)
            #Coarse search

            print("Coarse grid search")
            best_params, best_mse = grid_search_concat(X_train_scaled, y_train, gamma_vals, cost_vals, epsilon_vals)

            print("Done")
            '''
            best_C = np.log2(best_params['C'])
            best_epsilon = np.log2(best_params['epsilon'])
            best_gamma = np.log2(best_params['gamma'])
            print("Fine grid search")
            gamma_vals = [2 ** x for x in range(int(best_gamma) - 2, int(best_gamma) + 2)]
            cost_vals = [2 ** x for x in range(int(best_C) - 2, int(best_C) + 2)]
            epsilon_vals = [2 ** x for x in range(int(best_epsilon) - 2, int(best_epsilon) + 2)]
            # Fine search
            best_params, best_mse = grid_search_concat(X_train_scaled, y_train, gamma_vals, cost_vals, epsilon_vals)
            '''
            params[i] = best_params
            mse[i] = best_mse
        pickle.dump(params, open(path + '/best_params_1_2.pkl', 'wb'))
        pickle.dump(mse, open(path + '/best_mse.pkl_1_2', 'wb'))
        print("Completed**")
    else:
        params = pickle.load(open(load_dir, 'rb'))
        print("Loaded params")
        print(params)
        score_defn = {'accuracy' : make_scorer(accuracy_score), 'precision' : make_scorer(precision_score) }
        #{'accuracy' = make_scorer(accuracy_score), 'precision' = make_scorres(precision_score)}
        #Find train accuracies

        train_acc = []
        test_acc = []
        train_prec = []
        test_prec = []

        for i in range(14):
            X_train, X_test, y_train, y_test = split_data(i, concat_features, train_annotations)
            X_train_scaled, X_test_scaled = scale_features_concat(X_train, X_test)

            svr = SVR(kernel='rbf', gamma=params[i]['gamma'], C=params[i]['C'], epsilon=params[i]['epsilon'])
            svr.fit(X_train_scaled, y_train)

            y_train_predict = svr.predict(X_train_scaled)
            threshold = find_threshold(y_train)
            y_train_true_labels = find_labels(y_train, threshold)
            y_train_predict_labels = find_labels(y_train_predict, threshold)

            train_acc_score = accuracy_score(y_train_true_labels, y_train_predict_labels)
            train_prec_score = precision_score(y_train_true_labels, y_train_predict_labels)
            train_acc.append(train_acc_score)
            train_prec.append(train_prec_score)
            print("Training accuracy and precision for ")
            print(i)
            # print(len(accuracy))
            print(train_acc_score, train_prec_score)

            print("Testing accuracy and precision for ")
            print(i)
            y_test_predict = svr.predict(X_test_scaled)
            y_test_true_labels = find_labels(y_test, threshold)
            y_test_predict_labels = find_labels(y_test_predict, threshold)

            test_acc_score = accuracy_score(y_test_true_labels, y_test_predict_labels)
            test_prec_score = precision_score(y_test_true_labels, y_test_predict_labels)
            test_acc.append(test_acc_score)
            test_prec.append(test_prec_score)
            # print("Training accuracy and precision for ")
            # print(i)
            # print(len(accuracy))
            print(test_acc_score, test_prec_score)

        pickle.dump(train_acc, open(path + '/train_acc_1_2.pkl', 'wb'))
        pickle.dump(test_acc, open(path + '/test_acc_1_2.pkl', 'wb'))
        pickle.dump(train_prec, open(path + '/train_prec_1_2.pkl', 'wb'))
        pickle.dump(test_prec, open(path + '/test_prec_1_2.pkl', 'wb'))