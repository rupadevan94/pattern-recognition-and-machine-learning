from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import KFold
import pickle
from utils import *
import os

#path = 'C:/Users/rupa/Documents/Rupa/UCLA/Q1/M276A-PRML/project3_code_and_data/project4_code_and_data'
path = 'C:/Users/hsatreya/Desktop/RupaFolder/project4_code_and_data'
load_dir = path + '/best_params_1_1.pkl'
face_landmarks, train_annotations = load_data(path)

#if not os.path.exists(load_dir):
if 1 == 1:
    params = {}
    mse = {}
    for i in [7, 12]:
        print(i)
        X_train, X_test, y_train, y_test = split_data(i, face_landmarks, train_annotations)
        X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

        #gamma_vals = np.logspace(-15, 15, 15, base=2)
        #cost_vals = np.logspace(-15, 15, 15, base=2)
        #epsilon_vals = np.logspace(-15, 15, 15, base=2)
        if i == 7:
            gamma_vals = np.logspace(-7, -3, 10, base=2)
            cost_vals = np.logspace(-4, 0, 10, base=2)
            epsilon_vals = np.logspace(-4, 0, 10, base=2)
        else:
            gamma_vals = np.logspace(-5, -1, 10, base=2)
            cost_vals = np.logspace(-8, -4, 10, base=2)
            epsilon_vals = np.logspace(-4, 0, 10, base=2)
        #Coarse search
        best_params, best_mse = grid_search(X_train_scaled, y_train, gamma_vals, cost_vals, epsilon_vals)

        score_defn = {'accuracy' : make_scorer(accuracy_score), 'precision' : make_scorer(precision_score) }

        train_acc = []
        test_acc = []
        train_prec = []
        test_prec = []

        #X_train, X_test, y_train, y_test = split_data(4, face_landmarks, train_annotations)
        #X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

        svr = SVR(kernel='rbf', gamma=best_params['gamma'], C=best_params['C'], epsilon=best_params['epsilon'])
        svr.fit(X_train_scaled, y_train)

        y_train_predict = svr.predict(X_train_scaled)
        threshold = find_threshold(y_train)
        y_train_true_labels = find_labels(y_train, threshold)
        y_train_predict_labels = find_labels(y_train_predict, threshold)

        train_acc_score = accuracy_score(y_train_true_labels, y_train_predict_labels)
        train_acc.append(train_acc_score)
        train_prec_score = precision_score(y_train_true_labels, y_train_predict_labels)
        train_prec.append(train_prec_score)
        print("Training accuracy and precision for ")
        # print(len(accuracy))
        print(train_acc_score, train_prec_score)

        print("Testing accuracy and precision for ")
        y_test_predict = svr.predict(X_test_scaled)
        y_test_true_labels = find_labels(y_test, threshold)
        y_test_predict_labels = find_labels(y_test_predict, threshold)

        test_acc_score = accuracy_score(y_test_true_labels, y_test_predict_labels)
        test_prec_score = precision_score(y_test_true_labels, y_test_predict_labels)
        test_acc.append(test_acc_score)
        test_prec.append(test_prec_score)
        print(test_acc_score, test_prec_score)

        params[i] = best_params
        mse[i] = best_mse

