from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import KFold
import pickle
from utils import *
import os

path = 'C:/Users/hsatreya/Desktop/RupaFolder/project4_code_and_data'
load_dir = path + '/best_params_1_1.pkl'
face_landmarks, train_annotations = load_data(path)
if not os.path.exists(load_dir):
    params = {}
    mse = {}
    for i in range(14):
        X_train, X_test, y_train, y_test = split_data(i, face_landmarks, train_annotations)
        X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

        gamma_vals = np.logspace(-15, 15, 31, base=2)
        cost_vals = np.logspace(-15, 15, 31, base=2)
        epsilon_vals = np.logspace(-15, 15, 31, base=2)

        '''
        gamma_vals = [2 ** x for x in range(-15, 16, 5)]
        cost_vals = [2 ** x for x in range(-15, 16, 5)]
        epsilon_vals = [2 ** x for x in range(-15, 16, 5)]
        '''

        #Coarse search
        best_params, best_mse = grid_search(X_train_scaled, y_train, gamma_vals, cost_vals, epsilon_vals)
        '''
        best_C = np.log2(best_params['C'])
        best_epsilon = np.log2(best_params['epsilon'])
        best_gamma = np.log2(best_params['gamma'])
        '''
        '''
        gamma_vals = [2 ** x for x in range(int(best_gamma) - 5, int(best_gamma) + 5)]
        cost_vals = [2 ** x for x in range(int(best_C) - 5, int(best_C) + 5)]
        epsilon_vals = [2 ** x for x in range(int(best_epsilon) - 5, int(best_epsilon) + 5)]
        '''
        '''
        gamma_vals = np.logspace(best_gamma - 2, best_gamma + 2, 5, base=2)
        cost_vals = np.logspace(best_C - 2, best_C + 2, 5, base=2)
        epsilon_vals = np.logspace(best_epsilon - 2, best_epsilon + 2, 5, base=2)

        # Fine search
        best_params, best_mse = grid_search(X_train_scaled, y_train, gamma_vals, cost_vals, epsilon_vals)
        '''
        params[i] = best_params
        mse[i] = best_mse
    pickle.dump(params, open(path + '/best_params_1_1.pkl', 'wb'))
    pickle.dump(mse, open(path + '/best_mse_1_1.pkl', 'wb'))
else:
    params = pickle.load(open(load_dir, 'rb'))
    print("Loaded params")
    print(params)
    score_defn = {'accuracy' : make_scorer(accuracy_score), 'precision' : make_scorer(precision_score) }

    train_acc = []
    test_acc = []
    train_prec = []
    test_prec = []

    for i in range(14):
        X_train, X_test, y_train, y_test = split_data(i, face_landmarks, train_annotations)
        X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

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
        print(test_acc_score, test_prec_score)

        pickle.dump(train_acc, open(path + '/train_acc_1_1.pkl', 'wb'))
        pickle.dump(test_acc, open(path + '/test_acc_1_1.pkl', 'wb'))
        pickle.dump(train_prec, open(path + '/train_prec_1_1.pkl', 'wb'))
        pickle.dump(test_prec, open(path + '/test_prec_1_1.pkl', 'wb'))