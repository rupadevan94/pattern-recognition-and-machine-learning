from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import time
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def load_data(path):
    train_data = loadmat(path + '/train-anno.mat')
    face_landmarks = train_data['face_landmark']
    train_annotations = train_data['trait_annotation']

    return face_landmarks, train_annotations

def split_data(i, X, train_annotations):
    #Split data set into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, train_annotations[:, i], test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def split_data_rank_svm(X, y, i):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled[X_train_scaled > 1] = 1.0
    return X_train_scaled, X_test_scaled

def grid_search(X_train_scaled, y_train, gamma_vals, cost_vals, epsilon_vals):
    scorer = make_scorer(mean_squared_error, greater_is_better= False)

    start = time.clock()
    svr = SVR(kernel= 'rbf')
    parameters = {'gamma' : gamma_vals, 'C' : cost_vals, 'epsilon' : epsilon_vals}
    print("Param calculation started***")
    clf = GridSearchCV(svr, parameters, cv = 10, n_jobs=-1, scoring=scorer, verbose=1)
    clf.fit(X_train_scaled, y_train)
    print("Calculated**")
    print(clf.best_params_)
    print(abs(clf.best_score_))
    end = time.clock()
    print("Time taken : " + str(end - start))
    return clf.best_params_, clf.best_score_

def grid_search_rank_svm(X_train_scaled, y_train, cost_vals):
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    start = time.clock()
    svc = LinearSVC(fit_intercept=False, max_iter = 10000)
    parameters = {'C': cost_vals}
    print(parameters)
    print("Param calculation started***")
    clf = GridSearchCV(svc, parameters, cv=10, n_jobs=-1, scoring=scorer, verbose=1)
    clf.fit(X_train_scaled, y_train)
    print("Calculated**")
    print(clf.best_params_)
    print(abs(clf.best_score_))
    end = time.clock()
    print("Time taken : " + str(end - start))
    return clf.best_params_, clf.best_score_

def find_threshold(y_train):
    return np.mean(y_train)

def find_labels(y_vals, threshold):
    return np.where(y_vals > threshold, 1, -1)

def plot_values(vals1, vals2, labels1, labels2, fig_name, x_label, y_label, legend_pos, folder):
    path = 'C:/Users/rupa/Documents/Rupa/UCLA/Q1/M276A-PRML/project3_code_and_data/project4_code_and_data/final/' + folder
    plt.plot(vals1, marker = 'o', linestyle = 'dashed', label = labels1)
    plt.plot(vals2, marker = 'o', linestyle = 'dashed', label = labels2)

    plt.legend(loc=legend_pos)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    #plt.savefig(path + '/histogram_%d_' % t + self.boosting_type + '.png')
    plt.savefig(path + '/' + fig_name + '.png')
    plt.gcf().clear()

def plot_mse(vals, fig_name, folder):
    path = 'C:/Users/rupa/Documents/Rupa/UCLA/Q1/M276A-PRML/project3_code_and_data/project4_code_and_data/final/' + folder
    plt.plot([abs(vals[i]) for i in range(14)])
    plt.scatter(list(range(14)), [abs(vals[i]) for i in range(14)], marker="o", color='r')
    plt.xlabel('Facial Trait (Model Number)')
    plt.ylabel('Mean Squared Error')
    plt.savefig(path + '/' + fig_name + '.png')

def plot_mse_2(vals1, vals2, label1, label2, fig_name):
    path = 'C:/Users/rupa/Documents/Rupa/UCLA/Q1/M276A-PRML/project3_code_and_data/project4_code_and_data/final/part_1_2'
    #marker = 'o', linestyle = 'dashed'
    plt.plot([abs(vals1[i]) for i in range(14)], marker = 'o', linestyle = 'dashed', label = label1)
    #plt.scatter(list(range(14)), [abs(vals1[i]) for i in range(14)], marker="o", color='r')
    plt.plot([abs(vals2[i]) for i in range(14)], marker = 'o', linestyle = 'dashed', label = label2)
    #plt.scatter(list(range(14)), [abs(vals1[i]) for i in range(14)], marker="o", color='r')
    plt.legend(loc = 'upper right')
    plt.xlabel('Facial Trait (Model Number)')
    plt.ylabel('Mean Squared Error')
    plt.savefig(path + '/' + fig_name + '.png')

def plot_acc_comp(train_vals, test_vals, train_labels, test_labels, fig_name, x_label, y_label, folder):
    path = 'C:/Users/rupa/Documents/Rupa/UCLA/Q1/M276A-PRML/project3_code_and_data/project4_code_and_data/final/' + folder
    plt.plot(train_vals[0], marker='o', linestyle='dashed', label=train_labels[0])
    plt.plot(train_vals[1], marker='o', linestyle='dashed', label=train_labels[1])
    plt.plot(test_vals[0], marker='o', linestyle='dashed', label=test_labels[0])
    plt.plot(test_vals[1], marker='o', linestyle='dashed', label=test_labels[1])
    lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.2),  ncol=2)
    art = []
    art.append(lgd)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.savefig(path + '/histogram_%d_' % t + self.boosting_type + '.png')
    #plt.show()
    print(path + '/' + fig_name)
    plt.savefig(path + '/' + fig_name + '.png', bbox_inches="tight", additional_artists=art)