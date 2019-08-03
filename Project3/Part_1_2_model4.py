from skimage.feature import hog
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

#path = 'C:/Users/rupa/Documents/Rupa/UCLA/Q1/M276A-PRML/project3_code_and_data/project4_code_and_data'
path = 'C:/Users/hsatreya/Desktop/RupaFolder/project4_code_and_data'
load_dir = path + '/concat_features_1_2.pkl'
print(load_dir)

if not os.path.exists(load_dir):
    print("Dir not there yet")
    face_landmarks, train_annotations = load_data(path)
    h = 500
    w = 500
    files = os.listdir(path + '/img')
    fimages = [os.path.join(path + '/img', f) for f in files if f[-3:] == 'jpg']
    # hog_images = np.zeros((len(fimages), h, w))
    fd = np.zeros((len(fimages), 7688))

    for i, fi in enumerate(fimages):
        image = imread(fi)
        fd[i], hog_images = hog(image, orientations=8, pixels_per_cell=(16, 16),
                                cells_per_block=(1, 1), visualize=True, multichannel=True)
        # pdb.set_trace()

    print(fd.shape)
    print(np.shape(face_landmarks))
    # hog_images = np.reshape(hog_images, (491, 500*500))
    concat_features = np.concatenate((fd, face_landmarks), axis=1)
    print(np.shape(concat_features))
    pickle.dump(concat_features, open(path + '/concat_features_1_2.pkl', 'wb'))
    pickle.dump(train_annotations, open(path + '/y_1_2.pkl', 'wb'))

else:

    concat_features = pickle.load(open(path + '/concat_features_1_2.pkl', 'rb'))
    # concat_features = loadmat(path + '/hogfeatures.mat')
    # concat_features = concat_features['hogfeats']
    # concat_features = np.reshape(concat_features, ((491, 6272)))
    train_annotations = pickle.load(open(path + '/y_1_2.pkl', 'rb'))
    print("Loaded valuse**")
    # X_train, X_test, y_train, y_test = split_data(i, concat_features, train_annotations)
    # X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    # path = 'C:/Users/rupa/Documents/Rupa/UCLA/Q1/M276A-PRML/project3_code_and_data/project4_code_and_data'


    #load_dir = path + '/best_params_1_2.pkl'
    load_dir = path + '/model4_param.pkl'
    face_landmarks, train_annotations = load_data(path)
    #if not os.path.exists(load_dir):
    if 1 == 1:
        params = {}
        mse = {}

        X_train, X_test, y_train, y_test = split_data(4, concat_features, train_annotations)
        X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

        print(np.sum(np.where(X_train_scaled > 1)))
        print("Computed Scaled values")

        gamma_vals = [2 ** -16]
        cost_vals = [2 ** 15]
        epsilon_vals = [0.25]
        # Coarse search

        print("Coarse grid search")
        best_params, best_mse = grid_search(X_train_scaled, y_train, gamma_vals, cost_vals, epsilon_vals)

        print("Done")

        score_defn = {'accuracy': make_scorer(accuracy_score), 'precision': make_scorer(precision_score)}

        svr = SVR(kernel='rbf', gamma=best_params['gamma'], C=best_params['C'], epsilon=best_params['epsilon'])
        svr.fit(X_train_scaled, y_train)

        y_train_predict = svr.predict(X_train_scaled)
        threshold = find_threshold(y_train)
        y_train_true_labels = find_labels(y_train, threshold)
        y_train_predict_labels = find_labels(y_train_predict, threshold)

        train_acc_score = accuracy_score(y_train_true_labels, y_train_predict_labels)
        train_prec_score = precision_score(y_train_true_labels, y_train_predict_labels)
        print("Training accuracy and precision are")
        print(train_acc_score, train_prec_score)

        print("Testing accuracy and precision are ")
        y_test_predict = svr.predict(X_test_scaled)
        y_test_true_labels = find_labels(y_test, threshold)
        y_test_predict_labels = find_labels(y_test_predict, threshold)

        test_acc_score = accuracy_score(y_test_true_labels, y_test_predict_labels)
        test_prec_score = precision_score(y_test_true_labels, y_test_predict_labels)
        print(test_acc_score, test_prec_score)

        #pickle.dump(best_params, open(load_dir, 'wb'))
        #pickle.dump(best_params, open(path + '/model4_mse.pkl', 'wb'))