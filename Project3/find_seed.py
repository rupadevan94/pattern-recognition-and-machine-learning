import pickle
import os
from skimage.io import imread
from skimage.feature import  hog
from utils import *
from sklearn.metrics import accuracy_score, precision_score
import pdb
from sklearn.svm import LinearSVC
from joblib import Parallel, delayed

def get_data(path, img_folder, lm_file):
    print(path + img_folder)
    print(path + lm_file)
    files = os.listdir(path + img_folder)
    fimages = [os.path.join(path + img_folder, f) for f in files if f[-3:] == 'jpg']
    fd = np.zeros((len(fimages), 7688))
    for i, fi in enumerate(fimages):
        image = imread(fi)
        fd[i], hog_images = hog(image, orientations=8, pixels_per_cell=(16, 16),
                                    cells_per_block=(1, 1), visualize=True, multichannel=True)
    data = loadmat(path + lm_file)
    face_landmarks = data['face_landmark']
    vote_diff = data['vote_diff']
    return fd, face_landmarks, vote_diff

def find_rank_data(X, voting_diff):
    n = len(X)
    print(n)
    #.set_trace()
    Xp = []
    diff = []
    yp = []
    for i in range(0, int(n/2), 2):
        Xp.append(X[i + 1] - X[i])
        diff.append(abs(voting_diff[i]))
        if np.sign(voting_diff[i + 1] - voting_diff[i]) == 0:
            yp.append(1)
        else:
            yp.append(np.sign(voting_diff[i + 1] - voting_diff[i]))

    for i in range(int(n/2), n, 2):
        Xp.append(X[i] - X[i + 1])
        diff.append(abs(voting_diff[i]))
        if np.sign(voting_diff[i] - voting_diff[i + 1]) == 0:
            yp.append(1)
        else:
            yp.append(np.sign(voting_diff[i] - voting_diff[i + 1]))
    yp = [int(yp[i]) for i in range(len(yp))]
    print("Unique yp values are..")
    print(np.unique(yp))
    return Xp, diff, yp

def rank_svm(elec_type, img_folder, lm_file, i):
    # create features
    print("Loading data....")
    print(path + '/' + elec_type + 'Xp_2_1.pkl')
    if not os.path.exists(path + '/' + elec_type + 'Xp_2_1.pkl'):
        fd, face_landmarks, vote_diff = get_data(path, img_folder, lm_file)
        print("Concatenating features...")
        concat_features = np.concatenate((fd, face_landmarks), axis=1)
        print(np.shape(concat_features))
        Xp, diff, yp = find_rank_data(concat_features, vote_diff)
        # Save features
        print("Saving values....")
        #pickle.dump(Xp, open(path + '/' + elec_type + 'Xp_2_1.pkl', 'wb'))
        #pickle.dump(yp, open(path + '/' + elec_type +  'yp_2_1.pkl', 'wb'))
    else:
        print("Reading saved values..")
        Xp = pickle.load(open(path + '/' + elec_type +  'Xp_2_1.pkl', 'rb'))
        yp = pickle.load(open(path + '/' + elec_type + 'yp_2_1.pkl', 'rb'))
    # Split data
    print("Splitting and saving values...")
    X_train, X_test, y_train, y_test = split_data_rank_svm(Xp, yp, i)
    # Scale data
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    #pdb.set_trace()
    # grid search
    print("Grid search....")
    if not os.path.exists(path + '/' + elec_type + 'best_params_2_1.pkl'):
        cost_vals = [2**x for x in range(-15, 15)]
        best_params, best_mse = grid_search_rank_svm(X_train_scaled, y_train, cost_vals)
        #pickle.dump(best_params, open(path + '/' + elec_type + 'best_params_2_1.pkl', 'wb'))
        #pickle.dump(best_mse, open(path + '/' + elec_type + 'best_mse_2_1.pkl', 'wb'))
    else:
        best_params = pickle.load(open(path + '/' + elec_type + 'best_params_2_1.pkl', 'rb'))
        best_mse = pickle.load(open(path + '/' + elec_type + 'best_mse_2_1.pkl', 'rb'))
    # Create model
    clf = LinearSVC(C=best_params['C'], fit_intercept=False)
    # Find train accuracy and precision
    print("Training....")
    clf.fit(X_train_scaled, y_train)
    y_train_predict = clf.predict(X_train_scaled)
    train_acc_score = accuracy_score(y_train, y_train_predict)
    train_prec_score = precision_score(y_train, y_train_predict)
    print("Training accuracy and precision for " + elec_type)
    print(train_acc_score, train_prec_score)
    # Find test accuracy and precision
    print("Testing...")
    y_test_predict = clf.predict(X_test_scaled)
    test_acc_score = accuracy_score(y_test, y_test_predict)
    test_prec_score = precision_score(y_test, y_test_predict)
    print("Testing accuracy and precision for " + elec_type)
    print(test_acc_score, test_prec_score)
    # Save all values
    print("Saving values...")
    #pickle.dump(train_acc_score, open(path + '/' + elec_type + 'train_acc_2_1.pkl', 'wb'))
    #pickle.dump(test_acc_score, open(path + '/' + elec_type + 'test_acc_2_1.pkl', 'wb'))
    #pickle.dump(train_prec_score, open(path + '/' + elec_type + 'train_prec_2_1.pkl', 'wb'))
    #pickle.dump(test_prec_score, open(path + '/' + elec_type + 'test_prec_2_1.pkl', 'wb'))
    return train_acc_score, test_acc_score, train_prec_score, test_prec_score

elec_type = 'gov'
path = 'C:/Users/hsatreya/Desktop/RupaFolder/project4_code_and_data'

if elec_type == 'sen':
    print("Setting values in sen..")
    img_folder = '/img-elec/senator'
    lm_file = '/stat-sen.mat'
elif elec_type == 'gov':
    img_folder = '/img-elec/governor'
    lm_file = '/stat-gov.mat'

vals = Parallel(n_jobs = 36)(delayed(rank_svm)(elec_type, img_folder, lm_file, i) for i in range(50))
train_acc_vals = [vals[i][0] for i in range(50)]
test_acc_vals = [vals[i][1] for i in range(50)]
train_prec_vals = [vals[i][2] for i in range(50)]
test_prec_vals = [vals[i][3] for i in range(50)]

pickle.dump(train_acc_vals, open(path + '/' + elec_type + 'train_acc_vals_2_1.pkl', 'wb'))
pickle.dump(test_acc_vals, open(path + '/' + elec_type + 'test_acc_vals_2_1.pkl', 'wb'))
pickle.dump(train_prec_vals, open(path + '/' + elec_type + 'train_prec_vals_2_1.pkl', 'wb'))
pickle.dump(test_prec_vals, open(path + '/' + elec_type + 'test_prec_vals_2_1.pkl', 'wb'))