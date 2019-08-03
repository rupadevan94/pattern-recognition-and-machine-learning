from utils import *
import numpy as np

path = 'C:/Users/rupa/Documents/Rupa/UCLA/Q1/M276A-PRML/project3_code_and_data/project4_code_and_data'
face_landmarks, train_annotations = load_data(path)
for i in range(14):
    print('********************************')
    print(i)
    greater_than_0 = np.size(np.where(train_annotations[:, i] > 0))
    less_than_0 = np.size(np.where(train_annotations[:, i] < 0))
    print(greater_than_0)
    print(less_than_0)
    assert (greater_than_0 + less_than_0) == 491