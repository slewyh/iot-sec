import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, euclidean, cosine
import random
from wav_reader import get_fft_spectrum
import constants as c


def build_buckets(max_sec, step_sec, frame_step):
	buckets = {}
	frames_per_sec = int(1/frame_step)
	end_frame = int(max_sec*frames_per_sec)
	step_frame = int(step_sec*frames_per_sec)
	for i in range(0, end_frame+1, step_frame):
		s = i
		s = np.floor((s-7+2)/2) + 1  # conv1
		s = np.floor((s-3)/2) + 1  # mpool1
		s = np.floor((s-5+2)/2) + 1  # conv2
		s = np.floor((s-3)/2) + 1  # mpool2
		s = np.floor((s-3+2)/1) + 1  # conv3
		s = np.floor((s-3+2)/1) + 1  # conv4
		s = np.floor((s-3+2)/1) + 1  # conv5
		s = np.floor((s-3)/2) + 1  # mpool5
		s = np.floor((s-1)/1) + 1  # fc6
		if s > 0:
			buckets[i] = int(s)
	# print("buckets length {0}".format(len(buckets)))
	return buckets


def get_training_test_data_pairing():
	# the data, split between train and test sets
    (x_train, y_train) = get_fft_features_from_list_file(c.ENROLL_LIST_FILE, c.MAX_SEC)
    # print("y training ", y_train)
    (x_test, y_test) = get_fft_features_from_list_file(c.TEST_LIST_FILE, c.MAX_SEC)

    input_shape = x_train.shape[1:]
  
	# create training+test positive and negative pairs
    person_indices = [np.where(y_train == i)[0] for i in range(1,c.NUM_CLASSES+1)]
    # print(person_indices)
    tr_pairs, tr_y = create_pairs(x_train, person_indices)

    person_indices = [np.where(y_test == i)[0] for i in range(1,c.NUM_CLASSES+1)]
    print("test:", person_indices)
    te_pairs, te_y = create_pairs(x_test, person_indices)

    print("input_shape: {},  tr_pairs.shape: {}".format(input_shape, tr_pairs.shape))
    return input_shape, tr_pairs, tr_y, te_pairs, te_y 


def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    num_classes = c.NUM_CLASSES #- 1 #it is num_classes-1 because we don't have id 0 speaker. it is not zero index.
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1  
    print("n: ", n)
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


def get_fft_features_from_list_file(list_file, max_sec):
    
    buckets = build_buckets(max_sec, c.BUCKET_STEP, c.FRAME_STEP)
    result = pd.read_csv(list_file, delimiter=",")
    result['features'] = result['filename'].apply(lambda x: get_fft_spectrum(x, buckets))

    # iterate over rows with iterrows
    dataX = []
    dataY = []
    
    for index, row in result.iterrows():
        x = row['features']
        y = row['speaker']
        # print(row['speaker'])
        if isinstance(y, str): 
            y = y.replace(c.SPEAKER_PREFIX,'')
            y = int(y, 10)
       
        dataY.append(y)
        values = x.reshape(*x.shape,1)
        # print("here", values.shape)
        if(values.shape[1] == c.MAX_SEC*100):  #same time
            dataX.append(values)
 
    dataX = np.stack((dataX), axis=0)
    dataY = np.asarray(dataY)
    # print(dataY)
    print("X.shape: {}, Y.shape{}".format(dataX.shape, dataY.shape))

    return dataX, dataY
