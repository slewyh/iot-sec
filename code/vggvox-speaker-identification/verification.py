import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, euclidean, cosine
from glob import glob
from keras.utils import to_categorical
from model import vggvox_model, vggvox_mod_model, siamese_network, train_for_classification, compile_model
from wav_reader import get_fft_spectrum
from prepare import build_buckets, get_training_test_data_pairing, get_fft_features_from_list_file
import constants as c


def get_embeddings_from_list_file(model, list_file, max_sec):
	buckets = build_buckets(max_sec, c.BUCKET_STEP, c.FRAME_STEP)
	result = pd.read_csv(list_file, delimiter=",")
	result['features'] = result['filename'].apply(lambda x: get_fft_spectrum(x, buckets))
	result['embedding'] = result['features'].apply(lambda x: np.squeeze(model.predict(x.reshape(1,*x.shape,1))))
	return result[['filename','speaker','embedding']]

def retrain(x_train, y_train):
	print("Loading model weights from [{}]....".format(c.WEIGHTS_FILE))
	
	baseline_model = vggvox_model()
	baseline_model.load_weights(c.WEIGHTS_FILE)
	
	print("Creating base network ...")
	model = vggvox_mod_model(baseline_model)
	model.summary()
	
	train_for_classification(model, x_train, y_train)
	
	# print("Creating siamese network ...")	
	# siamese_model = siamese_network(input_shape, model)

	# print("Training....")
	# siamese_model = train_siamese(siamese_model, tr_pairs, tr_y)

	return model

def get_id_result():

	#get training and testing pair for training
	# input_shape, tr_pairs, tr_y, te_pairs, te_y  = get_training_test_data_pairing()
	(x_train, y_train) = get_fft_features_from_list_file(c.ENROLL_LIST_FILE, c.MAX_SEC)
	(x_test, y_test) = get_fft_features_from_list_file(c.TEST_LIST_FILE, c.MAX_SEC)
	y_train = to_categorical(y_train, num_classes=c.NUM_CLASSES)
	y_test = to_categorical(y_test, num_classes=c.NUM_CLASSES)

	print("Y_train.shape: ", y_train.shape)

	if c.RETRAIN: 
		model = retrain(x_train, y_train)
	else:
		baseline_model = vggvox_model()
		model = vggvox_mod_model(baseline_model)
		# model = siamese_network(input_shape, modified_model)
		model.load_weights(c.VGGM_WEIGHTS_FILE)
		# compile model
		model = compile_model(model)

	score = model.evaluate(x_test, y_test, verbose=1)
	print("loss: {}, top-1 accuracy (/%): {}, top-5 accuracy (/%): {}".format(score[0],score[1],score[2]))

	# print("Processing enroll samples....")
	# enroll_result = get_embeddings_from_list_file(model, c.ENROLL_LIST_FILE, c.MAX_SEC)
	# enroll_embs = np.array([emb.tolist() for emb in enroll_result['embedding']])
	# speakers = enroll_result['speaker']

	# print("Processing test samples....")
	# test_result = get_embeddings_from_list_file(model, c.TEST_LIST_FILE, c.MAX_SEC)
	# test_embs = np.array([emb.tolist() for emb in test_result['embedding']])

	# print("Comparing test samples against enroll samples....")
	# distances = pd.DataFrame(cdist(test_embs, enroll_embs, metric=c.COST_METRIC), columns=speakers)

	# scores = pd.read_csv(c.TEST_LIST_FILE, delimiter=",",header=0,names=['test_file','test_speaker'])
	# scores = pd.concat([scores, distances],axis=1)
	# scores['result'] = scores[speakers].idxmin(axis=1)
	# scores['correct'] = (scores['result'] == scores['test_speaker'])*1. # bool to int

	# print("Writing outputs to [{}]....".format(c.RESULT_FILE))
	# result_dir = os.path.dirname(c.RESULT_FILE)
	
	# if not os.path.exists(result_dir):
	#     os.makedirs(result_dir)
	# with open(c.RESULT_FILE, 'w') as f:
	# 	scores.to_csv(f, index=False)

if __name__ == '__main__':
	get_id_result()
