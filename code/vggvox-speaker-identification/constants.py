# from pyaudio import paInt16

# Signal processing
SAMPLE_RATE = 16000  #Hertz
PREEMPHASIS_ALPHA = 0.97
FRAME_LEN = 0.025  #hamming window width (seconds)
FRAME_STEP = 0.01   #hamming window step (seconds)
NUM_FFT = 512
BUCKET_STEP = 1
MAX_SEC = 3

# Model
WEIGHTS_FILE = "data/model/weights.h5"
VGGM_WEIGHTS_FILE = "data/model/VGG-M_weights.h5"
COST_METRIC = "cosine"  # euclidean or cosine
INPUT_SHAPE=(NUM_FFT,None,1)

# Training
EPOCHS = 2 #50
BATCH_SIZE = 128 #256
SGD_MOMENTUM = 0.9
WEIGHT_DECAY = 5E-4
NUM_CLASSES = 1251
# NUM_CLASSES = 3
# IO

ENROLL_LIST_FILE = "cfg/enroll_list.csv"
TEST_LIST_FILE = "cfg/test_list.csv"
RESULT_FILE = "res/results.csv"
TRAIN_LIST_FILE = "cfg/enroll_list_large.csv"
SPEAKER_PREFIX = "id1"
RETRAIN = False