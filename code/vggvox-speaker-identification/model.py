import scipy.io as sio
import numpy as np
import keras
import keras.backend as K
from keras.layers import Input, GlobalAveragePooling2D, Reshape, Lambda, Dense, Flatten
from keras.layers.convolutional import Conv2D, ZeroPadding2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Lambda, Activation
from keras.models import Model
from keras import optimizers
from keras.callbacks import LearningRateScheduler
import constants as c
import matplotlib.pyplot as plt
from keras import metrics

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

# Block of layers: Conv --> BatchNorm --> ReLU --> Pool
def conv_bn_pool(inp_tensor,layer_idx,conv_filters,conv_kernel_size,conv_strides,conv_pad,
	pool='',pool_size=(2, 2),pool_strides=None,
	conv_layer_prefix='conv'):
	x = ZeroPadding2D(padding=conv_pad,name='pad{}'.format(layer_idx))(inp_tensor)
	x = Conv2D(filters=conv_filters,kernel_size=conv_kernel_size, strides=conv_strides, padding='valid', name='{}{}'.format(conv_layer_prefix,layer_idx))(x)
	x = BatchNormalization(epsilon=1e-5,momentum=1,name='bn{}'.format(layer_idx))(x)
	x = Activation('relu', name='relu{}'.format(layer_idx))(x)
	if pool == 'max':
		x = MaxPooling2D(pool_size=pool_size,strides=pool_strides,name='mpool{}'.format(layer_idx))(x)
	elif pool == 'avg':
		x = AveragePooling2D(pool_size=pool_size,strides=pool_strides,name='apool{}'.format(layer_idx))(x)
	return x


# Block of layers: Conv --> BatchNorm --> ReLU --> Dynamic average pool (fc6 -> apool6 only)
def conv_bn_dynamic_apool(inp_tensor,layer_idx,conv_filters,conv_kernel_size,conv_strides,conv_pad,
	conv_layer_prefix='conv'):
	x = ZeroPadding2D(padding=conv_pad,name='pad{}'.format(layer_idx))(inp_tensor)
	x = Conv2D(filters=conv_filters,kernel_size=conv_kernel_size, strides=conv_strides, padding='valid', name='{}{}'.format(conv_layer_prefix,layer_idx))(x)
	x = BatchNormalization(epsilon=1e-5,momentum=1,name='bn{}'.format(layer_idx))(x)
	x = Activation('relu', name='relu{}'.format(layer_idx))(x)
	x = GlobalAveragePooling2D(name='gapool{}'.format(layer_idx))(x)
	x = Reshape((1,1,conv_filters),name='reshape{}'.format(layer_idx))(x)
	return x


# VGGVox verification model
def vggvox_model():
	inp = Input(c.INPUT_SHAPE,name='input')
	x = conv_bn_pool(inp,layer_idx=1,conv_filters=96,conv_kernel_size=(7,7),conv_strides=(2,2),conv_pad=(1,1),
		pool='max',pool_size=(3,3),pool_strides=(2,2))
	x = conv_bn_pool(x,layer_idx=2,conv_filters=256,conv_kernel_size=(5,5),conv_strides=(2,2),conv_pad=(1,1),
		pool='max',pool_size=(3,3),pool_strides=(2,2))
	x = conv_bn_pool(x,layer_idx=3,conv_filters=384,conv_kernel_size=(3,3),conv_strides=(1,1),conv_pad=(1,1))
	x = conv_bn_pool(x,layer_idx=4,conv_filters=256,conv_kernel_size=(3,3),conv_strides=(1,1),conv_pad=(1,1))
	x = conv_bn_pool(x,layer_idx=5,conv_filters=256,conv_kernel_size=(3,3),conv_strides=(1,1),conv_pad=(1,1),
		pool='max',pool_size=(5,3),pool_strides=(3,2))		
	x = conv_bn_dynamic_apool(x,layer_idx=6,conv_filters=4096,conv_kernel_size=(9,1),conv_strides=(1,1),conv_pad=(0,0),
		conv_layer_prefix='fc')
	x = conv_bn_pool(x,layer_idx=7,conv_filters=1024,conv_kernel_size=(1,1),conv_strides=(1,1),conv_pad=(0,0),
		conv_layer_prefix='fc')
	x = Lambda(lambda y: K.l2_normalize(y, axis=3), name='norm')(x)  #L2-normalization
	x = Conv2D(filters=1024,kernel_size=(1,1), strides=(1,1), padding='valid', name='fc8')(x)

	m = Model(inp, x, name='VGGVox')
	# print("*"*10, m.input_shape)
	return m

def siamese_network(input_shape, model):

	input_a = Input(shape=input_shape)
	input_b = Input(shape=input_shape)

	# network definition
	base_network = vggvox_mod_model(model)
	# because we re-use the same instance `base_network`,
	# the weights of the network
	# will be shared across the two branches
	processed_a = base_network(input_a)
	processed_b = base_network(input_b)

	distance = Lambda(euclidean_distance,
					output_shape=eucl_dist_output_shape)([processed_a, processed_b])

	model = Model([input_a, input_b], distance, name='Siamese')

	return model

def vggvox_mod_model(model):
	for layer in model.layers[:-1]:
		layer.trainable = False

	# # Check the trainable status of the individual layers
	# for layer in model.layers:
	# 	print(layer, layer.trainable)
	
	#hidden layers part
	model.layers.pop()  #pop last layer
	conv_layer = Conv2D(filters=256,kernel_size=(1,1), strides=(1,1), padding='valid', name='fc8')  #final fc layer reduced to 256	
	
	inp = model.input
	out = conv_layer(model.layers[-1].output)

	#classification part
	fc9 = Dense(c.NUM_CLASSES, activation='softmax', name='fc9')
	out =  Flatten()(out)
	out = fc9 (out)

	model2 = Model(inp, out, name='VGG-M')

	return model2

#define step decay function
class LossHistory_(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []
        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(exp_decay(len(self.losses)))
        print('lr:', exp_decay(len(self.losses)))

def exp_decay(epoch):
    initial_lrate = 1e-2
    k = 1e-8
    lrate = initial_lrate * np.exp(-k*epoch)
    return lrate

def plot_fig(i, history):
    fig = plt.figure()
    # plt.plot(range(1,c.EPOCHS+1),history.history['val_acc'],label='validation')
    plt.plot(range(1,c.EPOCHS+1),history.history['acc'],label='training')
    plt.legend(loc=0)
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.xlim([1,c.EPOCHS])
#   plt.ylim([0,1])
    plt.grid(True)
    plt.title("Model Accuracy")
    plt.show()
    fig.savefig('img/'+str(i)+'-accuracy.jpg')
    plt.close(fig)

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)

def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

def train_siamese(model, tr_pairs, tr_y):
	# define SGD optimizer
	sgd = optimizers.SGD(lr=0.00, decay=c.WEIGHT_DECAY, momentum=c.SGD_MOMENTUM, nesterov=True)
	model.compile(loss=contrastive_loss, optimizer=sgd, metrics=[accuracy])
	
	# compile the model
	# learning schedule callback
	loss_history_ = LossHistory_()
	lrate_ = LearningRateScheduler(exp_decay)
	callbacks_list_ = [loss_history_, lrate_]

	# fit the model
	print("tr_pairs 0 shape {}, tr_pairs 1 {}".format(tr_pairs[:, 0].shape, tr_pairs[:, 1].shape))
	history = model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
		validation_split=0.1,
		epochs=c.EPOCHS, 
		batch_size=c.BATCH_SIZE, 
		callbacks=callbacks_list_)	

	# Save the model
	weights_file = model.name + '_weights.h5'
	model.save(weights_file)

	# plot model accuracy
	# plot_fig(model.name, history)
	
	return model

def top_1_categorical_accuracy(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=1) 

def top_5_categorical_accuracy(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=5) 


def compile_model(model):
	# define SGD optimizer
	sgd = optimizers.SGD(lr=0.00, decay=c.WEIGHT_DECAY, momentum=c.SGD_MOMENTUM, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=[top_1_categorical_accuracy,top_5_categorical_accuracy])
			
	return model

def train_for_classification(model, tr_X, tr_y):
	# learning schedule callback
	loss_history_ = LossHistory_()
	lrate_ = LearningRateScheduler(exp_decay)
	callbacks_list_ = [loss_history_, lrate_]

	# fit the model
	history = model.fit(tr_X, tr_y,
		validation_split=0.1,
		epochs=c.EPOCHS,
		batch_size=c.BATCH_SIZE, 
		callbacks=callbacks_list_)	
	
	print(history.history)

	# Save the model
	weights_file = model.name + '_weights.h5'
	model.save(weights_file)

	# # plot model accuracy
	# plot_fig(model.name, history)
	
	return model

def test():
	model = vggvox_model()
	num_layers = len(model.layers)

	x = np.random.randn(1,512,30,1)
	outputs = []

	for i in range(num_layers):
		get_ith_layer_output = K.function([model.layers[0].input, K.learning_phase()],
		                              [model.layers[i].output])	
		layer_output = get_ith_layer_output([x, 0])[0] 	# output in test mode = 0
		outputs.append(layer_output)

	for i in range(11):
		print("Shape of layer {} output:{}".format(i, outputs[i].shape))

if __name__ == '__main__':
	test()

