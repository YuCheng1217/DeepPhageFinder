import os
import sys
import optparse

prog_base = os.path.split(sys.argv[0])[1]

parser = optparse.OptionParser()
parser.add_option("-l", "--len", action = "store", type = int, dest = "contigLength",default = 3000, help = "contig Length")
parser.add_option("-i", "--intr", action = "store", type = "string", dest = "inDir", help = "input directory for training and validation data")
parser.add_option("-o", "--out", action = "store", type = "string", dest = "outDir",default='./', help = "output directory")
parser.add_option("-f", "--fLen", action = "store", type = int, dest = "filter_len",default=10,help = "the length of filter")
parser.add_option("-n", "--fNum", action = "store", type = int, dest = "nb_filter",default=1000, help = "number of filters in the convolutional layer")
parser.add_option("-d", "--dense", action = "store", type = int, dest = "nb_dense",default=400, help = "number of neurons in the dense layer")
parser.add_option("-e", "--epochs", action = "store", type = int, dest = "epochs",default=10, help = "number of epochs")
parser.add_option("-g", "--gpu", action = "store", type = int, dest = "gpu",default=1, help = "number of gpu")

(options, args) = parser.parse_args()
if (options.inDir is None) :
    sys.stderr.write(prog_base + ": ERROR: missing required command-line argument")
    filelog.write(prog_base + ": ERROR: missing required command-line argument")
    parser.print_help()
    sys.exit(0)


contigLength = options.contigLength
filter_len = options.filter_len
nb_filter = options.nb_filter
nb_dense = options.nb_dense
inDir = options.inDir
outDir = options.outDir
nb_gpu = options.gpu

import numpy as np
import random
import tensorflow as tf
import sklearn
from sklearn.metrics import roc_auc_score

os.environ['KERAS_BACKEND']='tensorflow'

channel_num = 4
gpu_devices =  ''
gpu_devices_mirror = []
for i in range(nb_gpu):
    gpu_devices = gpu_devices + str(i) + ','
    gpu_devices_mirror.append("/gpu:"+str(i))
gpu_devices = gpu_devices[:-1]


#os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices

if not os.path.exists(outDir):
    os.makedirs(outDir)
epochs = options.epochs


contigLengthk = contigLength/1000
if contigLengthk.is_integer() :
    contigLengthk = int(contigLengthk)

contigLengthk = str(contigLengthk)

teSampleRate = 0.1
valSampleRate = 0.1
rdseed = 0
random.seed(rdseed)

######## loading data for training, validation ##########
print("...loading data...")
filepath_phagefw = inDir + '/phage_code.npy'
filepath_phagebw =inDir + '/phage_codeR.npy'
filepath_hostfw = inDir + '/host_code.npy'
filepath_hostbw = inDir + '/host_codeR.npy'

print("...loading positive strand data...")
hostRef_codefw = np.load(filepath_hostfw)
phageRef_codefw = np.load(filepath_phagefw)
Y = np.concatenate((np.repeat(0, hostRef_codefw.shape[0]), np.repeat(1, phageRef_codefw.shape[0])))
X_fw = np.concatenate((hostRef_codefw, phageRef_codefw), axis=0)
del hostRef_codefw, phageRef_codefw

print("...loading negative strand data...")
phageRef_codebw = np.load(filepath_phagebw)
hostRef_codebw = np.load(filepath_hostbw)
X_bw = np.concatenate((hostRef_codebw, phageRef_codebw), axis=0)
del hostRef_codebw, phageRef_codebw

print("...shuffling data...")
index_fw = list(range(0, X_fw.shape[0]))
np.random.shuffle(index_fw)
X_fw_shuf = X_fw[np.ix_(index_fw, range(X_fw.shape[1]), range(X_fw.shape[2]))]
del X_fw
X_bw_shuf = X_bw[np.ix_(index_fw, range(X_bw.shape[1]), range(X_bw.shape[2]))]
del X_bw
Y_shuf = Y[index_fw]

cut_point = int(len(X_bw_shuf)/9)
X_fw_val = X_fw_shuf[:cut_point]
X_fw_tr = X_fw_shuf[cut_point:]
X_bw_val = X_bw_shuf[:cut_point]
X_bw_tr = X_bw_shuf[cut_point:]
Y_val = Y_shuf[:cut_point]
Y_tr = Y_shuf[cut_point:]

del X_fw_shuf
del X_bw_shuf
del Y_shuf


######### training model #############
# parameters
POOL_FACTOR = 1
dropout_cnn = 0.1
dropout_pool = 0.1
dropout_dense = 0.1
learningrate = 0.001
#batch_size=int(X_fw_tr.shape[0]/(1000*10000/contigLength//4*4)) ## smaller batch size can reduce memory
batch_size=256
print('batch_size is ' + str(batch_size))
pool_len1 = int((contigLength-filter_len+1)/POOL_FACTOR)

modPattern = 'model_cl'+str(contigLength)+'_fl'+str(filter_len)+'_fn'+str(nb_filter)+'_dn'+str(nb_dense)
modName = os.path.join( outDir, modPattern + '.h5')
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=modName, verbose=1,save_best_only=True)
earlystopper = tf.keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5, verbose=1)

#strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1","/gpu:2","/gpu:3"])
strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1","/gpu:2","/gpu:3"])
#strategy = tf.distribute.MirroredStrategy()
print('Number of devices: %d' % strategy.num_replicas_in_sync)
##### build model #####

def get_output(input_layer, hidden_layers):
    output = input_layer
    for hidden_layer in hidden_layers:
        output = hidden_layer(output)
    return output
def create_model():
    forward_input = tf.keras.Input(shape=(None, channel_num))
    reverse_input = tf.keras.Input(shape=(None, channel_num))
    hidden_layers = [
      tf.keras.layers.Conv1D(filters = nb_filter, kernel_size = filter_len, activation='relu'),
      tf.keras.layers.GlobalMaxPooling1D(),
      tf.keras.layers.Dropout(dropout_pool),
      tf.keras.layers.Dense(nb_dense, activation='relu'),
      tf.keras.layers.Dropout(dropout_dense),
      tf.keras.layers.Dense(nb_dense, activation='relu'),
      tf.keras.layers.Dropout(dropout_dense),
      tf.keras.layers.Dense(1, activation='sigmoid')
    ]
    forward_output = get_output(forward_input, hidden_layers)
    reverse_output = get_output(reverse_input, hidden_layers)
    output = tf.keras.layers.Average()([forward_output, reverse_output])
    model_created = tf.keras.Model(inputs=[forward_input, reverse_input], outputs=output)
    return model_created


print("...building model...")
## if model exists
if os.path.isfile(modName):
    model = load_model(modName)
    print("...model exists...")
else :
    with strategy.scope():
        model = create_model()
        model.compile(tf.keras.optimizers.Adam(lr=learningrate), 'binary_crossentropy', metrics=['accuracy'])

print("...fitting model...")
print('cl' + str(contigLength)+'_fl'+str(filter_len)+'_fn'+str(nb_filter)+'_dn'+str(nb_dense)+'_ep'+str(epochs))
model.fit(x = [X_fw_tr, X_bw_tr], y = Y_tr, \
            batch_size=batch_size, epochs=epochs, verbose=2, \
            validation_data=([X_fw_val, X_bw_val], Y_val), \
            callbacks=[checkpointer, earlystopper])



## Final evaluation AUC ###

## train data
type = 'tr'
print("...predicting "+type+"...\n")
Y_pred = model.predict([X_fw_tr, X_bw_tr], batch_size=1)
auc = sklearn.metrics.roc_auc_score(Y_tr, Y_pred)
print('auc_'+type+'='+str(auc)+'\n')
del Y_tr, X_fw_tr, X_bw_tr


# val data
type = 'val'
print("...predicting "+type+"...\n")
Y_pred = model.predict([X_fw_val, X_bw_val], batch_size=1)
auc = sklearn.metrics.roc_auc_score(Y_val, Y_pred)
print('auc_'+type+'='+str(auc)+'\n')
np.savetxt(os.path.join(outDir, modPattern + '_' + type + 'fw_Y_pred.txt'), np.transpose(Y_pred))
np.savetxt(os.path.join(outDir, modPattern + '_' + type + 'fw_Y_true.txt'), np.transpose(Y_val))
del Y_val, X_fw_val, X_bw_val

# test data
filepath_hostfw = '/home/chengyu/database/encode_0.1_test_set_0525/host_code.npy'
filepath_phagefw = '/home/chengyu/database/encode_0.1_test_set_0525/phage_code.npy'
filepath_phagebw = '/home/chengyu/database/encode_0.1_test_set_0525/host_codeR.npy'
filepath_hostbw = '/home/chengyu/database/encode_0.1_test_set_0525/phage_codeR.npy'

print("...loading positive strand data for testing...")
hostRef_codefw = np.load(filepath_hostfw)
phageRef_codefw = np.load(filepath_phagefw)
Y_test = np.concatenate((np.repeat(0, hostRef_codefw.shape[0]), np.repeat(1, phageRef_codefw.shape[0])))
X_fw_test = np.concatenate((hostRef_codefw, phageRef_codefw), axis=0)
del hostRef_codefw, phageRef_codefw

print("...loading negative strand data for testing...")
phageRef_codebw = np.load(filepath_phagebw)
hostRef_codebw = np.load(filepath_hostbw)
X_bw_test = np.concatenate((hostRef_codebw, phageRef_codebw), axis=0)
del hostRef_codebw, phageRef_codebw

type = 'testing set'
print("...predicting "+type+"...\n")
Y_pred = model.predict([X_fw_test, X_bw_test], batch_size=1)
auc = sklearn.metrics.roc_auc_score(Y_test, Y_pred)
print('auc_'+type+'='+str(auc)+'\n')
del Y_val, X_fw_val, X_bw_val