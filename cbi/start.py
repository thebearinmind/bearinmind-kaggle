#%%
import sys
sys.path.insert(0, "/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages")
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import os
import numpy as np
import tqdm
from sklearn.model_selection import train_test_split
import random
import tensorflow as tf
from sklearn import preprocessing 
from sklearn.metrics import confusion_matrix, f1_score

# %%
df_train = pd.read_csv("train.csv")

#%% 
base_dir = "./small_sample/"
id_list = os.listdir("./small_sample")
random.shuffle(id_list)
train_idx = id_list[:80] 
test_idx = id_list[80:]

num_classes = df_train[df_train['filename'].isin(id_list)]['ebird_code'].unique().shape[0]

#%%
#le = preprocessing.LabelEncoder()
#le.fit(df_train['ebird_code'])
#df_train['label_encoded'] = le.transform(df_train['ebird_code'])

#%%
class Config(object):
    def __init__(self,
                 sampling_rate=22050, audio_duration=30, n_classes=16,
                 use_mfcc=False, n_folds=10, learning_rate=0.0001, 
                 max_epochs=50, n_mfcc=20):
        self.sampling_rate = sampling_rate
        self.audio_duration = audio_duration
        self.n_classes = n_classes
        self.use_mfcc = use_mfcc
        self.n_mfcc = n_mfcc
        self.n_folds = n_folds
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

        self.audio_length = self.sampling_rate * self.audio_duration
        if self.use_mfcc:
            self.dim = (self.n_mfcc, 1 + int(np.floor(self.audio_length/512)), 1)
        else:
            self.dim = (self.audio_length, 1)

# %%
def audio_norm(data):
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data-min_data)/(max_data-min_data+1e-6)
    return data-0.5

# %%
config = Config(sampling_rate=22050, n_classes = num_classes, audio_duration=5, n_folds=10, learning_rate=0.001)

def load_audio(id_list):
    cur_batch_size = len(id_list)
    X = np.empty((cur_batch_size, *config.dim))
    input_length = config.audio_length

    for i, ID in tqdm.tqdm(enumerate(id_list)):
        #print(f'{ID} started')
        data, rate = librosa.core.load(base_dir+ID, sr=22050,res_type='kaiser_fast')
        if len(data) > input_length:
            max_offset = len(data) - input_length
            offset = np.random.randint(max_offset)
            data = data[offset:(input_length+offset)]
        else:
            if input_length > len(data):
                max_offset = input_length - len(data)
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            data = np.pad(data, (offset, input_length - len(data) - offset), "constant")

        data = audio_norm(data)[:, np.newaxis]
        X[i,] = data
        #print(f'{ID} is ready')
    return(X)


# %% 
# Lables

def load_lables(id_list):
    cur_batch_size = len(id_list)
    y = np.empty(cur_batch_size, dtype=np.dtype("a16"))

    for i, ID in enumerate(id_list):
        lable = df_train[df_train['filename'] == ID]['ebird_code'].to_string(index= False)
        lable = lable.lstrip()
        y[i] = lable
        print(f'{ID} is ready, {lable}')
    
    return(y)

# %%
X_train = load_audio(train_idx)
X_test = load_audio(test_idx)

#%%
Y_train = load_lables(train_idx)
Y_test = load_lables(test_idx)


#%%
le = preprocessing.LabelEncoder()
le.fit(Y_train)
Y_train = le.transform(Y_train)
Y_test = le.transform(Y_test)

# %%
#X_train = tf.convert_to_tensor(X_train)
#X_test = tf.convert_to_tensor(X_test)
#Y_train = tf.convert_to_tensor(Y_train)
#Y_test = tf.convert_to_tensor(Y_test)

# %%
def model_train(config):
    nclass = config.n_classes
    input_length = config.audio_length

    inp = tf.keras.layers.Input(shape=(input_length, 1))
    x = tf.keras.layers.Conv1D(16, 9, activation="relu", padding="valid")(inp)
    x = tf.keras.layers.Conv1D(16, 9, activation="relu", padding="valid")(x)
    x = tf.keras.layers.MaxPool1D(16)(x)
    x = tf.keras.layers.Dropout(rate=0.1)(x)
    
    x = tf.keras.layers.Conv1D(32, 3, activation="relu", padding="valid")(x)
    x = tf.keras.layers.Conv1D(32, 3, activation="relu", padding="valid")(x)
    x = tf.keras.layers.MaxPool1D(4)(x)
    x = tf.keras.layers.Dropout(rate=0.1)(x)
    
    x = tf.keras.layers.Conv1D(32, 3, activation="relu", padding="valid")(x)
    x = tf.keras.layers.Conv1D(32, 3, activation="relu", padding="valid")(x)
    x = tf.keras.layers.MaxPool1D(4)(x)
    x = tf.keras.layers.Dropout(rate=0.1)(x)
    
    x = tf.keras.layers.Conv1D(256, 3, activation="relu", padding="valid")(x)
    x = tf.keras.layers.Conv1D(256, 3, activation="relu", padding="valid")(x)
    x = tf.keras.layers.GlobalMaxPool1D()(x)
    x = tf.keras.layers.Dropout(rate=0.2)(x)

    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dense(1028, activation="relu")(x)
    out = tf.keras.layers.Dense(nclass, activation="softmax")(x)

    model = tf.keras.Model(inputs=inp, outputs=out)
    opt = tf.keras.optimizers.Adam(learning_rate = config.learning_rate)

    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics = ['accuracy'])
    
    return model

# %%
model = model_train(config)

# %%
history = model.fit(X_train, Y_train, 
            batch_size = 16, 
            epochs = 30, 
            validation_data = (X_test, Y_test))

# %%
pred = model.predict(X_test)

# %%
preds_classes = np.argmax(pred, axis=-1)

# %%
confusion_matrix(Y_test, preds_classes)

# %%
f1_score(Y_test, preds_classes,average = "micro")

# %%
#################### Valid Data Set ##################
base_dir = "./small_valid/"
idx_valid = os.listdir("./small_valid")

X_valid = load_audio(idx_valid)
Y_valid = load_lables(idx_valid)

Y_valid = le.transform(Y_valid)

# %%
pred = model.predict(X_valid)
preds_classes = np.argmax(pred, axis=-1)

# %%
confusion_matrix(Y_valid, preds_classes)

# %%
f1_score(Y_valid, preds_classes,average = "micro")

# %%

################### Prepare Submit ###################

#PATH = '../input/birdsong-recognition/'
#TEST_FOLDER = '../input/birdsong-recognition/test_audio/'
PATH = "./"
TEST_FOLDER = "./long_test/"

test = pd.read_csv(os.path.join(PATH, 'test.csv'))

#submission = pd.read_csv(os.path.join(PATH, 'sample_submission.csv'))

def load_test_clip(path, start_time, duration=5):
    return librosa.load(path, offset=start_time, duration=duration)[0]

def transform_audio(data, input_length):
    if len(data) > input_length:
        max_offset = len(data) - input_length
        offset = np.random.randint(max_offset)
        data = data[offset:(input_length+offset)]
    else:
        if input_length > len(data):
            max_offset = input_length - len(data)
            offset = np.random.randint(max_offset)
        else:
            offset = 0

    data = np.pad(data, (offset, input_length - len(data) - offset), "constant")
    data = audio_norm(data)[:, np.newaxis]
    return(data)

# %%
preds = []
for index, row in test.iterrows():
    # Get test row information
    site = row['site']
    start_time = row['seconds'] - 5
    row_id = row['row_id']
    audio_id = row['audio_id']

    # Get the test sound clip
    if site == 'site_1' or site == 'site_2':
        test_sound = load_test_clip(TEST_FOLDER + audio_id + '.mp3', start_time, sr=22050)
    else:
        test_sound = load_test_clip(TEST_FOLDER + audio_id + '.mp3', 0, duration=None, sr=22050)
    
    test_sound = transform_audio(test_sound, input_length)
    # Make the prediction
    pred = model.predict(test_sound)
    pred = np.argmax(pred, axis=-1)
    # Store prediction
    preds.append([row_id, pred])

preds = pd.DataFrame(preds, columns=['row_id', 'birds'])
preds.to_csv('submission.csv', index=False)
