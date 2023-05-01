
import pandas as pd
import os
import tensorflow as tf
from keras import backend as K
import keras
import pickle
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from sys import platform # to get if we're on mac or windows, directories are diff

from data_processing import  *
from visualization import *
from models import  *

### these are for running a notification sound when training is completed
### optional
# import pygame
# from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
# from ctypes import cast, POINTER
# from comtypes import CLSCTX_ALL

#### GLOBALS ####

# set tensorflow logs. levels: 
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

### IMPORTANT: if you want to save a model, ensure the data processing and model training option lists are all length 1

# data visualization
VISUALIZE_RAW_DATA = False
VISUALIZE_WINDOWED_DATA = False

# data processing
LOAD_EXISTING_DATA = False
SAVE_PROCESSED_DATA = True
VERBOSE_DATAPROCESSING = True
EXIT_AFTER_DATAPROCESS = False
#PATIENT_IDS = [0, 1, 2, 3, 5, 6, 7, 8]
PATIENT_IDS = [1, 3, 9, 10]
# note that sets 4 and 9 don't have any freezing occurances and worsen model performance

# data processing params
# first split data into "partitions" - segments that will be shuffled and split into train/val/test
# window the data within the partitions only (no overlap) in order to prevent leakage between tran/val/test sets
# suggestion: window size is a multiple of stride, partition size is a multiple of both window size and stride
data_processing_options = {"window_size": [1000], "stride": [125], "partition_size": [10000]}
DATA_SHORTENINGS = {"window_size": "ws", "stride": "s", "partition_size": "ps"}

### TODO: LEGACY, need to test
VALIDATE_EXISTING_MODEL = False # validates model then exits
LOAD_EXISTING_MODEL = False
### 

# model training
SAVE_TRAINED_MODEL = False # results are saved to model history regardless, just for saving model files
PRINT_MODEL_SUMMARIES = False # prints keras model summeries (model architecture)
PLOT_ROC_ENSEMBLE = False
VERBOSE_TRAINING = 0 # 0 = no message, 1 = progress bar, 2 = line per epoch
MODEL_TRAIN_TYPE = 2 # 0 = k fold cross validation, 1 = standard train/val split, 2 = 3 model ensemble with k fold cross
REPETITIONS = 2 # How many times to iterate training/saving model to get average result

# for non k fold training:
train_fraction = 0.7 # use 70% data for train, 20% for val, 10% for test
val_fraction = 0.2

# training params
epochs = 10
model_fn = CNN_model_2D_simple # from model file. note model input dims need to match window size
ensemble_model_fn = CNN_model_ensemble_expanded_more_dropout # CNN_model_ensemble_expanded_more_dropout # TODO: dynamically load models so many options can be tested at once
MODEL_NAME_BASE = model_fn.__name__ + "_e:{}_".format(epochs) + datetime.now().strftime("%m/%d/%Y")
ENSEMBLE_MODEL_NAME_BASE = ensemble_model_fn.__name__ + "_e:{}_".format(epochs) + datetime.now().strftime("%m/%d/%Y")
ensemble_dim_indices = [[0, 1, 2], [3, 4, 5], [6, 7, 8]] # splits data into 1 sensor per model
# ensemble_dim_indices = [[0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2, 3, 4, 5, 6, 7, 8]] # all data for each ensemble model
loss = tf.keras.losses.BinaryCrossentropy()
training_options = {"batch_size": [1000], "learning_rates": [0.01], "callback_patiences": [50]} # we generate all permutations of the trining params to test
TRAIN_SHORTENINGS = {"batch_size": "bs", "learning_rates": "lr", "callback_patiences": "c"}
FOLD_NUMBER = 5 # for K fold cross validation

# if true, plays an audio message from notification file (mp3)
TRAINING_COMPLETED_AUDIO = False
NOTIFICATION_FILE = ""

# Pathing and directories
if platform == "win32": # if windows
    raw_data_path = os.getcwd() +'\dataset_fog_release\dataset\\'
    data_save_path = os.getcwd() +'\saved_data\\'
    dataset_base_name = 'dataset_partitioned' # datasets are saved with file names that also include the processing options
    test_dataset_base_name = 'test_dataset_partitioned'
else: # assume mac or linux
    #raw_data_path = os.getcwd() +'/dataset_fog_release/dataset/'
    raw_data_path = '/Users/wangyining/Desktop/Research/Xuanwu/Code/Filtered Data/Test'
    #data_save_path = os.getcwd() +'/saved_data'
    data_save_path = '/Users/wangyining/Desktop/Research/Xuanwu/Code/Filtered Data/saved_data'
    dataset_base_name = 'dataset_partitioned' # datasets are saved with file names that also include the processing options
    test_dataset_base_name = 'test_dataset_partitioned'

#### HELPERS ####

def generate_permutations(entry_dict):
    # given dict of lists, generate a list of dicts for each combination of entries in lists
    # note: does a shallow copy so nested dicts probably wont work

    output_list = [{}]
    topics = list(entry_dict)
    for topic in topics:
        temp_list = []
        values = entry_dict[topic]
        for value in values:
            for permutation in output_list:
                expanded_permutation = permutation.copy()
                expanded_permutation[topic] = value
                temp_list += [expanded_permutation]
        output_list = temp_list
    return output_list

def sensitivity(tp, fn):
    return tp / (tp + fn)

def specificity(tn, fp):
    return tn / (tn + fp)

def specificity_factory(): # used to show specificity as keras trains
    def fn(y_true, y_pred):
        tn = tf.reduce_sum(tf.cast(tf.logical_and((y_pred < 0.5),tf.math.equal(y_true, 0)), tf.float32))
        fp = tf.reduce_sum(tf.cast(tf.logical_and((y_pred >= 0.5),tf.math.equal(y_true, 0)), tf.float32))
        return tn / (tn + fp)
    
    fn.__name__ = 'specificity'
    return fn

def f1_score(tp, fp, tn, fn):
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)

    f1 =  2*(precision * recall)/(precision + recall)
    return f1

#### DATA PROCESSING ####

def plot_raw_sensor_vals(sensor_pos, sensor_axis, datapath):
    for file_name in os.listdir(datapath):
        vis_path = os.path.join(datapath, file_name)
        print("plotting: {}".format(file_name))
        x_plot_file(vis_path, grid=False, sensor_pos=sensor_pos, sensor_axis=sensor_axis)

def load_and_window_rawdata(pids_to_load, raw_data_path, window_size, stride):
    # given a list of patient IDs and a path to the folder with patient data
    # loads in and segments all data, concats data together
    # returns 2 arrays, both formatted: 0 - windows, 1 - labels (fog = 1 non fog = 0 unrelated = -1 ), 2 - original label for each window entry
    # 1st array is data put through low pass and other filters. Second array does not filter data

    filtered_data = [None, None]
    unfiltered_data = [None, None]
    #raw_data_path = '/Users/wangyining/Desktop/Research/Xuanwu/Code/Filtered Data/Test'
    print(os.listdir(raw_data_path))
    for file_name in os.listdir(raw_data_path):
        print(file_name)

        # setup pid and open file for data
        file = os.path.join(raw_data_path, file_name)
        print(raw_data_path)
        if ".DS" in file_name:
            continue
        pid = int(file_name[1:3].lstrip("0")) - 1 # Offset patient id to be 0-indexed

        if pid in pids_to_load:
            raw_data = pd.read_csv(file, delim_whitespace=True, header=None)

            # setup dataprocess object. get filtered and unfiltered data
            data = DataProcess(raw_data,pid)  # note all the data is preprocessed - can we do the same to data on the embedded system?
            data_unfilt = DataProcess(raw_data, pid, filter=False)
            # perhaps consider what ops we can do quickly on embedded and only preprocess those

            windows_fog_nonfog, windows_targets, freqs = data.get_segments(sequence_length = window_size, stride=stride, verbose = VERBOSE_DATAPROCESSING)
            windows_fog_nonfog_unfilt, windows_targets_unfilt, freqs = data_unfilt.get_segments(sequence_length = window_size, stride=stride, verbose = VERBOSE_DATAPROCESSING)
           
            # list formatting:
            # windows_fog_nonfog: [windows with non fog, windows with fog]
            # windows_targets: [all windows, array of targets for each window (0 = no fog, 1 = fog), array of targets for each element in each window]

            if VISUALIZE_WINDOWED_DATA:
                print("visualizing processed data")
                # plot_windows([data.input, data.target], 0)
                print(raw_data.to_numpy().shape)
                print(windows_targets[0].shape)
                print(data.input.shape)
                for i in range(windows_targets[0].shape[2]):
                    print('visualizing column {}'.format(i))
                    # plot_windows(raw_data.to_numpy()[:,1:-1], i)
                    # plot_windows(data.input, i)
                    # plot_fog_nonfog([data.input, data.target], i)
                    visualize_windowed_data(windows_targets, i, sequence_length = window_size, stride = stride)
                    visualize_windowed_data(windows_targets_unfilt, i, sequence_length = window_size, stride = stride)

            # concat all data
            if filtered_data[0] is not None:
                filtered_data[0] = np.concatenate((filtered_data[0], windows_targets[0]), axis=0)
                filtered_data[1] = np.append(filtered_data[1], windows_targets[1])
                filtered_data[2] = np.concatenate((filtered_data[2], windows_targets[2]), axis=0)
            else:
                filtered_data = windows_targets

            if unfiltered_data[0] is not None:
                unfiltered_data[0] = np.concatenate((unfiltered_data[0], windows_targets_unfilt[0]), axis=0)
                unfiltered_data[1] = np.append(unfiltered_data[1], windows_targets_unfilt[1])
                unfiltered_data[2] = np.concatenate((unfiltered_data[2], windows_targets_unfilt[2]), axis=0)
            else:
                unfiltered_data = windows_targets_unfilt

            if VERBOSE_DATAPROCESSING:
                print("processed file: {}, for patient: {}".format(file_name, pid))
    return filtered_data, unfiltered_data

def shuffle_partition_dataset(dataset, train_fraction, val_fraction, partition_size, window_size, stride):
    #dataset is list of np arrays, with entries: windows, window label, element labels, indices

    # first, we need to ensure the windows shuffled into train/val don't overlap. This would cause bleed
    # we partition the data according to partition size. 
    # we dont keep any windows that contain partition edge points (except if it is at the start of the window)

    # preallocate the dset size for the bulk of the data
    # this results in 1 incomplete frame at the end being left. We just stick that in training too
    frames_per_partition = int((partition_size-window_size)/stride + 1)
    n_frames = len(dataset[0])
    n_points = (n_frames-1)*(stride) + window_size # can verity with dataset[3][-1], which is the indices of the last frame
    n_partitions = int(n_points/partition_size)

    # default -1000, which should never be seen, allows us to check for missed data
    partitioned_windows = np.zeros((n_partitions, frames_per_partition, window_size, 9)) -1000
    partitioned_labels = np.zeros((n_partitions, frames_per_partition)) -1000
    partitioned_indices = np.zeros((n_partitions, frames_per_partition, window_size, 1)) -1000

    # allocate 1 extra frame for spillover (since frames wont evently split into partitions usually)
    spillover_windows = np.zeros((frames_per_partition, window_size, 9)) 
    spillover_labels = np.zeros((frames_per_partition)) 
    spillover_indices = np.zeros((frames_per_partition, window_size, 1)) 
    spillover_counter = 0

    # to verify that data is not being double copied to the same place
    verification_dict = {}

    for i in range(0, n_frames): # for ith frame
        frame_i = dataset[0][i]
        labels_i = dataset[1][i]
        indices_i = np.reshape(dataset[3][i], (window_size, 1))

        starting_index_frame_i = indices_i[0]
        current_partition = int(starting_index_frame_i/partition_size) # gets which partition currently on, zero indexed

        index_in_partition = starting_index_frame_i - current_partition * partition_size
        frame_in_partition = int(index_in_partition / stride) # position of current frame in current partition

        # by predetermining the number of frames in each partition
        # we ensure there is no overlap between partitions

        verification_string = str(current_partition) + "." + str(frame_in_partition)
        if verification_string in verification_dict: # we've attempted to allocate to this frame before, throw a warning
            print("WARN: writing multiple frames to same place in data partitioning")
            print(verification_string)
        else:
            verification_dict[verification_string] = verification_string

        if current_partition >= n_partitions:
            spillover_windows[spillover_counter, :, :] = frame_i
            spillover_labels[spillover_counter] = labels_i
            spillover_indices[spillover_counter, :, :] = indices_i
            spillover_counter += 1

        elif frame_in_partition < frames_per_partition: # this excludes frames that intersect the boundary between partitions
            partitioned_windows[current_partition, frame_in_partition, :,:] = frame_i
            partitioned_labels[current_partition, frame_in_partition] = labels_i
            partitioned_indices[current_partition, frame_in_partition, :,:] = indices_i

    if np.count_nonzero(partitioned_windows == -1000): # -1000 is used as a default that should not be seen in practise
        print("WARN: missing data in windows after partition")
    if np.count_nonzero(partitioned_labels == -1000):
        print("WARN: missing data in labels after partition")
    if np.count_nonzero(np.count_nonzero(partitioned_indices == -1000)):
        print("WARN: missing data in indices after partition")

    # shuffle the partitions of the dataset, now with overlap removed
    num_partitions = partitioned_windows.shape[0]
    p1 = np.random.RandomState(seed=42).permutation(num_partitions)
    partitioned_dataset = [partitioned_windows, partitioned_labels, partitioned_indices]
    spillover_data = [spillover_windows[0:spillover_counter], spillover_labels[0:spillover_counter], spillover_indices[0:spillover_counter]]

    ### debug visualization setup
    # merged_partitioned_windows = np.reshape(partitioned_windows, (-1,window_size,9))
    # merged_partitioned_labels = np.reshape(partitioned_labels, (-1))
    # merged_partitioned_indices = np.reshape(partitioned_indices, (-1,window_size,1))
    ### visualizes initial data
    # visualize_indexed_windowed_data((dataset[0], dataset[1], dataset[3]), 1)
    ### visualizes the data after partitioning and shuffling, the first reconstructs with indices (should look in init data)
    ### the second is without indices, should be shuffled
    # visualize_indexed_windowed_data((merged_partitioned_windows, merged_partitioned_labels, merged_partitioned_indices), 1)
    # visualize_windowed_data((merged_partitioned_windows, merged_partitioned_labels), 1)

    val_cutoff = int(num_partitions * (train_fraction))
    test_cutoff = int(num_partitions * (train_fraction + val_fraction))

    train_dataset = []
    val_dataset = []
    test_dataset = []

    shape_tuples = [(-1, window_size, 9), (-1), (-1, window_size, 1)] # used to reshape windows, labels, and indices

    for i in range(0, len(partitioned_dataset)): # shuffle all partitions to split into train,val,test randomly
        partitioned_dataset[i] = partitioned_dataset[i][p1]
        train_dataset += [np.reshape(partitioned_dataset[i][0:val_cutoff], shape_tuples[i])]
        val_dataset += [np.reshape(partitioned_dataset[i][val_cutoff:test_cutoff], shape_tuples[i])]
        test_dataset += [np.reshape(partitioned_dataset[i][test_cutoff:], shape_tuples[i])]

    # shuffle training data and add the spillover data to training
    # we dont shuffle the other sets since it shouldn't matter
    p2 = np.random.RandomState(seed=42).permutation(len(train_dataset[0]) + spillover_counter - 1)
    for i in range(0, len(train_dataset)):
        train_dataset[i] = np.concatenate((train_dataset[i], spillover_data[i]), axis = 0)
        train_dataset[i] = train_dataset[i][p2]

    ### debug to confirm shuffled training data reconstructs properly
    # visualize_indexed_windowed_data(train_dataset, 1)
    return train_dataset, val_dataset, test_dataset

#### TRAINING ####

def fit_model(windows_train, targets_train, windows_val, targets_val, train_permutation, model_function):
    # runs keras model.fit on the data
    print(windows_train.shape)
    print(targets_train.shape)

    l = train_permutation["learning_rates"]
    c = train_permutation["callback_patiences"]
    batch_size = train_permutation["batch_size"]

    optimizer = tf.keras.optimizers.Adam(learning_rate=l)
    print(f"\nTrain on learning rate={l} and call patience={c}\n")

    val_windows_ds = tf.data.Dataset.from_tensor_slices(windows_val *1000) # converts g to milli g, needed for embedded app
    val_targets_ds = tf.data.Dataset.from_tensor_slices(targets_val)
    val_set_target = tf.data.Dataset.zip((val_windows_ds, val_targets_ds))
    val_set_target = val_set_target.batch(batch_size)

    train_windows_ds = tf.data.Dataset.from_tensor_slices(windows_train *1000)
    train_targets_ds = tf.data.Dataset.from_tensor_slices(targets_train)
    train_set_target = tf.data.Dataset.zip((train_windows_ds, train_targets_ds))
    train_set_target = train_set_target.batch(batch_size)

    weights = [np.count_nonzero(targets_train == 0),
               np.count_nonzero(targets_train == 1)]  # non FOG, FOG frequencies
    print(targets_train.shape)
    print(weights)
    #weights = [528,1]
    class_weights = {0: 1 / weights[0] * (weights[1] + weights[0]) / 2, # this is done as per keras class weights example
                     1: 1 / weights[1] * (weights[1] + weights[0]) / 2}
    #print(class_weights)
    model = model_function(output_bias=np.log([weights[1] / weights[0]]))  # initialize bias as ln(true/false)

    # model.summary()
    specificity_metric = specificity_factory()

    model.compile(loss=loss, optimizer=optimizer,
                  metrics=[keras.metrics.TruePositives(name='tp'),
                           keras.metrics.FalsePositives(name='fp'),
                           keras.metrics.TrueNegatives(name='tn'),
                           keras.metrics.FalseNegatives(name='fn'),
                           keras.metrics.BinaryAccuracy(name='accuracy'),
                           keras.metrics.AUC(name='auc'),
                           keras.metrics.Recall(name='recall'),  # recall is same as sensitivity
                           specificity_metric,
                           ])

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_auc', min_delta=0.0001, patience=c,
                                                mode='max', verbose=1, restore_best_weights=1)
    #print(train_set_target)

    history = model.fit(train_set_target, epochs=epochs, validation_data=val_set_target,
                        class_weight=class_weights, verbose=VERBOSE_TRAINING, callbacks=[callback])

    if len(history.history['val_tp']) == epochs:
        stopping_index = -1
    else:
        stopping_index = len(history.history['val_tp']) - c - 1

    auc = history.history['val_auc'][stopping_index]
    tp = history.history['val_tp'][stopping_index]
    tn = history.history['val_tn'][stopping_index]
    fp = history.history['val_fp'][stopping_index]
    fn = history.history['val_fn'][stopping_index]

    return model, history, tp, tn, fp, fn, auc

def k_fold_cross(windows_train, targets_train, train_permutation, model_function, ensemble = False):
    # k fold cross validation with fold_number folds. Splits data here
    # accumulates results from each fold test

    fold_number = FOLD_NUMBER

    folded_windows_train = np.array_split(windows_train, fold_number)
    folded_targets_train = np.array_split(targets_train, fold_number)

    cumulative_results = {
        "tp": [],
        "tn": [],
        "fp": [],
        "fn": [],
        "auc": []
    }

    for i in range(0, len(folded_windows_train)):
        validation_window_fold = folded_windows_train[i]
        validation_target_fold = folded_targets_train[i]

        if i < len(folded_windows_train):
            training_window_folds = folded_windows_train[0:i] + folded_windows_train[
                                                                i + 1: len(folded_windows_train)]
            training_target_folds = folded_targets_train[0:i] + folded_targets_train[
                                                                i + 1: len(folded_windows_train)]
        else:
            training_window_folds = folded_windows_train[0:i]
            training_target_folds = folded_targets_train[0:i]

        training_window_folds = np.concatenate(training_window_folds)
        training_target_folds = np.concatenate(training_target_folds)

        if ensemble:
            tp, tn, fp, fn, auc, sense, spec = train_ensemble([training_window_folds, training_target_folds], [validation_window_fold, validation_target_fold], train_permutation)

        else:
            model, history, tp, tn, fp, fn, auc = fit_model(training_window_folds, training_target_folds, validation_window_fold, validation_target_fold, train_permutation, model_function)

        cumulative_results["tp"].append(tp)
        cumulative_results["tn"].append(tn)
        cumulative_results["fp"].append(fp)
        cumulative_results["fn"].append(fn)
        cumulative_results["auc"].append(auc)

    return np.sum(cumulative_results["tp"]), np.sum(cumulative_results["tn"]), np.sum(cumulative_results["fp"]), np.sum(cumulative_results["fn"]), np.average(cumulative_results["auc"])

def train_ensemble(train_dataset, val_dataset, train_permutation):
    histories = []
    models = []
    val_subsets = []
    for i in range(0, len(ensemble_dim_indices)):
        train_sensor_i = train_dataset[0][:,:,ensemble_dim_indices[i]]
        val_sensor_i = val_dataset[0][:,:,ensemble_dim_indices[i]]
        val_subsets += [val_sensor_i]
        #print("This is train dataset's shape:",len(train_dataset[0]))
        #print("This is train dataset's shape:",len(train_dataset[1]))
        model, history, tp, tn, fp, fn, auc_submodel = fit_model(train_sensor_i, train_dataset[1], val_sensor_i,
                                                val_dataset[1], train_permutation, ensemble_model_fn)
        
        # validate_AUC_function(model, val_sensor_i, val_dataset[1]) # validated, should be confirmed

        print("submodel sense: {}, spec: {}, auc: {}".format(sensitivity(tp, fn), specificity(tn, fp), auc_submodel))
        models += [model]
        histories += [history]
    tp, tn, fp, fn, auc, sense, spec = validate_AUC_ensemble(models, val_subsets, val_dataset[1], num_models = len(ensemble_dim_indices))

    print("validation ensemble performance: {}, {}, {}, {}, auc: {} sense: {}, spec: {}".format(tp, tn, fp, fn, auc, sense, spec))
    return tp, tn, fp, fn, auc, sense, spec

#### RESULTS SAVING ####

def format_permutation_string(permutation_dict, short_form_dict):
    perm_string = "_"
    for key, item in permutation_dict.items():
        perm_string += short_form_dict[key] + ":" + str(item) + "_"
    return perm_string

def save_model(history, model, train_permutation, data_permutation, custom_model_name = None):
    if custom_model_name is None:

        train_string = format_permutation_string(train_permutation, TRAIN_SHORTENINGS)
        data_string = format_permutation_string(data_permutation, DATA_SHORTENINGS)
        model_name = f"{train_string}{data_string}" + MODEL_NAME_BASE
    else:
        model_name = custom_model_name
        print("saving model w name: {}".format(model_name))

    if platform == "win32":
        plt_save_path = "figure\plt_" + model_name
        model_save_path = "models\\" + model_name
    else:
        plt_save_path = "figure/plt_" + model_name
        model_save_path = "models/" + model_name

    save_plot_loss(history, "cnn training", 'b', 'r', plt_save_path)
    
    model.save(model_save_path)

def save_results(tp, tn, fp, fn, auc, train_permutation, data_permutation, history = None, custom_name_base = None, custom_model_name = None):
    # records results in file model history
    name_base = MODEL_NAME_BASE
    if custom_name_base:
        name_base = custom_name_base

    if custom_model_name is None:
        train_string = format_permutation_string(train_permutation, TRAIN_SHORTENINGS)
        data_string = format_permutation_string(data_permutation, DATA_SHORTENINGS)
        model_name = f"{train_string}{data_string}" + name_base
    else:
        model_name = custom_model_name

    hist = pd.read_csv('model_history.csv').to_dict('list')

    hist["model_name"].append(model_name)
    if history is None:
        hist["epochs"].append("--")
    else:
        hist["epochs"].append(len(history.history['val_loss']))
    hist["sensitivity"].append(round(sensitivity(tp, fn), 3))
    hist["specificity"].append(round(specificity(tn, fp), 3))
    hist["tp"].append(tp)
    hist["tn"].append(tn)
    hist["fp"].append(fp)
    hist["fn"].append(fn)
    hist["acc"].append(round((tp + tn) / (tp + tn + fp + fn), 3))
    hist["auc"].append(round(auc, 3))
    df = pd.DataFrame(hist)
    df.to_csv('model_history.csv', index=False)

    keras.backend.clear_session()

#### VALIDATION ####

def validate_ensemble_model(model_predictions, val_y, num_models, thresh = 0.5, verbose = False):
    # model predictions is a list of np arrays that are the prediction outputs from each of the ensemble models
    # manually predicts with ensemble model and does a hard vote for final predictions
    # voting threshold is number of models / 2 rounded up (+1 if even)
    cumu_predictions = None
    val_y_reshaped = np.reshape(val_y, (val_y.shape[0], 1))

    for i, predictions in enumerate(model_predictions):
        bin_pred = (predictions > thresh).astype(int)
        incorrect = np.sum(np.abs(bin_pred - val_y_reshaped))

        if verbose: 
            print("incorrect number for model {}: {}".format(i, incorrect))

        if cumu_predictions is None:
            cumu_predictions = bin_pred
        else:
            cumu_predictions = np.add(cumu_predictions, bin_pred).astype(int)

        if verbose:
            print("predictions of this model (# 0, 1): {}".format(np.bincount(bin_pred[:, 0])))
            print("summed predictions so far: {}".format(np.bincount(cumu_predictions[:, 0])))

            print(cumu_predictions.shape)

    voting_thresh = int(num_models/2) + 1
    ensemble_pred = (cumu_predictions >= voting_thresh).astype(int)

    if verbose: 
        print("voting thresh: {}".format(voting_thresh))
        print("total predictions (# 0, 1): ".format(np.bincount(ensemble_pred[:,0])))

        # fpr_keras, tpr_keras, thresholds_keras = roc_curve(val_y_reshaped, ensemble_pred)
        # auc_keras = auc(fpr_keras, tpr_keras)

        print("--------------")
        # print("validation auc: {}".format(auc_keras))

    tp = np.sum(np.logical_and(ensemble_pred == 1, val_y_reshaped == 1))
    tn = np.sum(np.logical_and(ensemble_pred == 0, val_y_reshaped == 0))
    fp = np.sum(np.logical_and(ensemble_pred == 1, val_y_reshaped == 0))
    fn = np.sum(np.logical_and(ensemble_pred == 0, val_y_reshaped == 1))

    return tp, tn, fp, fn, sensitivity(tp, fn), specificity(tn, fp)

def get_AUC(xs, ys):
    print(ys[0])
    print(xs[0])
    area = (1 + ys[0])*xs[0]/2 # initial sliver of area before 1st point, if it exists
    for i in range(0, len(xs)-1): # trapezoid approximation
        dx = xs[i+1] - xs[i]
        if dx > 0:
            area += (ys[i] + ys[i+1])*dx/2
    print(ys[-1])
    print(xs[-1])
    area += (ys[-1])*(1-xs[-1])/2# last sliver of area
    return area  

def validate_AUC_function(model, val_x, val_y, step_size = 0.005):
    nsteps = int(1/step_size)
    # compare our AUC to the keras one to make sure it works
    predictions = model.predict(val_x * 1000)
    val_y_reshaped = np.reshape(val_y, (val_y.shape[0], 1))
    sensitivities = []
    specificities = []

    for i in range(0, nsteps):
        thresh = i/nsteps

        bin_pred = (predictions > thresh).astype(int)
        tp = np.sum(np.logical_and(bin_pred == 1, val_y_reshaped == 1))
        tn = np.sum(np.logical_and(bin_pred == 0, val_y_reshaped == 0))
        fp = np.sum(np.logical_and(bin_pred == 1, val_y_reshaped == 0))
        fn = np.sum(np.logical_and(bin_pred == 0, val_y_reshaped == 1))

        sensitivities += [sensitivity(tp, fn)]
        specificities += [specificity(tn, fp)]

    print("manual calc AUC")
    print(get_AUC(specificities, sensitivities))
    # plot ROC curve
    plt.plot(sensitivities, specificities)
    plt.xlabel("sensitivity")
    plt.ylabel("specificity")
    plt.show()

def validate_AUC_ensemble(models, val_xs, val_y, num_models, step_size = 0.005):
    nsteps = int(1/step_size)
    sensitivities = []
    specificities = []

    predictions = []
    for i, model in enumerate(models):
        predictions += [model.predict(val_xs[i] * 1000)]

    for i in range(0, nsteps):
        thresh = i/nsteps
        tp, tn, fp, fn, sense, spec = validate_ensemble_model(predictions, val_y, num_models, thresh)
        sensitivities += [sense]
        specificities += [spec]

    auc = get_AUC(specificities, sensitivities)
    tp, tn, fp, fn, sense, spec = validate_ensemble_model(predictions, val_y, num_models, thresh = 0.5)

    if PLOT_ROC_ENSEMBLE:   
        plt.plot(sensitivities, specificities)
        plt.xlabel("sensitivity")
        plt.ylabel("specificity")
        plt.show()

    return tp, tn, fp, fn, auc, sense, spec

#### MAIN CODE ####

def main():

    pids_to_load = PATIENT_IDS
    data_permutations = generate_permutations(data_processing_options)
    training_permutations = generate_permutations(training_options)

    for data_permutation in data_permutations:
        window_size = data_permutation["window_size"]
        stride = data_permutation['stride']
        partition_size = data_permutation['partition_size']

        if platform == "win32":
            dataset_path = data_save_path + '\{}_{}_{}_{}.pkl'.format(dataset_base_name, window_size, stride, partition_size)
            test_dataset_path = data_save_path + '\{}_{}_{}_{}.pkl'.format(test_dataset_base_name, window_size, stride, partition_size)
        else:
            dataset_path = data_save_path + '/{}_{}_{}_{}.pkl'.format(dataset_base_name, window_size, stride, partition_size)
            test_dataset_path = data_save_path + '/{}_{}_{}_{}.pkl'.format(test_dataset_base_name, window_size, stride, partition_size)

        print("processing data with window size: {}, stride {}, partition size: {}".format(window_size, stride, partition_size))

        if VISUALIZE_RAW_DATA: # plots raw sensor readings
            plot_raw_sensor_vals(sensor_pos=0, sensor_axis=1, datapath=raw_data_path)

        if LOAD_EXISTING_DATA:
            with open(dataset_path, 'rb') as f:
                partitioned_dataset = pickle.load(f)
                train_dataset, val_dataset, test_dataset, train_unfilt, val_unfilt, test_unfilt = partitioned_dataset
        else:
            processed_dataset, unfiltered_dataset = load_and_window_rawdata(pids_to_load, raw_data_path, window_size, stride)

            # assign an array of L * window size * 1 indices for every point in every window, so they can be reconstructed after shuffle
            indices = np.zeros((processed_dataset[0].shape[0], window_size))
            for i in range(0, (processed_dataset[0].shape[0])):
                indices[i] = np.linspace(i*stride, i*stride+window_size-1, num=window_size)
                
            processed_dataset += [indices]
            unfiltered_dataset += [indices]

            #visualize_windowed_data([processed_dataset[0], processed_dataset[1]], 0)
            #visualize_indexed_windowed_data([processed_dataset[0], processed_dataset[1], processed_dataset[3]], 0)

            # visualize_windowed_data(processed_dataset, 0)
            #data from all patients is concat together at this point

            # shuffle data in reproducable manner and cut into train/val/test
            train_dataset, val_dataset, test_dataset = shuffle_partition_dataset(processed_dataset, train_fraction, val_fraction, partition_size, window_size, stride)
            train_unfilt, val_unfilt, test_unfilt = shuffle_partition_dataset(unfiltered_dataset, train_fraction, val_fraction, partition_size, window_size, stride)

            partitioned_dataset = [train_dataset, val_dataset, test_dataset, train_unfilt, val_unfilt, test_unfilt]

            # save windowed data
            if SAVE_PROCESSED_DATA:
                with open(dataset_path, 'wb') as f:
                    pickle.dump(partitioned_dataset, f)

        # save test batch separately as well
        if SAVE_PROCESSED_DATA:
            with open(test_dataset_path, 'wb') as f:
                pickle.dump({"filtered windows":test_dataset[0]
                            , "unfiltered windows": test_unfilt[0], "targets": test_dataset[1]}, f)
                
        if not EXIT_AFTER_DATAPROCESS:
            ### Training ###
            for k in range(0, REPETITIONS): # repeat training multiple times so results can be averaged
                for train_permutation in training_permutations:
                    l = train_permutation["learning_rates"]
                    c = train_permutation['callback_patiences']
                    batch_size = train_permutation['batch_size']

                    if MODEL_TRAIN_TYPE == 0: # k fold cross validation
                        train_windows = np.concatenate((train_dataset[0], val_dataset[0]), axis=0)
                        train_targets = np.concatenate((train_dataset[1], val_dataset[1]), axis=0)
                        tp, tn, fp, fn, auc = k_fold_cross(train_windows, train_targets, train_permutation, model_fn)
                        
                        if PRINT_MODEL_SUMMARIES:
                            model_fn().summary()
                        save_results(tp, tn, fp, fn, auc, train_permutation, data_permutation)

                    elif MODEL_TRAIN_TYPE == 1: # standard model training (can save model)
                        # visualize_windowed_data([np.concatenate((train_dataset[0],val_dataset[0])), np.concatenate((train_dataset[1],val_dataset[1]))], 0)
                        #visualize_indexed_windowed_data([np.concatenate((train_dataset[0], val_dataset[0])),np.concatenate((train_dataset[1], val_dataset[1])),np.concatenate((train_dataset[3], val_dataset[3]))], 0)

                        model, history, tp, tn, fp, fn, auc = fit_model(train_dataset[0], train_dataset[1], val_dataset[0], val_dataset[1], train_permutation, model_fn)
                        
                        if PRINT_MODEL_SUMMARIES:
                            model_fn().summary()
                        save_results(tp, tn, fp, fn, auc, train_permutation, data_permutation)
                        if SAVE_TRAINED_MODEL:
                            save_model(history, model, train_permutation, data_permutation)

                    elif MODEL_TRAIN_TYPE == 2: # ensemble training with k fold cross validation
                        models = []
                        histories = []
                        val_subsets = []
                        train_windows = np.concatenate((train_dataset[0], val_dataset[0]), axis=0)
                        train_targets = np.concatenate((train_dataset[1], val_dataset[1]), axis=0)
                        
                        tp, tn, fp, fn, auc = k_fold_cross(train_windows, train_targets, train_permutation, model_fn, ensemble=True)

                        sense = sensitivity(tp, fn)
                        spec = specificity(tn, fp)

                        print("k fold ensemble summary: {}, {}, {}, {}, auc: {} sense: {}, spec: {}".format(tp, tn, fp, fn, auc, sense, spec))
                        
                        if PRINT_MODEL_SUMMARIES:
                            ensemble_model_fn().summary()
                        save_results(tp, tn, fp, fn, auc, train_permutation, data_permutation, custom_name_base = ENSEMBLE_MODEL_NAME_BASE)

                        # if SAVE_TRAINED_MODEL: # this cant be done currently for k fold cross
                        #     for i, model in enumerate(models):
                        #         # TODO: fix naming for below
                        #         save_model(histories[i], model, train_permutation, data_permutation, custom_model_name = "\ensembles\\" + f"base_train_{l}_{c}_" + ENSEMBLE_MODEL_NAME_BASE + "_submodel_{}".format(i))

    ##### if the notification libraries were imported you can try this: 
    # if TRAINING_COMPLETED_AUDIO:
    #     try:
    #         # notify when training done
    #         # Get default audio device using PyCAW
    #         devices = AudioUtilities.GetSpeakers()
    #         interface = devices.Activate(
    #             IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    #         volume = cast(interface, POINTER(IAudioEndpointVolume))

    #         volume.SetMasterVolumeLevel(-4, None)

    #         pygame.init()
    #         pygame.mixer.init()
    #         sound = pygame.mixer.Sound(NOTIFICATION_FILE)
    #         sound.set_volume(0.2) 
    #         sound.play()   
    #         while pygame.mixer.get_busy():
    #             pygame.time.delay(100)
    #     except Exception as e:
    #         print("audio failed with exception:")
    #         print(e)

if __name__ == "__main__":
    main()