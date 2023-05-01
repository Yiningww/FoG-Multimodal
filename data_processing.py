from collections import Counter
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
import tensorflow as tf

from scipy.signal import butter, lfilter, medfilt
from visualization import visualize_windowed_data, plot_raw, double_plot


#########################
# Deal with discontinuities with data later (split into separate large segments to evaluate)
#########################

class DataProcess():
    ''' Uses previous n sequence length acceleration data to predict next time step target (normal, stop)
    Try example 3 from following link for current timestamp --> target timestep prediction
    https://www.tensorflow.org/api_docs/python/tf/keras/utils/timeseries_dataset_from_array
    '''

    def __init__(self, data, patient_id, filter = True):
        self.timestep = data.iloc[:, 0]
        self.input = self.data_process(np.array(data.iloc[:, 1:10]), filter) #preprocesses data with butterworth filter
        self.target = np.array(data.iloc[:, 10]) # 0: unrelated; 1: non-freezing; 2: freezing # TODO: replace all 0 labels with 3 so the max function does what we want
        self.length = len(self.target)
        self.patient_id = patient_id

    def plot_data(self, grid=False, sensor_pos=0, sensor_axis=0):
        pcol = ['w','g','r']
        yltext=['X','Y','Z']
        ttext=['sensor ankle','sensor knee', 'sensor hip']
        y_data = self.input
        x_data = self.timestep

        # print(y_data.shape)
        if grid:
            fig=plt.figure(figsize=(20,16))
            for sensorpos in range(3):
                for sensoraxis in range(3):
                    ax = fig.add_subplot(3, 3, 1+sensorpos + 3*sensoraxis)
                    ax.plot(x_data/1000, y_data[:,sensoraxis+3*sensorpos], linewidth=0.2, zorder=2)

                ax.set_xlim([0, x_data.iloc[-1]/1000])
                ax.set_xlabel('time [s]')
                ax.set_ylabel('Acc {}[scaled]'.format(yltext[sensoraxis]))
                ax.set_title(ttext[sensorpos])
            plt.draw()
            plt.pause(0.001)
            input("enter to continue")
        else:
            plt.figure()
            plt.plot(x_data/1000, y_data[:,sensor_axis+3*sensor_pos], linewidth=0.2, zorder=2)
            plt.xlabel('time [s]')
            plt.ylabel('Acc {}[mg]'.format(yltext[sensor_axis]))
            plt.draw()
            plt.pause(0.001)
            input("enter to continue")
        return 0

    def data_process(self, data, filter = True):
        raw_cutoff = np.clip (data, -5000, 5000) #impose saturation for acceleration to help with scaling later
        # print("cutoff")
        # for i in range(0, 4):
        #     # double_plot(data, raw_cutoff, i)
        #     plot_windows(raw_cutoff, i)

        if filter:
            med = medfilt(raw_cutoff, kernel_size=(3, 1)) #first median filter data to denoise
                
            butter = None
            for i in range(med.shape[-1]):
                butter_channel = self.butter_lowpass_filter(med[:,i], 20, 5)
                if butter is None:
                    butter = butter_channel
                else:
                    butter = np.column_stack((butter, butter_channel))
            #butter = self.butter_lowpass_filter(med, 20, 5) #lowpass filter of order 5, 20hz cutoff

            # each column is minmax scaled to 0 - 1 range
            # perhaps look into a standard scalar to scale to 1 std dev too?
            # also need to add the scalar to model as layer for real data
            # for i in range(med.shape[-1]):
            #     double_plot(med, butter, i)
        else:
            butter = raw_cutoff
        preprocessed = None
        min_max_scaler = preprocessing.MinMaxScaler()
        for i in range(data.shape[-1]):
            butter_scaled = min_max_scaler.fit_transform(butter[:, i].reshape(-1, 1)) 
            if preprocessed is None:
                preprocessed = butter_scaled
            else:
                preprocessed = np.column_stack((preprocessed, butter_scaled))
        return preprocessed

    def butter_lowpass_filter(self, data, low_cut, order=5):
        b, a = butter(order, low_cut, btype='lowpass', fs=64) #setup lowpass filter for 64hz sampling
        y = lfilter(b, a, data)
        return y

    def get_segments(self, sequence_length=1000, stride=125, partition_size=10000, retain_all_windows=False, verbose = True):
        # organize the raw data input into training batches of shape 128x9 with overlap = stride
        # any windows that have non relevant data are completely discarded

        #intialize full size of arrays to save time

        # windows_fog_nonfog = [np.zeros((int(self.length/stride)+1,sequence_length,9)), np.zeros((int(self.length/stride)+1,sequence_length,9))] 
        #this gives us an array of FOG windows and another of non FOG windows, useful for sampling
        #formatted as non fog windows (0), fog windows (1)

        # windows_targets = [np.zeros((int(self.length/stride)+1,sequence_length,9)), np.zeros(int(self.length/stride)+1), np.zeros((int(self.length/stride)+1,sequence_length))]
        #this gives us an array of all windows, and an array denoting if they are FOG or not, preserves order. Useful for visualization
        #formatted as winodws (0), labels (fog = 1 non fog = 0 unrelated = -1 ) (1)
        #original label for each window entry (2)

        windows_fog_nonfog = [np.zeros((int(self.length/stride)+1,sequence_length,9)), np.zeros((int(self.length/stride)+1,sequence_length,9))] 
        #this gives us an array of FOG windows and another of non FOG windows, useful for sampling
        #formatted as non fog windows (0), fog windows (1)

        windows_targets = [np.zeros((int(self.length/stride)+1,sequence_length,9)), np.zeros(int(self.length/stride)+1), np.zeros((int(self.length/stride)+1,sequence_length))]
        #this gives us an array of all windows, and an array denoting if they are FOG or not, preserves order. Useful for visualization
        #formatted as winodws (0), labels (fog = 1 non fog = 0 unrelated = -1 ) (1)
        #original label for each window entry (2)

        #print('This is window targets:',windows_targets[0])
        print(np.zeros((int(self.length/stride)+1,sequence_length,9)).shape)

        counter = [0,0,0] #num of non fog windows, num of fog windows, total num

        self.target[self.target == 1] = 2
        self.target[self.target == 0] = 1
        #self.target[self.target==0] = 3
        for i in range(0, self.length - sequence_length, 1):
            if i % stride == 0:
                fog_index = max(self.target[i:i + sequence_length])
                if (fog_index == 2):
                    ratio = np.count_nonzero(self.target[i:i + sequence_length] == 2) / sequence_length
                    if(ratio < 0.85):
                        fog_index = 1 #changed from 1 to 1.0

                window = self.input[i:i + sequence_length, ...]
                target_window = self.target[i:i + sequence_length]

                # this discards any windows including non relevant data (index 0) if we don't want to retain all windows
                if retain_all_windows or (fog_index < 3): 
                    if not fog_index == 3:
                        windows_fog_nonfog[fog_index-1][counter[fog_index-1]] = window
                        counter[fog_index-1] += 1
                        #print(windows_fog_nonfog)
                    windows_targets[0][counter[2]] = window
                    windows_targets[1][counter[2]] = fog_index-1
                    windows_targets[2][counter[2]] = target_window

                    counter[2] += 1
        freqs = Counter({0:counter[0], 1:counter[1]})
        windows_fog_nonfog[0] = windows_fog_nonfog[0][0:counter[0]]
        windows_fog_nonfog[1] = windows_fog_nonfog[1][0:counter[1]]
        windows_targets[0] = windows_targets[0][0:counter[2]]
        windows_targets[1] = windows_targets[1][0:counter[2]]
        windows_targets[2] = windows_targets[2][0:counter[2]]

        #visualize_windowed_data(windows_targets, 2)

        # minmax scale again
        #print(windows_targets[0].shape)
        #print(windows_targets[0][:,:,0])
        for i in range(0, windows_targets[0].shape[2]):
            print("This is max value:",windows_targets[0][:,:,i].max())
            print("This is min value:",windows_targets[0][:,:,i].min())
            max_accel = windows_targets[0][:,:,i].max()
            min_accel = windows_targets[0][:,:,i].min()
            #print("max, min: {}, {}".format(max_accel, min_accel))
            windows_targets[0][:,:,i] = ((windows_targets[0][:,:,i]) - min_accel) / (max_accel - min_accel)
            windows_fog_nonfog[0][:,:,i] = ((windows_fog_nonfog[0][:,:,i]) - min_accel) / (max_accel - min_accel)
            windows_fog_nonfog[1][:,:,i] = ((windows_fog_nonfog[1][:,:,i]) - min_accel) / (max_accel - min_accel)

        if verbose:
            print("patient ID: ", self.patient_id + 1)
            print("Freezing windows: {}, Non freezing windows: {}".format(len(windows_fog_nonfog[1]), len(windows_fog_nonfog[0])))

        return windows_fog_nonfog, windows_targets, freqs

def evaluate_model(tflite_model, test_images, test_target, threshold):

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    prediction_digits = []

    for test_image in test_images:

        test_image = np.expand_dims(test_image, axis=0).astype(np.float32)


        interpreter.set_tensor(input_index, test_image)
        interpreter.invoke()
        predict = interpreter.get_tensor(output_index).flatten()[0]
        if predict > threshold:
            predict = 1
        else:
            predict = 0
        prediction_digits.append(predict)

    accurate_count = 0
    for i in range(len(test_target)):
        if prediction_digits[i] == test_target[i]:
            accurate_count += 1

    return accurate_count * 1.0 / len(prediction_digits)