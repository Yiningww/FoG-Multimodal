import numpy as np
import pandas as pd
import os
import datetime
data_folder = '/Users/wangyining/Desktop/Research/Xuanwu/Code/Filtered Data/'

data_list = []
patient_list = ["001", "003", "009", "010"]
for foldername in sorted(os.listdir(data_folder)):
    if foldername in patient_list:
        each_patient_path = os.path.join(data_folder,foldername)
        #print(each_patient_path)

        for each_task in sorted(os.listdir(each_patient_path)):
            if ".txt" in each_task:
                each_task_path = os.path.join(each_patient_path,each_task)
                #current_data_file = np.loadtxt(each_task_path, delimiter=",", dtype=str)
                data_list.append(each_task_path)
            elif "OFF" in each_task:
                each_task_path = os.path.join(each_patient_path,each_task)
                for each_task_txt in sorted(os.listdir(each_task_path)):
                    each_task_txt_path = os.path.join(each_task_path,each_task_txt)
                    #current_data_file = np.loadtxt(each_task_txt_path)
                    data_list.append(each_task_txt_path)

print(data_list)
#data_list = ['/Users/wangyining/Desktop/test.txt']

all_data = []
counter = 1
data_length = 0
for file_name in data_list:
    # Load the data from the file
    file_data = np.genfromtxt(file_name, delimiter=",", dtype=str)

    #date_strings = file_data[:, 1]
    #print(date_strings)
    #date_objects = np.array([datetime.datetime.strptime(date_string, '%H:%M:%S.%f') for date_string in date_strings])


    # Stack the datetime and integer arrays horizontally
    #stacked_data = np.hstack((date_objects.reshape(-1, 1), int_values))

    # Convert the string data to float data
    file_data_str = file_data[:,0].astype(int)
    file_data_float = file_data[:, 2:].astype(float)


    #file_data_int = file_data_float.astype(int)
    result = np.hstack((file_data_str[:, np.newaxis], file_data_float))
    #last_data_int = file_data[:, 60].astype(int)
    #result = np.hstack((con[:, np.newaxis],last_data_int))
    acc_part = result[:,[0, 31, 32, 33, 45, 46, 47, 52, 53, 54, 59]]
    print(file_name)
    file_split = file_name.split('/')
    name = "S"+file_split[-2][-2:]+"R0"+file_split[-1][5]+'.txt'
    counter += 1
    file_path = data_folder+'Test/'+name
    print(len(file_data_str))

    data_length += len(file_data_str)
    np.savetxt(file_path,acc_part,fmt="%i")
    '''
    is_zero_1 = np.all(result[:, [31, 32, 33]] == 0.0)
    print("111:",is_zero_1)
    is_zero_2 = np.all(result[:, [38, 39, 40]] == 0.0)
    print("222:",is_zero_2)
    is_zero_3 = np.all(result[:, [45, 46, 47]] == 0.0)
    print("333",is_zero_3)
    is_zero_4 = np.all(result[:, [52, 53, 54]] == 0.0)
    print("444",is_zero_4)
    '''
    #if is_zero_1 != 1 and is_zero_3 != 1 and is_zero_4 != 1:
        #print("patient does have sufficient acc data:",file_name)
    # Add the converted data to the list
    #else:
        #print("patient does not have acc data",file_name)
    #all_data.append(result)
print("data length is:",data_length)
exit()
# Combine the data from all the files into a single NumPy array
combined_data = np.concatenate(all_data, axis=0)

#np.savetxt("combined_data.txt", combined_data, fmt="%f")
np.savetxt("test_for_each_column.txt", combined_data, fmt="%f")
acc = np.loadtxt('test_for_each_column.txt')
accc = acc[:, [0, 31, 32, 33, 38, 39, 40, 45, 46, 47, 52, 53, 54, 59]]
np.savetxt("outcome_test.txt", accc, fmt = "%f")
print(accc.shape)
#print(accc[:,11])
exit()



#column_titles = "FP1, FP2, F3, F4, C4, C4, P3, P4, O1, O2, F7, F8, P7, P8, FZ, CZ, PZ, FC1, FC2, CP1, " \
               #"CP2, FC5, FC6, CP5, CP6, EMG-1, EMG-2, IO, EMG-3, EMG-4," \
               #"LS-acc-x, LS-acc-y, LS-acc-z, LS-Gyro-x, LS-Gyro-y, LS-Gyro-z, LS-NC," \
               #"RS-acc-x, RS-acc-y, RS-acc-z, RS-Gyro-x, RS-Gyro-y, RS-Gyro-z, RS-NC," \
               #"waist-acc-x, waist-acc-y, waist-acc-z, waist-Gyro-x, waist-Gyro-y, waist-Gyro-z, waist-NC, " \
               #"arm-acc-x, arm-acc-y, arm-acc-z, arm-Gyro-x, arm-Gyro-y, arm-Gyro-z, arm-SC, Label"

'''
# Save the combined data to a file
np.savetxt("combined_data.txt", combined_data, fmt="%f")
np.savetxt("combined_data.csv", combined_data, fmt="%f")

# Load the data from the text file
#data = np.genfromtxt('combined_data.txt', delimiter=',', dtype=str)

# Save the data to a CSV file
#np.savetxt('combined_data.csv', combined_data, delimiter=' ', fmt='%f')

#ata = np.genfromtxt('combined_data.csv', delimiter=',', skip_header=0)
# Define the input and output file paths

#data = np.loadtxt('combined_data.txt')
csv = np.loadtxt('combined_data.csv')
#print(csv[:5, :])
print("shape of all data:",csv.shape)
#column_means = np.mean(csv, axis=0)
#print("column means are:",column_means)
#print("shape of column means:",column_means.shape)
#z_scores = (csv - np.mean(csv, axis=0)) / np.std(csv, axis=0)
#print("Z-Scores are:",z_scores)
#print("shape of z scores:",z_scores.shape)
fog_counter = 0
fog_episode = []

same_episode = 0
test = [0,0,1,1,1,0,1,1,0,0,0,1,1,0,1]
for row in len(test):
    #print(row[58])
    if test[row] == 1 and test[fog_counter - 1] == 0 and row != 0:
        fog_episode.append(0.002)
        #fog_episode[row] += 0.002
        print(row)
        new_counter = fog_counter + 1
        label = csv[new_counter]
        while label == 1 and new_counter < len(csv) - 1:
            fog_episode[row] += 0.002
            fog_counter = new_counter
            new_counter += 1
            label = csv[new_counter][58]
    elif test[row] = 1 and row = 0:
        fog_episode.append(0.002)
        counter
        

    fog_counter += 1

for row in range(len(csv)):
    if csv[row][58] == 1.0:
        if row == 0:
            fog_episode.append(1)
        elif csv[row-1][58] == 0.0:
            fog_episode.append(1)
        else:
            fog_episode[-1] += 1


#print(fog_episode)
#for i in fog_episode:
    #print(i)

### EXTRACT ACC DATA
#1-59 --> 0-58  32-59 --> 30-57
#acc_data = csv[:, [30, 31, 32, 37, 38, 39, 44, 45, 46, 51, 52, 53]]
#np.savetxt("acc_data.csv", acc_data)
acc_csv = np.loadtxt('acc_data.csv')
# Find the indices of the elements that are equal to zero
zero_indices = np.where(acc_csv == 0)
# Create a new array that excludes the zero elements
new_data = np.delete(acc_csv, zero_indices, axis=0)
# Save the new array to a file
np.savetxt('omitted_acc_data.csv', new_data)
omitted_acc_csv = np.loadtxt("omitted_acc_data.csv")
column_means = np.mean(omitted_acc_csv, axis=0)
print(column_means, column_means.shape)
z_scores = (omitted_acc_csv - np.mean(omitted_acc_csv, axis=0)) / np.std(omitted_acc_csv, axis=0)
print("Z-Scores are:",z_scores)
print("shape of z scores:",z_scores.shape)
'''

### Data Segmentation
# Read the CSV file into a pandas dataframe
df = pd.read_csv('combined_data.csv', delimiter = " ", header = None)

# Define the size of the window and the step size
window_size = 1000 #2s/0.002s
step_size = 125 #0.25s/0.002s

# Create a list to store the segmented data
segmented_data_2s = []
labels_2s = []
label_counter = 0
# Loop through the dataframe and segment the data
print(df.shape)

for i in range(0, len(df), step_size):
    if i + window_size <= len(df):
        segment = df.iloc[i:i+window_size]
        for row in range(i,i+window_size):
            if df[58][row] == 1.0:
                label_counter += 1
        if label_counter >= 850:
            label = 1
            labels_2s.append(label)
            label_counter = 0
        else:
            label = 0
            labels_2s.append(label)
            label_counter = 0

        #print(segment)
        #print(segment.shape)

        segmented_data_2s.append(segment)


#print(labels_2s)
print(labels_2s.count(1))
print(labels_2s.count(0))
print(len(labels_2s))#should be 34457, since total_windows = ((len(df) - window_size) // step_size) + 1
print("segment shape is:")
print(segment.shape)
# Concatenate the segmented data into a new dataframe
segmented_df = pd.concat(segmented_data_2s)
print(segmented_df.shape)
segmented_df.to_csv('segmented_data_2s_0.25s.csv', index=True)

##### Implementation of Tian's partition method to current phase




