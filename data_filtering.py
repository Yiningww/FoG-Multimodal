import numpy as np
import os
import datetime
data_folder = '/Users/wangyining/Desktop/Research/Xuanwu/Code/Filtered Data/'
'''
data_list = []

for foldername in sorted(os.listdir(data_folder)):
    if "00" in foldername:
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
#print(data_list)

data_list = ['/Users/wangyining/Desktop/test.txt']

all_data = []
for file_name in data_list:
    # Load the data from the file
    file_data = np.genfromtxt(file_name, delimiter=",", dtype=str)

    #date_strings = file_data[:, 1]
    #print(date_strings)
    #date_objects = np.array([datetime.datetime.strptime(date_string, '%H:%M:%S.%f') for date_string in date_strings])


    # Stack the datetime and integer arrays horizontally
    #stacked_data = np.hstack((date_objects.reshape(-1, 1), int_values))

    # Convert the string data to float data
    file_data_float = file_data[:, 2:].astype(float)

    # Add the converted data to the list
    all_data.append(file_data_float)

# Combine the data from all the files into a single NumPy array
combined_data = np.concatenate(all_data, axis=0)
#column_titles = "FP1, FP2, F3, F4, C4, C4, P3, P4, O1, O2, F7, F8, P7, P8, FZ, CZ, PZ, FC1, FC2, CP1, " \
               #"CP2, FC5, FC6, CP5, CP6, EMG-1, EMG-2, IO, EMG-3, EMG-4," \
               #"LS-acc-x, LS-acc-y, LS-acc-z, LS-Gyro-x, LS-Gyro-y, LS-Gyro-z, LS-NC," \
               #"RS-acc-x, RS-acc-y, RS-acc-z, RS-Gyro-x, RS-Gyro-y, RS-Gyro-z, RS-NC," \
               #"waist-acc-x, waist-acc-y, waist-acc-z, waist-Gyro-x, waist-Gyro-y, waist-Gyro-z, waist-NC, " \
               #"arm-acc-x, arm-acc-y, arm-acc-z, arm-Gyro-x, arm-Gyro-y, arm-Gyro-z, arm-SC, Label"

# Save the combined data to a file
np.savetxt("combined_data.txt", combined_data, fmt="%f")
np.savetxt("combined_data.csv", combined_data, fmt="%f")

# Load the data from the text file
#data = np.genfromtxt('combined_data.txt', delimiter=',', dtype=str)

# Save the data to a CSV file
#np.savetxt('combined_data.csv', combined_data, delimiter=' ', fmt='%f')

#ata = np.genfromtxt('combined_data.csv', delimiter=',', skip_header=0)
# Define the input and output file paths
'''
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
'''
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
'''
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






