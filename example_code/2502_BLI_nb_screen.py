#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 12:52:26 2024

@author: paulruijgrok
"""
import xml.etree.ElementTree as ET
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from scipy.optimize import curve_fit
import os

# Function to decode base64 encoded data
def decode_base64(data):
    decoded_bytes = base64.b64decode(data)
    return np.frombuffer(decoded_bytes, dtype=np.float32)

# Function to process .frd files and extract necessary data
def process_frd_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    root = ET.fromstring(content)

    all_data = []
    exp_info = root.find('.//ExperimentInfo')
    sensor_name = exp_info.find('.//SensorName').text if exp_info.find('.//SensorName') is not None else "Unknown"

    for step in root.findall(".//Step"):
        step_type = step.find('.//StepType').text
        step_name = step.find('.//StepName').text
        time_data_encoded = step.find('.//AssayXData').text
        binding_data_encoded = step.find('.//AssayYData').text

        time_data = decode_base64(time_data_encoded)
        binding_data = decode_base64(binding_data_encoded)

        for t, b in zip(time_data, binding_data):
            all_data.append([sensor_name, step_name, step_type, t, b])

    return pd.DataFrame(all_data, columns=['Sensor', 'Step Name', 'Step Type', 'Time (sec)', 'Binding (nm)'])



   # Equation for piecewise exponential rise-exponential decay fit
def piecewise_exponential(t, A1, tau1, A2, tau2):
    
    func1 = lambda t:A1 + (A0 - A1) * np.exp((-1 * (t - t1))/
                           np.abs(tau1)) 
    A1_eff = A1 + (A0 - A1) * np.exp((-1 * (t2 - t1))/np.abs(tau1)) 
    func2 = lambda t:A2 + (A1_eff - A2) * np.exp((-1 * (t - t2))/
                           np.abs(tau2))
    
    funcs = [A0, func1, func2] 

    return np.piecewise(t,[t <  t1, (t >= t1) & (t < t2), t >= t2], funcs)

def piecewise_exponential_d(t, A1, tau1, A2, tau2, A3, tau3):
    
    func1 = lambda t:A1 + (A0 - A1) * np.exp((-1 * (t - t1))/
                           np.abs(tau1)) +(A0 - A3) * np.exp((-1 * (t - t1))/ np.abs(tau3))
    A1_eff = A1 + (A0 - A1) * np.exp((-1 * (t2 - t1))/np.abs(tau1)) + (A0 - A3) * np.exp((-1 * (t - t1))/np.abs(tau3))
    func2 = lambda t:A2 + (A1_eff - A2) * np.exp((-1 * (t - t2))/
                           np.abs(tau2))
    
    funcs = [A0, func1, func2] 

    return np.piecewise(t,[t <  t1, (t >= t1) & (t < t2), t >= t2], funcs)
   


warnings.filterwarnings("ignore")

# Directory where the .frd files are stored
directory_path = '/'.join(['/Users/paulruijgrok/Library/CloudStorage',
                          'GoogleDrive-ruijgrok@stanford.edu/Shared drives',
                          'MalariaDX/Data/2025/ProteinBindingMeasurements',
                          '250221_Nb/250221_LDH_Nb/'])
# List all .frd files in the directory
frd_files = [os.path.join(directory_path, f"250221_{str(i).zfill(3)}.frd") for i in range(1, 9)]

# Process all .frd files
data_frames = [process_frd_file(f) for f in frd_files]

# Combine all DataFrames into one for easier manipulation
combined_df = pd.concat(data_frames, ignore_index=True)

# Display the combined data
combined_df.head()

# Set the reference file index programmatically
reference_file_index = 2  # Index 2 corresponds to '130615_003.frd'
reference_data = data_frames[reference_file_index]

# Plot 1: Raw data from all sensors
plt.figure(figsize=(12, 6))
colors = plt.cm.get_cmap('tab10', len(frd_files))
for i, df in enumerate(data_frames):
    plt.plot(df['Time (sec)'], df['Binding (nm)'], label=f'Sensor {i+1}', color=colors(i))

plt.title('Raw Data from All Files')
plt.xlabel('Time (sec)')
plt.ylabel('Binding (nm)')
plt.legend()
plt.grid(True)
plt.show()




# Plot 2: Subtracted data (using the chosen file as reference), excluding files 7 and 8
# plt.figure(figsize=(12, 6))
# for i, df in enumerate(data_frames):
#     if i == reference_file_index or i in [3, 4]:  # Skip the reference file and files 7 and 8
#         continue
#     interpolated_ref = np.interp(df['Time (sec)'], reference_data['Time (sec)'], reference_data['Binding (nm)'])
#     subtracted_data = df['Binding (nm)'] - interpolated_ref
#     plt.plot(df['Time (sec)'], subtracted_data, label=f'Sensor {i+1}', color=colors(i))

# plt.title('Subtracted Data Relative to Chosen File as Reference')
# plt.xlabel('Time (sec)')
# plt.ylabel('Subtracted Binding (nm)')
# plt.legend()
# plt.grid(True)
# plt.show()

plt.figure(figsize=(12, 6))
stage_colors = {'BASELINE': 'blue', 'LOADING': 'green', 'ASSOC': 'red', 'DISASSOC': 'purple'}
for sensor in combined_df['Sensor'].unique():
    sensor_data = combined_df[combined_df['Sensor'] == sensor]
    for stage, color in stage_colors.items():
        stage_data = sensor_data[sensor_data['Step Type'] == stage]
        plt.plot(stage_data['Time (sec)'], stage_data['Binding (nm)'], label=f'{sensor} - {stage}', color=color)


#-----------Mb 8---------------------------------------------------------------



reference_file_index = 1  # Index 2 corresponds to '250221_002.frd'
reference_data = data_frames[reference_file_index]
sensor = 'B1'
reference_data  = combined_df[combined_df['Sensor'] == sensor]


# Select the correct time of the experiment for this monobody
t_start = 0
t_end =  800
mb09_data = sensor_data[(sensor_data['Time (sec)'] > t_start) 
                         & (sensor_data['Time (sec)'] < t_end ) ]

ref_data = reference_data[(reference_data['Time (sec)'] > t_start) 
                         & (reference_data['Time (sec)'] < t_end ) ]

data = mb09_data

interpolated_ref = np.interp(data['Time (sec)'], ref_data['Time (sec)'], ref_data['Binding (nm)'])
data['subtracted'] = data['Binding (nm)'] - interpolated_ref

data['subtracted'] = data['subtracted'] - np.mean(data['subtracted'].iloc[0:100].values)
data.to_csv('monobody8_(1t24_s24920_1_dimer.txt',sep='\t')

A0 = np.mean(data['subtracted'].iloc[0:100].values)
start_index = data[data['Step Type']=="ASSOC"].index[0]
data = data.iloc[data.index>start_index]
x_data = data['Time (sec)'] #- data['Time (sec)'].iloc[0]
y_data = data['subtracted']


A1_start, tau1_start = 0.05 , 10
A2_start, tau2_start = 0.04 , 10
A3_start, tau3_start = 0.02 , 300
startpoints= [A0, A1_start, tau1_start, A2_start, tau2_start]
#startpoints= [A0, A1_start, tau1_start, A2_start, tau2_start, A3_start, tau3_start]
t1 =  data[data['Step Type']=="ASSOC"].iloc[0]['Time (sec)']
t2 = data[data['Step Type']=="DISASSOC"].iloc[0]['Time (sec)']

initial_params = startpoints[1:]

x_data = x_data.values
y_data = data['subtracted'].values

# Perform the fit, return early if unsuccesfull
try:
    popt, pcov = curve_fit(
            piecewise_exponential, x_data, y_data, 
            p0=initial_params)
except:
   print("\n Fitting tried tried, fail!!!!!")
   
print('-------Monobody 8-----------')   
print('Observed association rate k_obs (1/s):')
print(1/popt[1])
k2 = 1/popt[3]
print('k_d (1/s) =', 1/popt[3])
k1=(1/popt[1] - 1/popt[3])/ monobody_concentration_um
print('k_a = ' ,(1/popt[1] - 1/popt[3])/ monobody_concentration_um)
K_d = k2/k1
print('K_d = ',K_d)
print('-----------------------------')   


plt.figure(figsize=(12, 6))  # Create a new figure for each plot
plt.plot(data['Time (sec)'], data['subtracted'])
plt.plot(data['Time (sec)'], np.array(piecewise_exponential( data['Time (sec)'].values, *popt)))
plt.title(f'Monobody 13')
#plt.ylim([0.01,0.2])
plt.title(f'Nanobody 1195_14')
plt.xlabel('Time (sec)')
plt.ylabel('Subtracted Binding (nm)')
plt.legend()
plt.grid(True)
plt.show()








#-----------Mb 15---------------------------------------------------------------

# Extract data for monobody 15 (1t24_s22282_1_dimer)
monobody_concentration_um = 5

# Extract data for monobody 5 (1t24_s24643_1_dimer)
sensor = 'B1'
sensor_data = combined_df[combined_df['Sensor'] == sensor]

reference_file_index = 2  # Index 2 corresponds to '130615_003.frd'
reference_data = data_frames[reference_file_index]
sensor = 'C1'
reference_data  = combined_df[combined_df['Sensor'] == sensor]


# Select the correct time of the experiment for this monobody
t_start = 800
t_end =  1500
mb09_data = sensor_data[(sensor_data['Time (sec)'] > t_start) 
                         & (sensor_data['Time (sec)'] < t_end ) ]

ref_data = reference_data[(reference_data['Time (sec)'] > t_start) 
                         & (reference_data['Time (sec)'] < t_end ) ]


data = mb09_data

interpolated_ref = np.interp(data['Time (sec)'], ref_data['Time (sec)'], ref_data['Binding (nm)'])
data['subtracted'] = data['Binding (nm)'] - interpolated_ref

data['subtracted'] = data['subtracted'] - np.mean(data['subtracted'].iloc[0:100].values)
data.to_csv('monobody15_1t24_s24643_1_dimer.txt',sep='\t')

A0 = np.mean(data['subtracted'].iloc[0:100].values)
start_index = data[data['Step Type']=="ASSOC"].index[0]
data = data.iloc[data.index>start_index]
x_data = data['Time (sec)'] #- data['Time (sec)'].iloc[0]
y_data = data['subtracted']


A1_start, tau1_start = 0.05 , 10
A2_start, tau2_start = 0.04 , 10
A3_start, tau3_start = 0.02 , 300
startpoints= [A0, A1_start, tau1_start, A2_start, tau2_start]
#startpoints= [A0, A1_start, tau1_start, A2_start, tau2_start, A3_start, tau3_start]
t1 =  data[data['Step Type']=="ASSOC"].iloc[0]['Time (sec)']
t2 = data[data['Step Type']=="DISASSOC"].iloc[0]['Time (sec)']

initial_params = startpoints[1:]

x_data = x_data.values
y_data = data['subtracted'].values

# Perform the fit, return early if unsuccesfull
try:
    popt, pcov = curve_fit(
            piecewise_exponential, x_data, y_data, 
            p0=initial_params)
except:
   print("\n Fitting tried tried, fail!!!!!")
   
print('-------Monobody 15-----------')   
print('Observed association rate k_obs (1/s):')
print(1/popt[1])
k2 = 1/popt[3]
print('k_d (1/s) =', 1/popt[3])
k1=(1/popt[1] - 1/popt[3])/ monobody_concentration_um
print('k_a = ' ,(1/popt[1] - 1/popt[3])/ monobody_concentration_um)
K_d = k2/k1
print('K_d = ',K_d)
print('-----------------------------')   


plt.figure(figsize=(12, 6))  # Create a new figure for each plot
plt.plot(data['Time (sec)'], data['subtracted'])
plt.plot(data['Time (sec)'], np.array(piecewise_exponential( data['Time (sec)'].values, *popt)))
#plt.ylim([0.01,0.2])
plt.title(f'Monobody 15')
plt.xlabel('Time (sec)')
plt.ylabel('Subtracted Binding (nm)')
plt.legend()
plt.grid(True)
plt.show()



#-----------Mb 5---------------------------------------------------------------

# Extract data for monobody 5 (1t24_s20286_2)
monobody_concentration_um = 20
sensor = 'B1'
sensor_data = combined_df[combined_df['Sensor'] == sensor]

reference_file_index = 2  # Index 2 corresponds to '130615_003.frd'
reference_data = data_frames[reference_file_index]
sensor = 'C1'
reference_data  = combined_df[combined_df['Sensor'] == sensor]


# Select the correct time of the experiment for this monobody
t_start = 1600
t_end =  2400
mb09_data = sensor_data[(sensor_data['Time (sec)'] > t_start) 
                         & (sensor_data['Time (sec)'] < t_end ) ]

ref_data = reference_data[(reference_data['Time (sec)'] > t_start) 
                         & (reference_data['Time (sec)'] < t_end ) ]


data = mb09_data

interpolated_ref = np.interp(data['Time (sec)'], ref_data['Time (sec)'], ref_data['Binding (nm)'])
data['subtracted'] = data['Binding (nm)'] - interpolated_ref

data['subtracted'] = data['subtracted'] - np.mean(data['subtracted'].iloc[0:100].values)
data.to_csv('monobody05_1t24_s20286_2.txt',sep='\t')

A0 = np.mean(data['subtracted'].iloc[0:100].values)
start_index = data[data['Step Type']=="ASSOC"].index[0]
data = data.iloc[data.index>start_index]
x_data = data['Time (sec)'] #- data['Time (sec)'].iloc[0]
y_data = data['subtracted']


A1_start, tau1_start = 0.05 , 10
A2_start, tau2_start = 0.04 , 10
A3_start, tau3_start = 0.02 , 300
startpoints= [A0, A1_start, tau1_start, A2_start, tau2_start]
#startpoints= [A0, A1_start, tau1_start, A2_start, tau2_start, A3_start, tau3_start]
t1 =  data[data['Step Type']=="ASSOC"].iloc[0]['Time (sec)']
t2 = data[data['Step Type']=="DISASSOC"].iloc[0]['Time (sec)']

initial_params = startpoints[1:]

x_data = x_data.values
y_data = data['subtracted'].values

# Perform the fit, return early if unsuccesfull
try:
    popt, pcov = curve_fit(
            piecewise_exponential, x_data, y_data, 
            p0=initial_params)
except:
   print("\n Fitting tried tried, fail!!!!!")
   
print('-------Monobody 5-----------')   
print('Observed association rate k_obs (1/s):')
print(1/popt[1])
k2 = 1/popt[3]
print('k_d (1/s) =', 1/popt[3])
k1=(1/popt[1] - 1/popt[3])/ monobody_concentration_um
print('k_a = ' ,(1/popt[1] - 1/popt[3])/ monobody_concentration_um)
K_d = k2/k1
print('K_d = ',K_d)
print('-----------------------------')   


plt.figure(figsize=(12, 6))  # Create a new figure for each plot
plt.plot(data['Time (sec)'], data['subtracted'])
plt.plot(data['Time (sec)'], np.array(piecewise_exponential( data['Time (sec)'].values, *popt)))
#plt.ylim([0.01,0.2])
plt.title(f'Monobody 5')
plt.xlabel('Time (sec)')
plt.ylabel('Subtracted Binding (nm)')
plt.legend()
plt.grid(True)
plt.show()



#-----------Mb 9---------------------------------------------------------------

# Extract data for monobody 9 (s4753_1_dim)
monobody_concentration_um = 20
sensor = 'B1'
sensor_data = combined_df[combined_df['Sensor'] == sensor]

reference_file_index = 2  # Index 2 corresponds to '130615_003.frd'
reference_data = data_frames[reference_file_index]
sensor = 'C1'
reference_data  = combined_df[combined_df['Sensor'] == sensor]


# Select the correct time of the experiment for this monobody
t_start = 2200
t_end =  3000
mb09_data = sensor_data[(sensor_data['Time (sec)'] > t_start) 
                         & (sensor_data['Time (sec)'] < t_end ) ]

ref_data = reference_data[(reference_data['Time (sec)'] > t_start) 
                         & (reference_data['Time (sec)'] < t_end ) ]


data = mb09_data

interpolated_ref = np.interp(data['Time (sec)'], ref_data['Time (sec)'], ref_data['Binding (nm)'])
data['subtracted'] = data['Binding (nm)'] - interpolated_ref

data['subtracted'] = data['subtracted'] - np.mean(data['subtracted'].iloc[0:100].values)
data.to_csv('monobody09_s1t24_s1137_1_dimer.txt',sep='\t')

A0 = 0.04#np.mean(data['subtracted'].iloc[0:100].values)
start_index = data[data['Step Type']=="ASSOC"].index[0]
data = data.iloc[data.index>start_index]
x_data = data['Time (sec)'] #- data['Time (sec)'].iloc[0]
y_data = data['subtracted']


A1_start, tau1_start = 0.05 , 10
A2_start, tau2_start = 0.04 , 10
A3_start, tau3_start = 0.02 , 300
startpoints= [A0, A1_start, tau1_start, A2_start, tau2_start]
#startpoints= [A0, A1_start, tau1_start, A2_start, tau2_start, A3_start, tau3_start]
t1 =  data[data['Step Type']=="ASSOC"].iloc[0]['Time (sec)']
t2 = data[data['Step Type']=="DISASSOC"].iloc[0]['Time (sec)']

initial_params = startpoints[1:]

x_data = x_data.values
y_data = data['subtracted'].values

# Perform the fit, return early if unsuccesfull
try:
    popt, pcov = curve_fit(
            piecewise_exponential, x_data, y_data, 
            p0=initial_params)
except:
   print("\n Fitting tried tried, fail!!!!!")
   
print('-------Monobody 9-----------')   
print('Observed association rate k_obs (1/s):')
print(1/popt[1])
k2 = 1/popt[3]
print('k_d (1/s) =', 1/popt[3])
k1=(1/popt[1] - 1/popt[3])/ monobody_concentration_um
print('k_a = ' ,(1/popt[1] - 1/popt[3])/ monobody_concentration_um)
K_d = k2/k1
print('K_d = ',K_d)
print('-----------------------------')   


plt.figure(figsize=(12, 6))  # Create a new figure for each plot
plt.plot(data['Time (sec)'], data['subtracted'])
plt.plot(data['Time (sec)'], np.array(piecewise_exponential( data['Time (sec)'].values, *popt)))
#plt.ylim([0.01,0.2])
plt.title(f'Monobody 9')
plt.xlabel('Time (sec)')
plt.ylabel('Subtracted Binding (nm)')
plt.legend()
plt.grid(True)
plt.show()



#-----------Mb 13---------------------------------------------------------------

# Extract data for monobody 13 (1t24_s17456_1)
monobody_concentration_um = 4

sensor = 'D1'
sensor_data = combined_df[combined_df['Sensor'] == sensor]

reference_file_index = 2  # Index 2 corresponds to '130615_003.frd'
reference_data = data_frames[reference_file_index]
sensor = 'F1'
reference_data  = combined_df[combined_df['Sensor'] == sensor]


# Select the correct time of the experiment for this monobody
t_start = 0
t_end =  800
mb09_data = sensor_data[(sensor_data['Time (sec)'] > t_start) 
                         & (sensor_data['Time (sec)'] < t_end ) ]

ref_data = reference_data[(reference_data['Time (sec)'] > t_start) 
                         & (reference_data['Time (sec)'] < t_end ) ]


data = mb09_data

interpolated_ref = np.interp(data['Time (sec)'], ref_data['Time (sec)'], ref_data['Binding (nm)'])
data['subtracted'] = data['Binding (nm)'] - interpolated_ref

data['subtracted'] = data['subtracted'] - np.mean(data['subtracted'].iloc[0:100].values)
data.to_csv('monobody13_1t24_s17456_1.txt',sep='\t')

A0 = np.mean(data['subtracted'].iloc[0:100].values)
start_index = data[data['Step Type']=="ASSOC"].index[0]
data = data.iloc[data.index>start_index]
x_data = data['Time (sec)'] #- data['Time (sec)'].iloc[0]
y_data = data['subtracted']



A1_start, tau1_start = 0.05 , 10
A2_start, tau2_start = 0.04 , 10
A3_start, tau3_start = 0.02 , 300
startpoints= [A0, A1_start, tau1_start, A2_start, tau2_start]
#startpoints= [A0, A1_start, tau1_start, A2_start, tau2_start, A3_start, tau3_start]
t1 =  data[data['Step Type']=="ASSOC"].iloc[0]['Time (sec)']
t2 = data[data['Step Type']=="DISASSOC"].iloc[0]['Time (sec)']

initial_params = startpoints[1:]

x_data = x_data.values
y_data = data['subtracted'].values

# Perform the fit, return early if unsuccesfull
try:
    popt, pcov = curve_fit(
            piecewise_exponential, x_data, y_data, 
            p0=initial_params)
except:
   print("\n Fitting tried tried, fail!!!!!")
   
print('-------Monobody 13-----------')   
print('Observed association rate k_obs (1/s):')
print(1/popt[1])
k2 = 1/popt[3]
print('k_d (1/s) =', 1/popt[3])
k1=(1/popt[1] - 1/popt[3])/ monobody_concentration_um
print('k_a = ' ,(1/popt[1] - 1/popt[3])/ 30)
K_d = k2/k1
print('K_d = ',K_d)
print('-----------------------------')   


plt.figure(figsize=(12, 6))  # Create a new figure for each plot
plt.plot(data['Time (sec)'], data['subtracted'])
plt.plot(data['Time (sec)'], np.array(piecewise_exponential( data['Time (sec)'].values, *popt)))
plt.title(f'Monobody 13')
plt.xlabel('Time (sec)')
plt.ylabel('Subtracted Binding (nm)')
plt.legend()
plt.grid(True)
plt.show()


#-----------Mb 14---------------------------------------------------------------

# Extract data for monobody 14 (1t24_s6260_2)

monobody_concentration_um = 5
sensor = 'D1'
sensor_data = combined_df[combined_df['Sensor'] == sensor]

reference_file_index = 2  # Index 2 corresponds to '130615_003.frd'
reference_data = data_frames[reference_file_index]
sensor = 'F1'
reference_data  = combined_df[combined_df['Sensor'] == sensor]


# Select the correct time of the experiment for this monobody
t_start = 800
t_end =  1500
mb09_data = sensor_data[(sensor_data['Time (sec)'] > t_start) 
                         & (sensor_data['Time (sec)'] < t_end ) ]

ref_data = reference_data[(reference_data['Time (sec)'] > t_start) 
                         & (reference_data['Time (sec)'] < t_end ) ]


data = mb09_data

interpolated_ref = np.interp(data['Time (sec)'], ref_data['Time (sec)'], ref_data['Binding (nm)'])
data['subtracted'] = data['Binding (nm)'] - interpolated_ref

data['subtracted'] = data['subtracted'] - np.mean(data['subtracted'].iloc[0:100].values)
data.to_csv('monobody14_1t24_s6260_2.txt',sep='\t')

#data = mb09_data
interpolated_ref = np.interp(data['Time (sec)'], ref_data['Time (sec)'], ref_data['Binding (nm)'])
data['subtracted'] = data['Binding (nm)'] - interpolated_ref

data['subtracted'] = data['subtracted'] - np.mean(data['subtracted'].iloc[0:100].values)
data.to_csv('monobody2_1t24_s4634_1.txt',sep='\t')

A0 = np.mean(data['subtracted'].iloc[0:100].values)
start_index = data[data['Step Type']=="ASSOC"].index[0]
data = data.iloc[data.index>start_index]
x_data = data['Time (sec)'] #- data['Time (sec)'].iloc[0]
y_data = data['subtracted']



A1_start, tau1_start = 0.05 , 10
A2_start, tau2_start = 0.04 , 10
A3_start, tau3_start = 0.02 , 300
startpoints= [A0, A1_start, tau1_start, A2_start, tau2_start]
#startpoints= [A0, A1_start, tau1_start, A2_start, tau2_start, A3_start, tau3_start]
t1 =  data[data['Step Type']=="ASSOC"].iloc[0]['Time (sec)']
t2 = data[data['Step Type']=="DISASSOC"].iloc[0]['Time (sec)']

initial_params = startpoints[1:]

x_data = x_data.values
y_data = data['subtracted'].values

# Perform the fit, return early if unsuccesfull
try:
    popt, pcov = curve_fit(
            piecewise_exponential, x_data, y_data, 
            p0=initial_params)
except:
   print("\n Fitting tried tried, fail!!!!!")
   
print('-------Monobody 14-----------')   
print('Observed association rate k_obs (1/s):')
print(1/popt[1])
k2 = 1/popt[3]
print('k_d (1/s) =', 1/popt[3])
k1=(1/popt[1] - 1/popt[3])/ monobody_concentration_um
print('k_a = ' ,(1/popt[1] - 1/popt[3])/ 30)
K_d = k2/k1
print('K_d = ',K_d)
print('-----------------------------')   


plt.figure(figsize=(12, 6))  # Create a new figure for each plot
plt.plot(data['Time (sec)'], data['subtracted'])
plt.plot(data['Time (sec)'], np.array(piecewise_exponential( data['Time (sec)'].values, *popt)))
#plt.ylim([0.01,0.2])
plt.title(f'Monobody 14')
plt.xlabel('Time (sec)')
plt.ylabel('Subtracted Binding (nm)')
plt.legend()
plt.grid(True)
plt.show()



#-----------Mb 2---------------------------------------------------------------

# Extract data for monobody 2 (1t24_s4634_1)
monobody_concentration_um = 30

sensor = 'D1'
sensor_data = combined_df[combined_df['Sensor'] == sensor]

reference_file_index = 2  # Index 2 corresponds to '130615_003.frd'
reference_data = data_frames[reference_file_index]
sensor = 'F1'
reference_data  = combined_df[combined_df['Sensor'] == sensor]


# Select the correct time of the experiment for this monobody
t_start = 1500
t_end =  2000
mb09_data = sensor_data[(sensor_data['Time (sec)'] > t_start) 
                          & (sensor_data['Time (sec)'] < t_end ) ]

ref_data = reference_data[(reference_data['Time (sec)'] > t_start) 
                          & (reference_data['Time (sec)'] < t_end ) ]


data = mb09_data

interpolated_ref = np.interp(data['Time (sec)'], ref_data['Time (sec)'], ref_data['Binding (nm)'])
data['subtracted'] = data['Binding (nm)'] - interpolated_ref

data['subtracted'] = data['subtracted'] - np.mean(data['subtracted'].iloc[0:100].values)
data.to_csv('monobody2_1t24_s4634_1.txt',sep='\t')

A0 = np.mean(data['subtracted'].iloc[0:100].values)
data = data.iloc[data.index>52592]
x_data = data['Time (sec)'] #- data['Time (sec)'].iloc[0]
y_data = data['subtracted']



A1_start, tau1_start = 0.05 , 10
A2_start, tau2_start = 0.04 , 10
A3_start, tau3_start = 0.02 , 300
startpoints= [A0, A1_start, tau1_start, A2_start, tau2_start]
#startpoints= [A0, A1_start, tau1_start, A2_start, tau2_start, A3_start, tau3_start]
t1 =  data[data['Step Type']=="ASSOC"].iloc[0]['Time (sec)']
t2 = data[data['Step Type']=="DISASSOC"].iloc[0]['Time (sec)']

initial_params = startpoints[1:]

x_data = x_data.values
y_data = data['subtracted'].values

# Perform the fit, return early if unsuccesfull
try:
    popt, pcov = curve_fit(
            piecewise_exponential, x_data, y_data, 
            p0=initial_params)
except:
    print("\n Fitting tried tried, fail!!!!!")
   
print('-------Monobody 2-----------')   
print('Observed association rate k_obs (1/s):')
print(1/popt[1])
k2 = 1/popt[3]
print('k_d (1/s) =', 1/popt[3])
k1=(1/popt[1] - 1/popt[3])/ monobody_concentration_um
print('k_a = ' ,(1/popt[1] - 1/popt[3])/ 30)
K_d = k2/k1
print('K_d = ',K_d)
print('-----------------------------')   



plt.figure(figsize=(12, 6))  # Create a new figure for each plot
plt.plot(data['Time (sec)'], data['subtracted'])
plt.plot(data['Time (sec)'], np.array(piecewise_exponential( data['Time (sec)'].values, *popt)))
#plt.ylim([0.01,0.2])
plt.title(f'Monobody 2')
plt.xlabel('Time (sec)')
plt.ylabel('Subtracted Binding (nm)')
plt.legend()
plt.grid(True)
plt.show()









