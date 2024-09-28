import numpy as np 
import matplotlib.pyplot as plt 
import scipy as sp
import h5py as hp  
import os 
import pandas as pd
import re
import json 



   


# def read_zemax_phase(zemax_file_path):
#     # NOTE: zemax x coordinate may be in mm 
#     # Check
#      #=============================================
#     # read the optimized phase from Zemax 
#     #=============================================
#     # Define file path (change it to your actual file path)
#     file_path = zemax_file_path

#     # Initialize lists to store the data
#     x_zemax = []
#     y_zemax = []
#     phase_zemax = []

#     # Try reading the file with a different encoding, such as 'utf-16' or 'latin-1'
#     try:
#         with open(file_path, 'r', encoding='utf-8') as file:
#             lines = file.readlines()
#     except UnicodeDecodeError:
#         with open(file_path, 'r', encoding='utf-16') as file:
#             lines = file.readlines()

#     # Process each line
#     for line in lines:
#         # Match lines that contain valid floating point numbers
#         match = re.match(r'\s*([-+]?\d*\.\d+E[-+]?\d+)\s+([-+]?\d*\.\d+E[-+]?\d+)\s+([-+]?\d*\.\d+E[-+]?\d+)', line)
        
#         if match:
#             # Extract the values and append them to the respective lists
#             x_coord = float(match.group(1))
#             y_coord = float(match.group(2))
#             phase = float(match.group(3))
            
#             x_zemax.append(x_coord)
#             y_zemax.append(y_coord)
#             phase_zemax.append(phase)


#     return x_zemax, y_zemax, phase_zemax



# Function to read CSV and convert it to dictionary
def read_csv_with_pandas(filename):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(filename)
    
    # Convert the DataFrame to a dictionary
    result_dict = df.to_dict(orient='list')
    
    return result_dict

def create_phase_target(phase_zemax, x_zemax, x_mask):
    interp_func = sp.interpolate.interp1d(x_zemax,phase_zemax, kind='cubic')
    phase_target = interp_func(x_mask)
    
    return phase_target