import utility 
import re 
import logging
import numpy as np 
import matplotlib.pyplot as plt
import json 
import os 
import apexatoms as aa 




# #=================================================
# Read the Lens Configuration File 
# #=================================================
current_directory= os.getcwd()
current_directory = r"{}".format(current_directory)

config_file_path = current_directory+ '\\LensConfiguration.json'

with open(config_file_path, 'r') as file:
    LensConfig = json.load(file)

#=================================================
    
    
class  PhaseDesign:
    def __init__(self):
        self.Wavelength = LensConfig["Wavelength"]
        self.FocalLength = LensConfig["FocalLength"]
    def load_optimized_phase(self,file_path):
        # NOTE: zemax x coordinate may be in mm 
        # Check
        #=============================================
        # read the optimized phase from Zemax 
        #=============================================
        # Define file path (change it to your actual file path)
        

        # Initialize lists to store the data
        self.x_zemax = []
        self.y_zemax = []
        self.phase_zemax = []

        # Try reading the file with a different encoding, such as 'utf-16' or 'latin-1'
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='utf-16') as file:
                lines = file.readlines()

        # Process each line
        for line in lines:
            # Match lines that contain valid floating point numbers
            match = re.match(r'\s*([-+]?\d*\.\d+E[-+]?\d+)\s+([-+]?\d*\.\d+E[-+]?\d+)\s+([-+]?\d*\.\d+E[-+]?\d+)', line)
            
            if match:
                # Extract the values and append them to the respective lists
                x_coord = float(match.group(1))
                y_coord = float(match.group(2))
                phase = float(match.group(3))
                
                self.x_zemax.append(x_coord)
                self.y_zemax.append(y_coord)
                self.phase_zemax.append(phase)

        self.x_zemax= self.x_zemax= np.array([i* 1e3 for i in self.x_zemax]) # make sure that zemax dimension is in mm
        return self.x_zemax, self.phase_zemax
   
    def compare_optimized_ideal(self):
        # the x of zemax comes in mm 
        self.phase_zemax= (2*np.pi)*np.array([i for i in self.phase_zemax])
        self.phase_zemax= self.phase_zemax- min(self.phase_zemax)
        #phase_zemax= np.mod(phase_zemax, 2* np.pi)

        phase_hyperbolic = -(2*np.pi/self.Wavelength)*(np.sqrt(self.x_zemax**2+self.FocalLength**2)-self.FocalLength) # Hyperbolic lens is assumed with mod 2pi phase target
        #phase_hyperbolic= -np.mod(phase_hyperbolic, 2* np.pi)
        phase_hyperbolic= phase_hyperbolic- min(phase_hyperbolic)


        plt.scatter(self.x_zemax, self.phase_zemax, color= 'darkblue',label= 'Zemax Optimized Phase')
        plt.plot(self.x_zemax, phase_hyperbolic,linestyle= '--',color='darkred', label= 'Ideal Hyperbolic Phase', linewidth= 2.0)
        plt.ylabel('Phase (rad)', fontdict={'size': 14, 'weight': 'bold', 'font': 'arial'})
        plt.xlabel('Radius (μm)', fontdict={'size': 14, 'weight': 'bold', 'font': 'arial'})
        #plt.ylim(0,2* np.pi)
        plt.legend()
        plt.tight_layout()
        plt.show()


class  UnitCellDesign:
    def __init__(self):
        self.Wavelength = LensConfig["Wavelength"]
        self.FocalLength = LensConfig["FocalLength"]

        self.RadiusMin = LensConfig["UnitCell"]["Radius"][0]
        self.RadiusMax= LensConfig["UnitCell"]["Radius"][1]
        self.Pitch= LensConfig["UnitCell"]["Period"]
        self.Height= LensConfig["UnitCell"]["Height"]
        self.NumAtom= LensConfig["UnitCell"]["NumAtom"]
        self.SubstrateIndex= LensConfig["SubstrateIndex"]
        self.PillarIndex= LensConfig["PillarIndex"]
        self.diameter= np.linspace(2* self.RadiusMin, 2*self.RadiusMax, self.NumAtom)
        
    def create_unit_cell_library(self,show_plot=False,file=None):       
        simple_library = aa.SimpleCircleLibrary(
							pitch=self.Pitch,
							diameter=self.diameter,
							depth= self.Height,
							wavelength=self.Wavelength,
							inIndex= self.SubstrateIndex,
							outIndex=1,
							atomIndex= self.PillarIndex,
							surroundIndex=1,
							grid="square")

        simple_library.calculate(numBasis=125)
        phase, transmission, reflection = simple_library.getResults()
        if show_plot:
            plt.plot(self.diameter,phase, color= 'darkred', linewidth = 2.5)
            plt.xlabel("Radius (μm)", fontdict={'size': 18})
            plt.ylabel("Phase (Radian)", fontdict={'size': 18})
            plt.title('Unit Cell Libray: Diameter vs Phase', fontdict={'size': 18})
            plt.show()
        return self.diameter, phase, transmission

        # print("Results for Simple Circle Atom")
        # print(f"\tPhase: {phase}")
    # print(f"\tTransmission: {transmission}")
    # print(f"\tReflection: {reflection}")

    # get a dict of all the atom parameters:

    # params = simple_atom.getParams()

    # print("Atom parameters:")
    # for key, value in params.items():
    #     print("\t",key,"\t",value)
# class  LensDesign(PhaseDesign):
#     def __init__(self, use_ideal_phase= False):
#         print()

