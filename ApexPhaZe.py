import utility 
import re 
import logging
import numpy as np 
import matplotlib.pyplot as plt
import json 
import os,sys
import apexatoms as aa 
import scipy as sp 
import matplotlib.patches as patches
from scipy.integrate import  simpson
from scipy.optimize import curve_fit
from scipy.constants import epsilon_0, mu_0
sys.path.append("C:\\Program Files\\Lumerical\\v242\\api\\python") # this line is to direct the python script ot the lumerical python api 
import lumapi

current_directory= os.getcwd()
current_directory= r"{}".format(current_directory)
sys.path.append(current_directory)

class ConfigReader:
    _config_data = None

    @classmethod
    def load_config(cls, config_file= None):

        with open(config_file, 'r') as file:
            cls._config_data = json.load(file)

    @classmethod
    def get_config(cls):
        return cls._config_data

#==============================================
#  If no configuration file path is provided
# default configuration path is chosen
#==============================================
config_check = ConfigReader.get_config()
if config_check== None:
    print("Using default configuration file path.")

default_config_path = current_directory+ '\\LensConfiguration.json'
ConfigReader.load_config(default_config_path)



class  PhaseDesign:
    def __init__(self):
        LensConfig= ConfigReader.get_config()
        self.FocalLength= LensConfig["FocalLength"]
        self.Wavelength = LensConfig["Wavelength"]
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
        self.phase_zemax= (2*np.pi)*np.array([i for i in self.phase_zemax])
        self.phase_zemax= self.phase_zemax- min(self.phase_zemax)
       
        return self.x_zemax, self.phase_zemax
   
    def compare_optimized_ideal(self):
        phase_hyperbolic = -(2*np.pi/self.Wavelength)*(np.sqrt(self.x_zemax**2+self.FocalLength**2)-self.FocalLength) # Hyperbolic lens is assumed with mod 2pi phase target
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
        LensConfig= ConfigReader.get_config()
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
            plt.plot(self.diameter/2,phase, color= 'darkred', linewidth = 2.5)
            plt.xlabel("Radius (μm)", fontdict={'size': 18})
            plt.ylabel("Phase (Radian)", fontdict={'size': 18})
            plt.title('Unit Cell Libray: Diameter vs Phase', fontdict={'size': 18})
            plt.show()

        # self.phase_interp = np.linspace(0, 2* np.pi, )
        return self.diameter/2, phase, transmission
    def unit_cell_read(self,file_path):
        None

        # print("Results for Simple Circle Atom")
        # print(f"\tPhase: {phase}")
    # print(f"\tTransmission: {transmission}")
    # print(f"\tReflection: {reflection}")

    # get a dict of all the atom parameters:

    # params = simple_atom.getParams()

    # print("Atom parameters:")
    # for key, value in params.items():
    #     print("\t",key,"\t",value)
class  LensDesign:
    def __init__(self):
        LensConfig= ConfigReader.get_config()
        self.LensDiameter = LensConfig["LensDiameter"]
        self.Wavelength = LensConfig["Wavelength"]
        self.FocalLength = LensConfig["FocalLength"]
        self.RadiusMin = LensConfig["UnitCell"]["Radius"][0]
        self.RadiusMax= LensConfig["UnitCell"]["Radius"][1]
        self.Pitch= LensConfig["UnitCell"]["Period"]
        self.Height= LensConfig["UnitCell"]["Height"]
        self.NumAtom= LensConfig["UnitCell"]["NumAtom"]
        self.SubstrateIndex= LensConfig["SubstrateIndex"]
        self.PillarIndex= LensConfig["PillarIndex"]
        self.LensRadius= self.LensDiameter/2 

    def make_lens_geometry(self,radius_zemax, phase_zemax, radius_unitcell, phase_unitcell, idealphase= False, show_lens= False):
        self.radius_zemax= radius_zemax
        self.phase_zemax= phase_zemax
        self.radius_unitcell= radius_unitcell
        self.phase_unitcell= phase_unitcell


        # this file was written to visualize the meta-atom placement 
        FullLensNumMetaAtomRow = round(self.LensRadius/self.Pitch)

        x_mask = self.Pitch* np.arange(-FullLensNumMetaAtomRow, FullLensNumMetaAtomRow+1)
        y_mask = self.Pitch* np.arange(-FullLensNumMetaAtomRow, FullLensNumMetaAtomRow+1)
    
        #print(x_mask)
        if idealphase:
            phase_target = -np.mod((2*np.pi/self.Wavelength)*(np.sqrt(x_mask**2+self.FocalLength**2)-self.FocalLength),2*np.pi) # Hyperbolic lens is assumed with mod 2pi phase target
            phase_target  = phase_target  - min(phase_target )
        else:
            phase_target= utility.create_phase_target(self.phase_zemax, self.radius_zemax, x_mask)


        phase_target= np.mod(phase_target, 2* np.pi)
        

        # #===========================
        # plt.scatter(x_mask,phase_target, color = 'darkblue', linewidth= 2.5)
        # plt.ylabel('Phase (rad)', fontdict={'size': 14, 'weight': 'bold', 'font': 'arial'})
        # plt.xlabel('Radius (μm)', fontdict={'size': 14, 'weight': 'bold', 'font': 'arial'})
        # plt.title('Phase target')
        # plt.show()
        # # ===========================
    
        # Step 3: Obtain the radius vs. position yielding the desired phase profile    
        # Create the interpolation function (linear by default)
        self.phase_unitcell_unwrapped= np.unwrap(self.phase_unitcell)
        self.phase_unitcell_unwrapped= self.phase_unitcell_unwrapped- self.phase_unitcell_unwrapped[0]

        interp_func = sp.interpolate.interp1d(self.phase_unitcell_unwrapped, self.radius_unitcell, kind='cubic')
        
        # Interpolate to find corresponding 'rad' values for 'phase_target'
        radius_mask = interp_func(phase_target)

                
        # Create a new figure
        plt.figure()

        # Get the current axis
        ax = plt.gca()
        x_y_rad= []

        # Loop over x_mask and y_mask to compute r and add circles where r <= LensRadius
        for i, x in enumerate(x_mask):
            for j, y in enumerate(y_mask):
                r = np.sqrt(x**2 + y**2)
            # print(r)
                if r <= self.LensRadius:
                    # Find the index where x_mask is close to the radius r
                    # (use np.abs to handle floating point precision)
                    idx = np.where(np.abs(x_mask - r) <= 0.2)[0]
                    if idx.size > 0:  # If index found
                        pillar_rad=  radius_mask[idx[0]]
                        circle = patches.Circle((x, y), pillar_rad, color='b', fill=True)
                        x_y_rad.append({'x': x, 'y': y, 'radius': pillar_rad})
                        ax.add_patch(circle)
                    
                    
                

        # Set axis limits to match the range of x_mask and y_mask
        delta= self.LensRadius/4
        ax.set_xlim(min(x_mask)-delta, max(x_mask)+ delta)
        ax.set_ylim(min(y_mask)-delta, max(y_mask)+ delta)

        # Equal aspect ratio for correct circle appearance
        ax.set_aspect('equal')
        plt.title('Metalens')
        if show_lens:
            # Display the plot
            plt.show()

        x_python= np.array([elem['x']*1e-6 for elem in x_y_rad])
        y_python= np.array([elem['y']*1e-6 for elem in x_y_rad])
        rad_python= np.array([elem['radius']*1e-6 for elem in x_y_rad])

        # Stack arrays column-wise
        data = np.column_stack((x_python, y_python, rad_python))

        # Save to CSV
        if idealphase:
            fname= '_idealphase'
        else:
            fname= '_optimized'
            
        np.savetxt('fulllens_geometry_x_y_rad_apexPHAZE'+ fname+ '.txt', data, delimiter=",")
        return x_python, y_python, rad_python
            
    
class FullLensSim:
    def __init__(self):
        LensConfig= ConfigReader.get_config()

        self.LensDiameter = LensConfig["LensDiameter"]
        self.Wavelength = LensConfig["Wavelength"]
        self.FocalLength = LensConfig["FocalLength"]
        self.RadiusMin = LensConfig["UnitCell"]["Radius"][0]
        self.RadiusMax= LensConfig["UnitCell"]["Radius"][1]
        self.Pitch= LensConfig["UnitCell"]["Period"]
        self.Height= LensConfig["UnitCell"]["Height"]
        self.NumAtom= LensConfig["UnitCell"]["NumAtom"]
        self.SubstrateIndex= LensConfig["SubstrateIndex"]
        self.PillarIndex= LensConfig["PillarIndex"]
        self.LensRadius= self.LensDiameter/2


        self.MeshRefinementInteger= LensConfig["FullLens"]["MeshRefinementInteger"]
        self.PMLDistance= LensConfig["FullLens"]["PMLDistanceFactor"]* self.Wavelength

        self.um= 1e-6 
        
    def run_fdtd(self, xcoord,ycoord, pillar_radius):
        fdtd= lumapi.FDTD()
        self.xcoord= xcoord 
        self.ycoord= ycoord 
        self.pillar_radius= pillar_radius

        lens_geometry_file_name=  'fulllens_geometry_x_y_rad_apexPHAZE_NOT_optimized.txt'
        
        

        setup_script= """
        # create an aperture matching the diameter of the metalens 
        R = getnamed("metalens","lens_radius");
        W = getnamed("metalens::substrate","x span");
        V = R*exp(1i*linspace(0,2*pi,200));
        V = [W/2,0; W/2,-W/2; -W/2,-W/2; -W/2,W/2; W/2,W/2; W/2,0; real(V),imag(V)];
        setnamed("aperture","vertices",V);
        """
        fdtd.set("setup script",  setup_script)

        #radius_phase_file = current_directory+ '\\2_meta_library\\unit cell'+'\\' + unit_cell_library_file_name
        #geometry_file= current_directory+ '\\'+ lens_geometry_file_name

        # #==================================================
        # Add FDTD
        # #==================================================
        xspan =  (self.LensDiameter+ 2* self.PMLDistance)* self.um
        yspan = (self.LensDiameter+ 2* self.PMLDistance)* self.um
        zspan= 1.8* self.um 

        fdtd.addfdtd()
        fdtd.set("dimension",2);  #  1 = 2D, 2 = 3D
        fdtd.set("x",0)
        fdtd.set("x span",xspan)
        fdtd.set("y",0)
        fdtd.set("y span",yspan)
        fdtd.set("z",0.6e-6)
        fdtd.set("z span",zspan) # change the z-span later
        fdtd.set("mesh type", "auto non-uniform")
        fdtd.set("mesh accuracy", self.MeshRefinementInteger)
        fdtd.set("x min bc","Anti-Symmetric")
        fdtd.set("y min bc","Symmetric")



        # #==================================================
        # Add the metalens geometry 
        # #==================================================
        fdtd.addstructuregroup()
        fdtd.set("name", "metalens")
        # fdtd.adduserprop("radius_vs_phase_data_file", 1, radius_phase_file)
        fdtd.adduserprop("lens_geometry_file", 1, lens_geometry_file_name)
        fdtd.adduserprop("focal_length", 2, self.FocalLength* self.um)
        fdtd.adduserprop("lens_radius", 2, self.LensRadius* self.um)
        # fdtd.adduserprop("substrate_index", 0, self.SubstrateIndex)
        # fdtd.adduserprop("pillar_index", 0, self.PillarIndex)



        myscript =f"""deleteall;
        np = round({self.LensRadius* self.um}/ {self.Pitch* self.um});
        x_mask = {self.Pitch* self.um} * (-np:1:np);
        y_mask = {self.Pitch* self.um} * (-np:1:np);

        addrect; # substrate
        set("name", "substrate");
        set("index",{self.SubstrateIndex});


        set("x span",3*{max(self.xcoord)});
        set("y span",3*{max(self.ycoord)});
        set("z max",0);
        set("z min",-2*{self.Wavelength* self.um});

        M=readdata(lens_geometry_file);
        len= length(M)/3;
        for(i=1:len) {{
            addcircle({{"name":"pillar", "x":M(i),"y":M(len+i),"radius":M(2*len+i)}});
        }}
        select("pillar");
        set("index",{self.PillarIndex});
        # set("material", "si");


        set("z min",0);
        set("z max",1.3e-6);
        """

        fdtd.set("script", myscript)

        # #==================================================
        # Add the aperture
        # #==================================================

        fdtd.add2dpoly()
        fdtd.set("name", "aperture")
        fdtd.set("z", -0.05e-6)

        # #==================================================
        # Add the source 
        # #==================================================
        fdtd.addplane()
        fdtd.set("name", "source")
        fdtd.set("injection axis","z")
        fdtd.set("direction","forward")
        fdtd.set("x",0)
        fdtd.set("x span",(self.LensDiameter+ 2* self.PMLDistance) * self.um)

        fdtd.set("y",0)
        fdtd.set("y span", (self.LensDiameter+ 2* self.PMLDistance)* self.um)

        fdtd.set("z",-0.15e-6) # change this later for automation

        fdtd.set("center wavelength", self.Wavelength* self.um)
        fdtd.set("wavelength span",0.0e-6)

        # # #==================================================
        # # Add the FIELD monitor
        # # #==================================================
        # fdtd.addpower()
        # fdtd.set("name","field")
        # fdtd.set("monitor type",7)  # 2D z-normal
        # fdtd.set("x",0)
        # fdtd.set("x span",LensDiameter+ 2* DeltaL)
        # fdtd.set("y",0)
        # fdtd.set("y span",LensDiameter+ 2* DeltaL)
        # fdtd.set("z",1.365e-6)
        # # #==================================================

        # #==================================================
        # Add the Transmission monitor
        # #==================================================
        fdtd.addprofile()
        fdtd.set("name","transmission")
        fdtd.set("monitor type",7)  # 2D z-normal
        fdtd.set("x",0)
        fdtd.set("x span", (self.LensDiameter+ 2* self.PMLDistance)* self.um)
        fdtd.set("y",0)
        fdtd.set("y span", (self.LensDiameter+ 2* self.PMLDistance)* self.um)
        fdtd.set("z",1.365e-6)
        fdtd.set("output Pz", True)

        fdtd.setglobalmonitor("frequency points", 1) # setting the global frequency resolution
        fdtd.save("lumerical_full_lens_test")
        fdtd.run()

        # ***************************************************************************************************

    def get_result_fdtd(self, show_plot= False):
        fdtd= lumapi.FDTD()
        
        
        fdtd.load("lumerical_full_lens_test.fsp")
        

        # #====================================================
        monitor_name="transmission"
        fdtd.getdata(monitor_name,"f")
        T1=fdtd.transmission(monitor_name) # transmission coefficient 
        print(f"normalized power through surface near the metalens (using direct transmission): {T1}")

        # #=====================================================
        # Calculate the power using Poynting vector Pz
        # #=====================================================
        # second way to calculate power through a surface - transmission
        x=fdtd.getdata(monitor_name,"x")
        y=fdtd.getdata(monitor_name,"y")
        f=fdtd.getdata(monitor_name,"f")
        Pz=fdtd.getdata(monitor_name,"Pz")
        SourcePower= fdtd.sourcepower(f)



        Pz_real= np.real(Pz)
        Pz_real= np.squeeze(Pz_real)
        x= np.squeeze(x)
        y= np.squeeze(y)


        T_Poynting= (utility.integrate2D(Pz_real, x, y)* 0.5)/ SourcePower # transmission coefficient calculated using a different method 
        print(f"normalized power through surface near the metalens (by using Poynting vector): {T_Poynting}")


        # #=====================================================
        # Calculate the Focal Length 
        # #=====================================================

        # choose area to plot and spatial resolution
        x = self.um* np.linspace(-self.LensRadius,self.LensRadius,200)
        y = self.um* np.linspace(-self.LensRadius,self.LensRadius,200)
        z =  np.linspace(1e-6,1.5*self.FocalLength* self.um,100)



        # get the focal length 
        farfield_E_along_z = fdtd.farfieldexact3d(monitor_name,0,0,z)
        farfield_E_along_z= np.squeeze(farfield_E_along_z)
        farfield_E2_z= abs(farfield_E_along_z)**2 # E^2
        farfield_E2_z= farfield_E2_z[:,0] # this is the actual farfield e2_z
        focal_calculated= z[np.where(farfield_E2_z== max(farfield_E2_z))[0]] # clauclated focal length 


        # #=====================================================
        # Calculate the FWHM of the beam 
        # #=====================================================
        # EFocal = fdtd.farfieldexact3d('field', x, y, focal_calculated, {"field":"E"})

        farfield_E_along_x_at_focal = fdtd.farfieldexact3d(monitor_name,x,0, focal_calculated)
        farfield_E_along_x_at_focal= np.squeeze(farfield_E_along_x_at_focal)
        farfield_E2_x= abs(farfield_E_along_x_at_focal)**2 # Ex^2
        farfield_E2_x= farfield_E2_x[:,0] # this is the actual farfield e2_x



        # Calculate initial guess for the parameters
        initial_amplitude = np.max(farfield_E2_x)  # Amplitude as the maximum value in the data
        initial_mean = x[np.argmax(farfield_E2_x)]  # Mean as the x-value at the maximum y-value
        initial_stddev = np.std(x)  # Standard deviation, can be adjusted based on data spread

        # Combine into an initial guess
        initial_guess = [initial_amplitude, initial_mean, initial_stddev]

        # Fit the Gaussian to the data
        popt, pcov = curve_fit(utility.gaussian, x, farfield_E2_x, p0=initial_guess)

        # Extract the fitted parameters
        amplitude, mean, stddev = popt

        # Calculate FWHM
        fwhm = 2 * np.sqrt(2 * np.log(2)) * stddev
        # #=====================================================
        # #=====================================================
        # Total power through the FOCAL PLANE
        # #=====================================================
        farfield_E_total= fdtd.farfieldexact3d(monitor_name, x, y, focal_calculated, {"field":"E"})
        farfield_E_total= np.squeeze(farfield_E_total)
        Ex= farfield_E_total[:,:,0]
        Ey= farfield_E_total[:,:,1]
        Ez= farfield_E_total[:,:,2]
        E_squared= abs(Ex)**2+ abs(Ey)**2



        TotalPower = 0.5* utility.integrate2D(E_squared, x, y)* np.sqrt(epsilon_0/mu_0) # total power through the focal plane 


        # #=====================================================
        # Total power through the FOCAL SPOT (2* FWHM radius)
        # #=====================================================

        FocalSpotRadius= 2* fwhm
        # Generate meshgrid
        X, Y = np.meshgrid(x, y)
        x_center= 1e-6 
        y_center= 1e-6 

        # Calculate the distance from the center
        distances = np.sqrt((X - x_center)**2 + (Y - y_center)**2)

        # Create the filter
        filter_mask = (FocalSpotRadius >= distances)

        # Apply the filter to I_Focal
        I_Focal_filtered = filter_mask *E_squared





        PowerAtFocalSpot=  (0.5* utility.integrate2D(I_Focal_filtered, x, y)* np.sqrt(epsilon_0/mu_0)) # total power at the focal spot 
        # #=====================================================
        # Efficiency Calculation
        # #=====================================================

        OverallEfficiency= PowerAtFocalSpot/ SourcePower 
        FocusingEfficiency= OverallEfficiency/ T_Poynting

        print(f"""Focal Length: {focal_calculated[0]* 1e6} μm, 
              \nFWHM: {fwhm* 1e6} μm, 
              \nOverall Efficiency:{OverallEfficiency}, 
              \nFocusing Efficiency: {FocusingEfficiency}, 
              \nTransmission Efficiency: {T1},
              \nTotalPower through focal plane/ source power: {TotalPower/SourcePower}""")
        
        # monitor_name = "transmission"
        # Ex = fdtd.getresult(monitor_name,"Ex")
        # Ey = fdtd.getresult(monitor_name,"Ey")
        # x= fdtd.getresult(monitor_name, "x")
        # y= fdtd.getresult(monitor_name, "y")
        # z= fdtd.getresult(monitor_name, "z")
        # return x,y,Ex 
        

        if show_plot:
            # #=====================================================
            # plot FWHM
            # #=====================================================
            plt.scatter(x*1e6, farfield_E2_x, label='X Intensity at the focal spot', color='lightblue')
            plt.plot(x*1e6, utility.gaussian(x, *popt), label='Fitted Gaussian', color='darkred', linewidth = 2)
            plt.xlabel('x (μm)')
            plt.ylabel('Intensity (arb. unit.)')
            plt.title(f"FWHM= {round(fwhm*1e6,3)} μm")
            plt.show()
            #=====================================================


            # #=====================================================
            # plot Intensity at focal plane 
            # #=====================================================
            # Intensity plot at the focal plane 
            extent = [np.min(x), np.max(x), np.min(y), np.max(y)]
            extent= [i* 1e6  for i in extent]
            plt.imshow(E_squared, cmap ="viridis", interpolation ='nearest', extent= extent)
            plt.xlabel('x (μm)')
            plt.ylabel('y (μm)')
            plt.title('Intensity at the focal plane')
            plt.show()


            # #=====================================================
            # Plot intensity along z-axis 
            # #=====================================================
            plt.plot(z*1e6, farfield_E2_z, linewidth = 2, color= 'darkgreen')
            plt.vlines(x= focal_calculated*1e6, ymin= min(farfield_E2_z), ymax=max(farfield_E2_z), linestyle= '--', color= 'black')
            plt.xlabel('z (μm)')
            plt.ylabel('Intensity (arb. unit.)')
            plt.show()