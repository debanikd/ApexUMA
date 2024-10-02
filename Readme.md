### **README: ApexUMA (Apex Unified Metalens Application)**
#### **Version**: 1.0.0

---

### **Overview**

**What is it?** 

**Metalens** design workflow involves several steps 
1) Target Phase Design using Ray Optics Software such as Zemax,
2) Unit Cell Libray creation and Library Optimization for complex scenario such as achromatic response, 
3) Unit Cell to Full Lens Design conversion, 
4) Full Lens Simulation using FDTD software such as Lumerical FDTD, 
5) Final optical performance evaluation of the metalens using Zemax,
6) Export the finalized design to GDS Layout.

This rquires mannual transfer of data and running multiple softwares seperately. To bypass this unstructured approach, the ApexUMA is a **modularized** effort to streamline the process from phase design to fab-ready GDS layout. At this stage it is dependent on two third party softwares **Zemax OpticStudio** and **Lumerical FDTD**.

**Dependencies**:
* `numpy` 
* `scipy` 
* `matplotlib`
* `json` 
* `lumapi` (ensure it's linked properly to Lumerical)
* `ApexAtoms` (Apex inhouse unit cell simualtion framework)

### **Key Features**
- **ConfigReader**: A utility to load and manage configuration files in JSON format. This configuration file contains useful info about the lens such as Focal length, wavelength of operation, diameter etc.

- **Lumerical API Integration**: Connects Python to the Lumerical simulation platform using `lumapi`.

- **PhaseDesign Class**: 
   - Loads and processes optimized phase data from Zemax files.
   - Compares optimized phase with an ideal hyperbolic lens phase.

- **UnitCellDesign Class**: 
   - Creates a unit cell design library using parameters like wavelength, focal length, and material indices from the configuration file.
   - Provides an option to visualize phase vs. diameter relationships for unit cells.
- **UnitCellOptimize Class**: (TO BE ADDED )
- **LensDesign Class**:
   - Generates lens geometries based on optimized phase profiles and unit cell data.
   - Visualizes the lens design and saves the geometry to a text file.

- **FullLensSim Class**:
   - Sets up and runs FDTD simulations of the designed lens.
   - Configures the simulation environment, including lens geometry, sources, and monitors.
   - Get result from the FDTD simulation. Running FDTD and getting the results from FDTD are intentionally seperated for debugging purposes. 
   - **Transmission Coefficient Calculation**: Using both direct transmission and potentially Poynting vector methods (commented out). 
   - **Focal Length Calculation**:
    The focal length is calculated by analyzing the far-field electric field data along the z-axis and finding the point of maximum intensity.
    - **FWHM Calculation**:FWHM is calculated using the x-y intensity profile at the focal plane.
    - **Focusing and Overall Calculations**: 'Encircled Power' is defined as the the power concentrated in the circle around the focal spot with radius = 2* FWHM. The 'Overall Efficiency' is defined as the ratio of the 'Encircled Power' and the 'Source Power'. On the other hand, the 'Focusing Efficiency' is the ratio of the 'Overall Efficiecny' and the transmission efficiency. Overall Efficiency =  Transmission Efficiency $\times$ Focusing Efficiency.

   - **Plotting**:Visualization of the intensity profile, x-y Intensity Profile, and focal plane intensity.

---

### **Setup**

1. Ensure Python 3.x is installed.
2. Install required packages:

   ```bash
   pip install numpy scipy matplotlib lumapi
   ```

3. Make sure Lumerical API is installed, and its path is correctly appended in the script:
   
   ```python
   sys.path.append("C:\\Program Files\\Lumerical\\v242\\api\\python")

4. **ApexAtoms**: This should be distributed with the ApexUMA. (Installation may or may not be required.)




---

### **How to Use**

1. **Load Configurations** (Optional):

   ```python
   from ApexUMA.utility import ConfigReader

   # Load default config or custom path
   ConfigReader.load_config('your_config.json')
   config = ConfigReader.get_config()
   ```
   If no configuration file is mentioned the default configuration file name is chose. The default is ```current_directory + "\\LensConfiguration.json"```.

2. **Lumerical API**: The Lumerical path is set in the script; no further action is required if the API is installed properly. Lumerical version is mentioned in the system path ```sys.path.append("C:\\Program Files\\Lumerical\\v242\\api\\python")```.


3. **PhaseDesign Class**:

   Load and compare optimized phase data from Zemax:

   ```python
   phase_design = PhaseDesign()
   x, phase_zemax = phase_design.load_optimized_phase('path_to_zemax_file.txt')
   phase_design.compare_optimized_ideal()
   ```
   - Now the Zemax generated phase data has to be supplied manually - which should be **automated** later.

4. **UnitCellDesign Class**:

   Generate a unit cell library based on configuration parameters:

   ```python
   unit_cell_design = UnitCellDesign()
   diameter, phase, transmission = unit_cell_design.make_unit_cell_library(show_plot=True)
   ```
- This step depends on the package ApexAtoms package written by Dr. David Lombardo, Apex Microdevices.
- **(Not Implemented Yet)** In the case, one wants to provide the file name of the x,y,radius data instead of generating it using inbuilt ApexAtoms- one can accomplish that using 
    ```python 
    read_unit_cell_library(file_path= 'x_y_pillar_radius_data.txt')
    ```

5. **LensDesign Class**:

   Create and visualize lens geometry:

   ```python
   lens_design = LensDesign()
   x_python, y_python, rad_python = lens_design.make_lens_geometry(radius_zemax, phase_zemax, radius_unitcell, phase_unitcell, idealphase=True, show_lens=True)
   ```

6. **FullLensSim Class**:

   Run full FDTD simulations based on the lens design:

   ```python
   full_lens_sim = FullLensSim()
   full_lens_sim.run_fdtd(xcoord, ycoord, pillar_radius)
   ```


---

### **Debugging**
- Ensure Lumerical API is correctly installed.
- Verify the JSON config file structure if loading fails.
- The lens design function can visualize different phases (optimized or ideal) depending on the `idealphase` flag.
- Adjust the mesh refinement and simulation parameters according to your specific needs for accuracy and performance.

---
**Date Last Updated:**  September 30, 2024  






