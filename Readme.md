### **README: ApexUMA**
#### **Version**: 1.0.0

---

### **Overview**

**What is it?**

**Dependencies**:
* `numpy` 
* `scipy` 
* `matplotlib`
* `json` 
* `lumapi` (ensure it's linked properly to Lumerical)

### **Key Features**
- **ConfigReader**: A utility to load and manage configuration files in JSON format.
- **Lumerical API Integration**: Connects Python to the Lumerical simulation platform using `lumapi`.
- **Default Configuration**: Automatically uses a default configuration if none is provided.


- **PhaseDesign Class**: 
   - Loads and processes optimized phase data from Zemax files.
   - Compares optimized phase with an ideal hyperbolic lens phase.

- **UnitCellDesign Class**: 
   - Creates a unit cell design library using parameters like wavelength, focal length, and material indices from the configuration file.
   - Provides an option to visualize phase vs. diameter relationships for unit cells.
- **LensDesign Class**:
   - Generates lens geometries based on optimized phase profiles and unit cell data.
   - Visualizes the lens design and saves the geometry to a text file.

- **FullLensSim Class**:
   - Sets up and runs FDTD simulations of the designed lens.
   - Configures the simulation environment, including lens geometry, sources, and monitors.
   - Get result from the FDTD simulation. Run FDTD and Get result FDTD are intentionally seperated. 
   - Transmission Coefficient Calculation: Using both direct transmission and potentially Poynting vector methods (commented out).
    Outputting the normalized power through the surface near the metalens.
   - Focal Length Calculation:
    The focal length is calculated by analyzing the far-field electric field data along the z-axis and finding the point of maximum intensity.
    - FWHM Calculation:A Gaussian fitting approach to determine the beam's Full Width at Half Maximum (FWHM) using curve_fit from scipy.optimize.
    - Power Calculations: Calculation of total power through the focal plane and at the focal spot.
    Efficiency calculations based on the power through the focal spot and source power.
   - Plotting:Visualization of the intensity profile, Gaussian fit, and focal plane intensity.

---

### **Setup**

1. Ensure Python 3.x is installed.
2. Install required packages:

   ```bash
   pip install numpy scipy matplotlib apexatoms lumapi
   ```

3. Make sure Lumerical API is installed, and its path is correctly appended in the script:
   
   ```python
   sys.path.append("C:\\Program Files\\Lumerical\\v242\\api\\python")

4. **Install ApexAtoms**:
   
   ```bash
   pip install apexatoms
   ```

5. **Ensure Zemax data files** are available for phase design.


---

### **How to Use**

1. **Load Configurations**:

   ```python
   from ApexUMA.utility import ConfigReader

   # Load default config or custom path
   ConfigReader.load_config('your_config.json')
   config = ConfigReader.get_config()
   ```

2. **Lumerical API**: The Lumerical path is set in the script; no further action is required if the API is installed properly.


3. **PhaseDesign Class**:

   Load and compare optimized phase data from Zemax:

   ```python
   phase_design = PhaseDesign()
   x, phase_zemax = phase_design.load_optimized_phase('path_to_zemax_file.txt')
   phase_design.compare_optimized_ideal()
   ```

4. **UnitCellDesign Class**:

   Generate a unit cell library based on configuration parameters:

   ```python
   unit_cell_design = UnitCellDesign()
   diameter, phase, transmission = unit_cell_design.make_unit_cell_library(show_plot=True)
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






