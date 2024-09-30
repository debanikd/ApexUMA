import os,sys,time

from inkstone import Inkstone
import numpy as np
import matplotlib.pyplot as plt
from p_tqdm import p_map
import apexatoms.geometry as aag
from apexatoms.materials import Material
from apexatoms.layers import Layer
import copy
from apexatoms.helper import setupMaterials, setupLayers


# Adds materials from materials dict into InkStone simulater, then returns the simulator
# Sets material epsilon based on passed wavelength
def setSimMaterials(sim,materials,wavelength):
	for name, mat in materials.items():
		sim.AddMaterial(name = name, epsilon = mat(wavelength)**2)

	return sim

# Adds layers from layers dict into InkStone simulater, then returns the simulator
def setLayers(sim,layers):
	for name,layer in layers.items():
		sim.AddLayer(name = name, thickness = layer["thickness"], material_background = layer["background"].name)

	return sim

# Sets the excitation of the sim
# Currently assumes normal incidence, no rotation
# polarization can be one of the following:
#	h = Horizontal - E Field Along X Axis
#	v = Vertical - E Field Along Y Axis
#	d = Diagonal -  E Field at Pi/4 from X Axis
#	a = Anti-Diagonal - E Field at -Pi/4 from X Axis
#	l = Left Hand Circular
#	r = Right Hand Circular
def setSimExcitation(sim,wavelength,polarization):
	# Set frequency of excitation
	sim.frequency = 1/wavelength

	# Set incoming polarization
	if polarization=='h': # Horizontal - E Field Along X Axis
		s = 0
		p = 1
	elif polarization=='v': # Vertical - E Field Along Y Axis
		s = 1
		p = 0
	elif polarization=='d': # Diagonal -  E Field at Pi/4 from X Axis
		s = 1/np.sqrt(2)
		p = 1/np.sqrt(2)
	elif polarization=='a': # Anti-Diagonal - E Field at -Pi/4 from X Axis
		s = -1/np.sqrt(2)
		p = 1/np.sqrt(2)
	elif polarization=='l': # Left Hand Circular
		s = -1j/np.sqrt(2)
		p = 1/np.sqrt(2)
	elif polarization=='r': # Right Hand Circular
		s = 1j/np.sqrt(2)
		p = 1/np.sqrt(2)

	sim.SetExcitation(
		theta=0,
		phi=0,
		s_amplitude=s,
		p_amplitude=p
	)

	return sim

# Takes h and v (x and y) polarization and returns diagonal and anti-diagonal
def getDiag(h,v):
	return (1/np.sqrt(2))*(h-v), (1/np.sqrt(2))*(h+v)

# Takes h and v (x and y) polarization and returns LCP and RCP
def getCirc(h,v):
	return (1/np.sqrt(2))*(h+1j*v), (1/np.sqrt(2))*(h-1j*v)

# # Applies the passed geometry function to the passed sim
# # Returns the sim with added geometry
# def setGeometry(sim,geometry):
# 	return geometry["func"](sim,geometry["layer"],geometry["material"],**geometry["args"])

def setNumBasis(sim,numBasis):
	sim.SetNumG(numBasis)
	return sim

# Calculates the fields and flux of the passed Inkstone simulator
# Returns fields, flux if return_flux is True
# Returns fields otherwise
# Returns raw fields, not adjusted for input polarization Ex, Ey, Ez, Hx, Hy, Hz
# Measures fields at twice the total stack thickness, usually means twice the atom depth/height
# This is to avoid near field effects
def calculateSim(sim,return_flux=True):
	fields = np.squeeze(sim.GetFields(x=0,y=0,z=2*sim.total_thickness))

	if return_flux:
		inc,refl = sim.GetPowerFlux(list(sim.layers)[0])
		trans,_ = sim.GetPowerFlux(list(sim.layers)[-1])

		T = trans.real/inc.real
		R = -refl.real/inc.real

		return fields, T, R

	else:
		return fields


# Plots XY and XZ slices of the atom for fields and intensity
# Should upate for specific polarizations and such, currently defaults to Ex
# Currently only works for 'simple' style atoms with input/slab/output layer structure
# Currently looks very wonky for non square grids.
def plotAtom(sim,polarization='h'):

		
		x_width = 1.15
		y_width = 1.15
		z_min = -sim.total_thickness*0.2
		z_max = 1.2*sim.total_thickness

		step = 0.01

		xs = np.arange(-x_width/2,x_width/2+step/2,step)
		ys = np.arange(-y_width/2,y_width/2+step/2,step)
		zs = np.arange(z_min,z_max+step/2,step)

		layer_eps = []
		layer_index = []
		layer_args = [np.argmin(np.abs(zs))]
		tot_thickness = 0
		for layer,value in sim.layers.items():
			xx, yy, eps, _ = sim.ReconstructLayer(layer,xs.size,ys.size)
			layer_eps.append(eps)
			layer_index.append(np.sqrt(np.abs(eps[:,:,0,0])).T)
			if value.thickness>0:
				tot_thickness += value.thickness
				layer_args += [np.argmin(np.abs(zs-tot_thickness))]
		layer_args += [np.argmin(np.abs(zs-sim.total_thickness))]

		xz_index = np.zeros((len(xs),len(zs)))

		# arg_low = 
		# arg_high = np.argmin(np.abs(zs-sim.total_thickness))

		y_ind = np.argmin(np.abs(ys))

		xz_index[:,0:layer_args[0]] = layer_index[0][:,y_ind,None]
		xz_index[:,layer_args[1]:] = layer_index[-1][:,y_ind,None]

		for i in range(1,len(sim.layers)-1):
			xz_index[:,layer_args[i-1]:layer_args[i]] = layer_index[i][:,y_ind,None]

		fields_xy = np.squeeze(sim.GetFields(x=xs,y=ys,z=sim.total_thickness/2))
		fields_xz = np.squeeze(sim.GetFields(x=xs,y=0,z=zs))

		if polarization=='h':
			plot_xy = np.real(fields_xy[0])
			plot_xz = np.real(fields_xz[0]).T
		elif polarization=='v':
			plot_xy = np.real(fields_xy[1])
			plot_xz = np.real(fields_xz[1]).T
		elif polarization=='a':
			phase = np.angle(getDiag(fields[0],fields[1])[0])
		elif polarization=='d':
			phase = np.angle(getDiag(fields[0],fields[1])[1])
		elif polarization=='l':
			phase = np.angle(getCirc(fields[0],fields[1])[0])
		elif polarization=='r':
			phase = np.angle(getCirc(fields[0],fields[1])[1])

		intensity_xy = np.squeeze(np.abs(fields_xy[0]**2)+np.abs(fields_xy[1]**2)+np.abs(fields_xy[2]**2))
		intensity_xz = np.squeeze(np.abs(fields_xz[0]**2)+np.abs(fields_xz[1]**2)+np.abs(fields_xz[2]**2)).T

		fig, axs = plt.subplots(3,2)

		axs[0,0].imshow(plot_xy,cmap="RdBu",origin="lower",extent=[-x_width/2,x_width/2,-y_width/2,y_width/2])
		axs[0,1].imshow(plot_xz,cmap="RdBu",origin="lower",extent=[-x_width/2,x_width/2,z_min,z_max],aspect=(x_width/(z_max-z_min)))
		axs[1,0].imshow(intensity_xy,cmap="inferno",origin="lower",extent=[-x_width/2,x_width/2,-y_width/2,y_width/2])
		axs[1,1].imshow(intensity_xz,cmap="inferno",origin="lower",extent=[-x_width/2,x_width/2,z_min,z_max],aspect=(x_width/(z_max-z_min)))

		axs[2,0].imshow(layer_index[1].T,cmap="Greys",origin="lower",extent=[-x_width/2,x_width/2,-y_width/2,y_width/2])
		axs[2,1].imshow(xz_index.T,cmap="Greys",origin="lower",extent=[-x_width/2,x_width/2,z_min,z_max],aspect=(x_width/(z_max-z_min)))

		axs[0,0].set_title("Ex Field - X/Y Slice")
		axs[0,1].set_title("Ex Field - X/Z Slice")
		axs[1,0].set_title("Intensity - X/Y Slice")
		axs[1,1].set_title("Intensity - X/Z Slice")
		axs[2,0].set_title("Index - X/Y Slice")
		axs[2,1].set_title("Index - X/Z Slice")

		axs[0,0].set_xlabel(r"X Axis ($\mu$m)")
		axs[0,1].set_xlabel(r"X Axis ($\mu$m)")
		axs[1,0].set_xlabel(r"X Axis ($\mu$m)")
		axs[1,1].set_xlabel(r"X Axis ($\mu$m)")
		axs[2,0].set_xlabel(r"X Axis ($\mu$m)")
		axs[2,1].set_xlabel(r"X Axis ($\mu$m)")

		axs[0,0].set_ylabel(r"Y Axis ($\mu$m)")
		axs[0,1].set_ylabel(r"Z Axis ($\mu$m)")
		axs[1,0].set_ylabel(r"Y Axis ($\mu$m)")
		axs[1,1].set_ylabel(r"Z Axis ($\mu$m)")
		axs[2,0].set_ylabel(r"Y Axis ($\mu$m)")
		axs[2,1].set_ylabel(r"Z Axis ($\mu$m)")

		plt.tight_layout()

		plt.show()




# Atom class
# Minimum needed to initialize is layers
# layers: dict of dicts, starting with input plane and ending with output plane, key is layer name
#				* background : name of material (in materials input), refractive index value, or Material object
#				* thickness : thickness of layer, ignored for first and last entries
#			i.e. {"input" : {"background : 3.4699}, "slab" : {"background" : 1, thickness : 6}, , "output" : {"background" : 1}}
#
# materials: dict of Material objects, or indices:
#			i.e. {"inIndex" : 3.4699, "outIndex" : 1}
#			In the future it's probably best to store the materials globally, rather than a copy in each atom
# geometry: dict with the following keys (in the future can do a list for adding geometry to multiple layers)
#				* func : function used to apply geometry of form func(sim,layer,**args) which returns the sim with added geometry
#				* layer : name of the layer the geometry should be applied to
#				* material : material for the geometry
#				* args : dict of arguments required by the geometry function
# wavelength: wavelength of value
# polarization: character from h,v,d,a,r,l 
#				currently nonfunctional, will be added back in later
class Atom:

	def __init__(self,layers,materials=None,pitch=None,geometry=None,wavelength=None,polarization=None,grid="square"):

		
		
		# if isinstance(self.geometry["material"],dict):
		# 	self.geometry["material"] = self.materials[self.geometry["material"]["name"]]

		

		# # Set up materials for the atom
		# # Currently no checks to see if a material already exists
		# if not materials is None: # if materials are provided, turn them into Material objects if necessary 
		# 	for m in materials:
		# 		if isinstance(m,Material):
		# 			self.materials[m.name]=m
		# 		else:
		# 			self.materials[m["name"]]=Material.newMat(m["name"],m["index"])
		# 	# for name,value in materials.items():
		# 	# 	if isinstance(value,Material):
		# 	# 		self.materials[name] = value
		# 	# 	else:
		# 	# 		self.materials[name] = Material(name,value)

		# Set up layers for the atom
		self.layers = setupLayers(layers)

		# for i, layer in enumerate(layers):
		# # for i, layer in enumerate(layers):
		# 	# if isinstance(layer,Layer)
		# 	self.layers[layer["name"]] = {} # Add new entry to Layers

		# 	# Set layer thickness
		# 	if (i==0) or (i==len(layers)-1): # If it's the first or last entry, set thickness to 0 (semi-infinite)
		# 		self.layers[layer["name"]]["thickness"] = 0

		# 	else: # Otherwise set to 'thickness' from input
		# 		self.layers[layer["name"]]["thickness"] = layer["thickness"]

		# 	# Set layer background material
		# 	# Handling different inputs is now done in _setupMaterials()
		# 	self.layers[layer["name"]]["background"] = layer["background"]


		# 	# # If it's already a Material object, just use that
		# 	# if isinstance(layer["background"],Material):
		# 	# 	if not layer["background"] in self.materials:
		# 	# 		self.materials[layer["background"].name]=layer["background"]
		# 	# 	self.layers[layer["name"]]["background"] = layer["background"]
		# 	# # If passed a number, create new Material named after that layer
		# 	# elif isinstance(layer["background"],(int, float, complex)):
		# 	# 	self.materials[layer["name"]] = Material.newMat(layer["name"],layer["background"])
		# 	# 	self.layers[layer["name"]]["background"] = self.materials[layer["name"]]
		# 	# # If passed a string, grab that Material
		# 	# # Does not currently fail gracefully
		# 	# elif isinstance(layer["background"],str):
		# 	# 	self.layers[layer["name"]]["background"] = self.materials[layer["background"]]
		# 	# else:
		# 	# 	print("Something went wrong with the materials.")
		
		self.geometry = geometry
		self.materials = setupMaterials(materials,self.layers,self.geometry)


		# # Check material in Geometry
		# if isinstance(self.geometry["material"],Material):
		# 	if not self.geometry["material"] in self.materials:
		# 		self.materials[self.geometry["material"].name]=self.geometry["material"]
		# elif isinstance(self.geometry["material"],str):
		# 	if not self.geometry["material"] in self.materials:
		# 		print("WARNING! Unknown material in geometry!")
		# 	else:
		# 		self.geometry["material"] = self.materials[self.geometry["material"]]


		# if np.ndim(lattice)==0: # If lattice is single value, set as square grid
		# 	self.lattice = ((lattice,0),(0,lattice))
		# elif np.ndim(lattice)==1: # If lattice is two values, set as rectangular grid
		# 	self.lattice = ((lattice[0],0),(0,lattice[1]))
		# else: # Otherwise use lattice as presented
		# 	self.lattice = lattice

		self.grid = grid
		self.pitch = pitch

		if grid=="hex":
			# self.lattice = [[3/2,0],[np.sqrt(3)/2,np.sqrt(3)]]*self.pitch/np.sqrt(3)
			# self.lattice = ((self.pitch*3/np.sqrt(3),0),(self.pitch*1/2,self.pitch*1))

			self.lattice = ((self.pitch*np.sqrt(3)/2,self.pitch*1/2),(self.pitch*np.sqrt(3)/2,self.pitch*-1/2))
			# self.lattice = ((self.pitch,0),(0,self.pitch))
		else:
			self.lattice = ((self.pitch,0),(0,self.pitch))

		# self.pitch = pitch
		self.wavelength = wavelength
		self.polarization = polarization

	# Set Atom excitation given wavelength and polarization
	# Defaults to horizontal polarization
	def setExcitation(self,wavelength,polarization='h'):
		self.wavelength = wavelength
		self.polarization = polarization


	def _baselinePhase(self):
		sim = Inkstone() # Create initial inkstone simulator
		sim = setNumBasis(sim,numBasis=1)
		sim.SetLattice(self.lattice) # Set square lattice based on pitch
		sim = setSimMaterials(sim,self.materials,self.wavelength)
		sim = setLayers(sim,self.layers)
		sim = setSimExcitation(sim,self.wavelength,self.polarization)
		fields = calculateSim(sim,return_flux=False)

		# Measure field matching input polarization
		# Might be best to break this off into a separate method
		if self.polarization=='h':
			phase = np.angle(fields[0])
		elif self.polarization=='v':
			phase = np.angle(fields[1])
		elif self.polarization=='a':
			phase = np.angle(getDiag(fields[0],fields[1])[0])
		elif self.polarization=='d':
			phase = np.angle(getDiag(fields[0],fields[1])[1])
		elif self.polarization=='l':
			phase = np.angle(getCirc(fields[0],fields[1])[0])
		elif self.polarization=='r':
			phase = np.angle(getCirc(fields[0],fields[1])[1])

		return phase

	# Return induced phase, transmission, and reflection of the Atom
	def calcAtom(self,numBasis=151,calc_baseline=True):

		if calc_baseline:
			baseline_phase = self._baselinePhase()
		else:
			baseline_phase = 0

		sim = Inkstone() # Create initial inkstone simulator
		sim = setNumBasis(sim,numBasis=numBasis)
		sim.SetLattice(self.lattice) # Set square lattice based on pitch
		sim = setSimMaterials(sim,self.materials,self.wavelength)
		sim = setLayers(sim,self.layers)
		sim = setSimExcitation(sim,self.wavelength,self.polarization)

		sim = aag.geometryHandler(sim,self.geometry) # Add geometry

		# # Add Geometry
		# # If geometry is passed as a string (as from the getParams method) look up the actual function in geometry
		# if isinstance(self.geometry["func"],str):
		# 	func = getattr(aag,self.geometry["func"])
		# 	sim = func(sim,layer=self.geometry["layer"],material=self.geometry["material"].name,**self.geometry["args"])
		# # Otherwise apply the function as is
		# else:
		# 	sim = self.geometry["func"](sim,layer=self.geometry["layer"],material=self.geometry["material"].name,**self.geometry["args"])
		
		fields, transmission, reflection = calculateSim(sim)
		# fields = calculateSim(sim,numBasis=numBasis)

		# Measure field matching input polarization
		# Might be best to break this off into a separate method
		if self.polarization=='h':
			phase = np.angle(fields[0])
		elif self.polarization=='v':
			phase = np.angle(fields[1])
		elif self.polarization=='a':
			phase = np.angle(getDiag(fields[0],fields[1])[0])
		elif self.polarization=='d':
			phase = np.angle(getDiag(fields[0],fields[1])[1])
		elif self.polarization=='l':
			phase = np.angle(getCirc(fields[0],fields[1])[0])
		elif self.polarization=='r':
			phase = np.angle(getCirc(fields[0],fields[1])[1])

		# Wrap phase within 0 to 2 Pi
		phase = (phase-baseline_phase)%(2*np.pi)

		return phase, transmission, reflection


	def plotAtom(self,numBasis=151):
		sim = Inkstone() # Create initial inkstone simulator
		sim = setNumBasis(sim,numBasis=numBasis)
		sim.SetLattice(self.lattice) # lattice based on pitch
		sim = setSimMaterials(sim,self.materials,self.wavelength)
		sim = setLayers(sim,self.layers)
		sim = setSimExcitation(sim,self.wavelength,self.polarization)
		sim = aag.geometryHandler(sim,self.geometry) # Add geometry
		# Add Geometry
		# If geometry is passed as a string (as from the getParams method) look up the actual function in geometry
		# if isinstance(self.geometry["func"],str):
		# 	func = getattr(aag,self.geometry["func"])
		# 	sim = func(sim,self.geometry["layer"],material=self.geometry["material"].name,**self.geometry["args"])
		# # Otherwise apply the function as is
		# else:
		# 	sim = self.geometry["func"](sim,layer=self.geometry["layer"],material=self.geometry["material"].name,**self.geometry["args"])
		
		plotAtom(sim,polarization=self.polarization)

	# Returns a dictionary of Atom parameters.
	def getParams(self):
		atom_parameters_dict = {}

		# Saves all object variables that do not start with an underscore 
		for v in vars(self):
			if (not v.startswith('_')) and (not v=="lattice"): 
				if v=="materials":
					materials = []
					for name,mat in vars(self)[v].items():
						materials.append(mat.getData())
						# mat_dict[name] = mat(self.wavelength)
					atom_parameters_dict[v] = materials
				elif v=="layers":
					layers = []
					for name,layer in vars(self)[v].items():
						layers.append({"name":name,
										"thickness" : layer["thickness"],
										"background" : layer["background"].name}
							)
						# layers_dict[name] = {"thickness" : layer["thickness"],
						# 						"background" : layer["background"].name,
						# 						"index" : layer["background"](self.wavelength)}
					atom_parameters_dict[v] = layers
				elif v=="geometry":
					geometry_dict = {}
					for name,value in vars(self)[v].items():
						if name=='func':
							if isinstance(value,str):
								geometry_dict[name] = value
							else:
								geometry_dict[name] = value.__name__
						elif name=='material':
							geometry_dict[name] = value.name
						else:
							geometry_dict[name] = value

					# geometry_dict["type"] = self.type
					atom_parameters_dict[v] = geometry_dict

				else:
					atom_parameters_dict[v] = vars(self)[v]

		return atom_parameters_dict

	# This could be simplified by folding the if/elif/else type checking into a sub function
	def _setupMaterials(self,materials,layers,geometry):
		material_dict = {}

		# First look through input materials
		if not materials is None:
			if isinstance(materials,dict):
				for name,mat in materials.items():
					if isinstance(mat,Material):
						material_dict[name] = mat
					elif isinstance(mat,(int, float, complex)):
						material_dict[name] = Material.newMat(name,mat)
					else:
						raise Exception("Unknown material type!")
			elif isinstance(materials,list):
				for mat in materials:
					if isinstance(mat,Material):
						material_dict[mat.name] = mat
					elif isinstance(mat,dict):
						material_dict[mat["name"]] = Material.newMat(**mat)
					else:
						raise Exception("Unknown material type!")
			else:
				raise Exception("Unknown material formatting!")

		# Then check each layer
		for name, layer in layers.items():
			if isinstance(layer["background"],Material):
				material_dict[layer["background"].name] = layer["background"]
			elif isinstance(layer["background"],(int, float, complex,list)):
				material_dict[name] = Material.newMat(name,layer["background"])
				layer["background"] = material_dict[name]
			elif isinstance(layer["background"],str): # If material passed as string name, replace with reference to actual Material object
				try: 
					layer["background"] = material_dict[layer["background"]]
				except:
					raise Exception("Layer " + name + " background material not found!")
		
		# Then check the geometry
		if isinstance(geometry["material"],Material):
			material_dict[geometry["material"].name] = geometry["material"]
		elif isinstance(geometry["material"],(int, float, complex)): # gotta fix this to check if it's a list of materials, or a list of coefficients
			material_dict[geometry["layer"]+"_geometry_mat"] = Material.newMat(geometry["layer"]+"_geometry_mat",geometry["material"])
			geometry["material"] = material_dict[geometry["layer"]+"_geometry_mat"]
		elif isinstance(geometry["material"],str):
			try:
				geometry["material"] = material_dict[geometry["material"]]
			except:
				raise Exception("Geometry material not found!")
		# Need to add handling here to allow coeff lists to be turned into Materials
		# Or just make it so you can't generate new materials from inside the Geometry input
		# That option is probably much easier
		elif isinstance(geometry["material"],(list,tuple)):	
			for i,mat in enumerate(geometry["material"]):
				if isinstance(mat,Material):
					material_dict[mat.name] = mat
				elif isinstance(mat,str):
					try:
						geometry["material"][i] = material_dict[mat]
					except:
						raise Exception("Geometry material not found!")
		return material_dict




# Class for generating simple atoms with three layers: input/slab/output
# For replicating prior ApexAtoms behavior
class SimpleAtom(Atom):

	def __init__(self,pitch,func,args,depth,wavelength,polarization,inIndex,outIndex,atomIndex,surroundIndex,grid="square"):

		layers = [
					{"name":"input","background" : inIndex},
					{"name":"slab", "background" : outIndex, "thickness" : depth},
					{"name":"output","background" : surroundIndex}
				]

		# materials = [Material("inMat",inIndex), Material("outMat",outIndex), Material("atomMat",atomIndex), Material("surroundMat",surroundIndex)]

		geometry = {"func" : func, "layer" : "slab", "material" : atomIndex, "args" : args}

		super().__init__(
							layers=layers,
							# materials=materials,
							pitch=pitch,
							geometry=geometry,
							wavelength=wavelength,
							polarization=polarization,
							grid=grid)




# A class for generating a simple circular atom structure with three layers
class SimpleCircleAtom(SimpleAtom):

	def __init__(self,pitch,diameter,depth,wavelength,inIndex,outIndex,atomIndex,surroundIndex,grid="square"):


			args = {"diameter" : diameter}
			func = aag.setCircleGeometry

			super().__init__(pitch=pitch,
								func=func,
								args=args,
								depth=depth,
								wavelength=wavelength,
								polarization='h',
								inIndex=inIndex,
								outIndex=outIndex,
								atomIndex=atomIndex,
								surroundIndex=surroundIndex,
								grid=grid)

			# self.type="circular"


class SimpleSquareAtom(SimpleAtom):

	def __init__(self,pitch,width,depth,wavelength,inIndex,outIndex,atomIndex,surroundIndex,grid="square"):


			args = {"width" : width}
			func = aag.setSquareGeometry

			super().__init__(pitch=pitch,
								func=func,
								args=args,
								depth=depth,
								wavelength=wavelength,
								inIndex=inIndex,
								outIndex=outIndex,
								atomIndex=atomIndex,
								surroundIndex=surroundIndex,
								grid=grid)

			# self.type="square"


def calcAtomFromParams(inputs):
			cur_atom = inputs["atom_obj"](**inputs["params"])
			# cur_atom.addGeometry()
			return cur_atom.calcAtom(numBasis=inputs["numBasis"])


if __name__ == "__main__":
	print("Running on main.")
	# Stuff would go here