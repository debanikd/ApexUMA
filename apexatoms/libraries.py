import os,sys,time

from inkstone import Inkstone
import numpy as np
import matplotlib.pyplot as plt
from p_tqdm import p_map
import apexatoms.geometry as aag
from apexatoms.materials import Material
from apexatoms.atom import Atom
from apexatoms.helper import NumpyEncoder, setupMaterials, setupLayers
from itertools import product
import copy
import json

def makeAndCalcAtom(params,numBasis=151):
	atom = Atom(**params)
	results = atom.calcAtom(numBasis)
	return atom,results

class Library:

	def __init__(self,layers,pitch,geometry,wavelength,polarization,materials=None,grid="square"):

		
		self.layers = setupLayers(layers)
		self.geometry = geometry
		self.materials = setupMaterials(materials,self.layers,self.geometry)

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



		self.scan_parameters = [] # dicts indicating type of varying parameter, name, and the sweep range
	
		# Check to see which parameters should be scanned
		# This currently only supports a subset of parameters
		# * layer height
		# * lattice (primarily just square lattice pitch)
		# * geometric parameters (ex: diameter for circular atoms) for a single geometric type
		# * wavelength
		# Does not support
		# * Materials
		# * Polarization
		# * Mixing shapes/geometry

		# First check layer properties for varying layer height
		for name,layer in self.layers.items():
			if "thickness" in layer:
				if np.ndim(layer["thickness"])>0:
					self.scan_parameters.append({"type":"layer_thickness","layer":name,"values":layer["thickness"]})

		# Old code for variable lattice
		'''
		# Next check Lattice
		# Also add grid type and pitch (where possible) as top level parameter
		if np.ndim(lattice)==0: # Single Square Lattice
			self.pitch = lattice
			self.grid = "square"	
		if np.ndim(lattice)==1: # Single Rectangular or Variable Square Lattice
			if len(lattice)>2:
				self.pitch = lattice
				self.grid = "square"
				self.scan_parameters.append({"type":"lattice","shape":"square","values":lattice})
			else:
				self.grid = "rectangle"
		if np.ndim(lattice)==2: # Single flex/hex or Variable Rectangular Lattice
			if len(lattice)>2:
				self.grid = "rectangle"
				self.scan_parameters.append({"type":"lattice","shape":"rectangle","values":lattice})
			else:
				angle_1 = np.arctan(lattice[0][0][1]/lattice[0][0][0]) if lattice[0][0][0] else np.pi/2
				angle_2 = np.arctan(lattice[0][1][1]/lattice[0][1][0]) if lattice[0][1][0] else np.pi/2
				diff_angle = np.abs(angle_2-angle_1)
				if np.abs((diff_angle - 2*np.pi/3))<0.001 or np.abs((diff_angle - np.pi/3))<0.001:
					self.grid = "hex"
					self.pitch = np.sqrt(lattice[0][0]**2+lattice[0][1]**2)
				else:
					self.grid = "flex"

		if np.ndim(lattice)==3: # Variable Hex or Flex (freeform) lattice
			angle_1 = np.arctan(lattice[0][0][1]/lattice[0][0][0]) if lattice[0][0][0] else np.pi/2
			angle_2 = np.arctan(lattice[0][1][1]/lattice[0][1][0]) if lattice[0][1][0] else np.pi/2
			diff_angle = np.abs(angle_2-angle_1)
			if np.abs((diff_angle - 2*np.pi/3))<0.001 or np.abs((diff_angle - np.pi/3))<0.001:
				self.scan_parameters.append({"type":"lattice","shape":"hex","values":lattice})
				self.pitch = np.array([np.sqrt(v[0][0]**2+v[0][1]**2) for v in sp["values"]])
				self.grid = "hex"
			else:
				self.scan_parameters.append({"type":"lattice","shape":"flex","values":lattice})
				self.grid = "flex"
		'''

		# Next check pitch
		if np.ndim(pitch)>0:
			self.scan_parameters.append({"type":"pitch","values":pitch})

		# Next check geometric parameters
		for name,value in geometry["args"].items():
			if np.ndim(value)>0:
				self.scan_parameters.append({"type":"geometry","arg":name,"values":value})

		# Next check wavelength
		if np.ndim(wavelength)>0:
			self.scan_parameters.append({"type":"wavelength","values":wavelength})
		
		self.pitch = pitch
		self.wavelength = wavelength
		self.polarization = polarization
		self.grid = grid

		# Generate output shape
		self.shape = []
		for param in self.scan_parameters:
			self.shape.append(np.shape(param["values"])[0])

		self.transmission = np.zeros(self.shape)
		self.reflection = np.zeros(self.shape)
		self.phase = np.zeros(self.shape)

		# Generate arguments list
		self.args = list(product(*[np.arange(n) for n in self.shape]))

		self._generateParams()
		self._setParamVars()






	def _setParamVars(self):
		new_params = {}
		for sp in self.scan_parameters:
			if sp["type"] == "geometry":
				name = sp["arg"]
				value = sp["values"]
			# elif sp["type"] == "lattice":
			# 	name = "pitch"
			# 	if sp["shape"] == "square":
			# 		value = sp["values"]
			# 	elif sp["shape"] == "hex":
			# 		value = np.array([np.sqrt(v[0][0]**2+v[0][1]**2) for v in sp["values"]])
			elif sp["type"] == "layer_thickness":
				name = sp["layer"]+"_thickness"
				value = sp["values"]
			else:
				name = sp["type"]
				value = sp["values"]

			if not name in self.__dict__:
				if not name in new_params:
					new_params[name] = value

		self.__dict__.update(new_params)

	def _generateParams(self):
		# Generate Parameters
		self.params = []

		template_atom_params = {}
		template_atom_params["layers"] = self.layers
		template_atom_params["materials"] = self.materials
		template_atom_params["pitch"] = self.pitch
		template_atom_params["grid"] = self.grid
		template_atom_params["geometry"] = self.geometry
		template_atom_params["wavelength"] = self.wavelength
		template_atom_params["polarization"] = self.polarization

		# Only some aspects of this need a bespoke switch
		# Rewrite to allow more generic setting of parameters
		for arg in self.args:
			atom_params = copy.deepcopy(template_atom_params)
			for i,a in enumerate(arg):
				cur_param = self.scan_parameters[i]
				if cur_param["type"] == "layer_thickness":
					atom_params["layers"][cur_param["layer"]]["thickness"] = cur_param["values"][a]
				elif cur_param["type"] == "pitch":
					atom_params["pitch"] = cur_param["values"][a]
				elif cur_param["type"] == "geometry":
					atom_params["geometry"]["args"][cur_param["arg"]] = cur_param["values"][a]
				elif cur_param["type"] == "wavelength":
					atom_params["wavelength"] = cur_param["values"][a]
			self.params.append(atom_params)

	def calculate(self,numBasis=151):
		self.atoms = []
		for i,r in enumerate(p_map(makeAndCalcAtom,self.params,[numBasis]*len(self.params))):
			self.phase[self.args[i]] = r[1][0]
			self.transmission[self.args[i]] = r[1][1]
			self.reflection[self.args[i]] = r[1][2]
			self.atoms.append(r[0])

	def getData(self):
		save_dict = {}

		# Saves all object variables that do not start with an underscore 
		for v in vars(self):
			if not v.startswith('_'): 
				if not v == "atoms": # Don't save the actual atom objects
					if v == "params":
						save_dict[v] = np.array([a.getParams() for a in self.atoms])
					elif v=="materials":
						materials = {}
						for name,mat in vars(self)[v].items():
							materials[name] = mat.getData()
						# 	if isinstance(mat,Material):
						# 		materials.append({"name":mat.name,"index":mat(self.wavelength)})
						# 	else:
						# 		materials.append(mat)
						# 	# mat_dict[name] = mat(self.wavelength)
						save_dict[v] = materials
					elif v=="layers":
						layers = []
						for name,layer in vars(self)[v].items():
							if isinstance(layer["background"],Material):
								background = layer["background"].name
							else:
								background = layer["background"]

							if "thickness" in layer:
								thickness = layer["thickness"]
							else:
								thickness = 0

							layers.append({"name":name,
											"thickness" : thickness,
											"background" : background}
								)
							# layers_dict[name] = {"thickness" : layer["thickness"],
							# 						"background" : layer["background"].name,
							# 						"index" : layer["background"](self.wavelength)}
						save_dict[v] = layers
					elif v=="geometry":
						geometry_dict = {}
						for name,value in vars(self)[v].items():
							if name=='func':
								geometry_dict[name] = value.__name__
							elif name=='material':
								if isinstance(value,Material):
									geometry_dict[name] = value.name
								else:
									geometry_dict[name] = value
							else:
								geometry_dict[name] = value

						# geometry_dict["type"] = self.type
						save_dict[v] = geometry_dict
					else:
						save_dict[v] = vars(self)[v]
		return save_dict


	# Does not save atom objects
	def saveData(self,fileName,info=None):

		save_dict = self.getData()

		# If provided, also save an Info variable with user provided notes
		if info:
			save_dict['info'] = info

		# save_dict["atoms"] = self.atoms

		if fileName.endswith(".json"):
			with open(fileName,'w') as fp:
				json.dump(save_dict,fp,indent=4,cls=NumpyEncoder)
		elif fileName.endswith(".npy"):
			np.save(fileName,save_dict,allow_pickle=True)
		else:
			np.save(fileName+"_ApexLibrary.npy",save_dict,allow_pickle=True)

	def getResults(self):
		return self.phase, self.transmission, self.reflection

	def plot(self,fileName=None):
		if len(self.scan_parameters) == 1:

			plot_phase = np.unwrap(self.phase)
			plot_phase -= np.min(plot_phase)
			plot_phase /= (2*np.pi)

			fig = plt.figure()
			ax1 = fig.add_subplot(111)

			ax1.plot(self.scan_parameters[0]["values"],plot_phase,label="Induced Phase", color="tab:red")
			ax1.tick_params(axis='y', colors='tab:red', which="both")
			ax1.set_ylabel(r"Induced Phase ($\phi/2\pi$)")
			ax1.yaxis.label.set_color('tab:red')
			ax1.tick_params(axis='y', colors='tab:red', which="both")
			ax1.set_xlim([np.min(self.scan_parameters[0]["values"]), np.max(self.scan_parameters[0]["values"])])
			ax1.set_ylim(np.min(plot_phase),np.max(plot_phase))

			if np.max(plot_phase)>1:
				ax1.plot(self.scan_parameters[0]["values"],[1]*len(self.scan_parameters[0]["values"]),'--',color="tab:red",alpha=0.5)

			ax2 = ax1.twinx()
			ax2.plot(self.scan_parameters[0]["values"],self.transmission,label="Transmission", color="tab:blue")
			ax2.set_ylim(0,1)
			ax2.set_ylabel("Transmission")
			ax2.yaxis.label.set_color('tab:blue')
			ax2.tick_params(axis='y', colors='tab:blue', which="both")

			lines1, labels1 = ax1.get_legend_handles_labels()
			lines2, labels2 = ax2.get_legend_handles_labels()

			ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

			
			try: 
				ax1.set_xlabel(self.scan_parameters[0]["arg"])
			except:
				ax1.set_xlabel(self.scan_parameters[0]["type"])

			# plt.legend()
			plt.tight_layout()
			if fileName:
				plt.savefig(fileName)
				plt.close()
			else:
				plt.show()

		elif len(self.scan_parameters) == 2:
			plot_phase = np.unwrap(self.phase,axis=0)
			plot_phase = np.unwrap(plot_phase,axis=1)
			plot_phase -= np.min(plot_phase)
			plot_phase /= (2*np.pi)

			aspect = (np.min(self.scan_parameters[0]["values"])-np.max(self.scan_parameters[0]["values"]))/(np.min(self.scan_parameters[1]["values"])-np.max(self.scan_parameters[1]["values"]))
			extent = np.min(self.scan_parameters[0]["values"]),np.max(self.scan_parameters[0]["values"]),np.min(self.scan_parameters[1]["values"]),np.max(self.scan_parameters[1]["values"])



			fig, (ax1,ax2) = plt.subplots(2,1,figsize=(4,7))

			ppm = ax1.imshow(plot_phase, 
							aspect=aspect, 
							origin="lower", 
							extent = extent)

			ppt = ax2.imshow(self.transmission, 
							aspect=aspect, 
							origin="lower", 
							extent = extent,
							cmap = 'inferno',
							vmin=0,
							vmax=1)

			ax1.set_title(r"Induced Phase ($\phi/2\pi$)")
			ax2.set_title("Transmission")

			try: 
				name1 = self.scan_parameters[0]["arg"]
			except:
				name1 = self.scan_parameters[0]["type"]
			try: 
				name2 = self.scan_parameters[1]["arg"]
			except:
				name2 = self.scan_parameters[1]["type"]

			ax1.set_xlabel(name1)
			ax1.set_ylabel(name2)

			ax2.set_xlabel(name1)
			ax2.set_ylabel(name2)

			fig.colorbar(ppm,ax=ax1)
			fig.colorbar(ppt,ax=ax2)

			fig.subplots_adjust(hspace=0.4,right=0.926)

			if fileName:
				plt.savefig(fileName)
				plt.close()
			else:
				plt.show()


		else:
			print("Automatic plotting only supports up to 2 dimensional data.")





class SimpleLibrary(Library):


	def __init__(self,pitch,geometry,depth,wavelength,inIndex,outIndex,atomIndex,surroundIndex,grid):

		# materials = [
		# 		Material("inMat",inIndex),
		# 		Material("outMat",outIndex),
		# 		Material("atomMat",atomIndex),
		# 		Material("surroundMat",surroundIndex)]

		layers = [
					{"name":"input","background" : inIndex},
					{"name":"slab", "background" : surroundIndex, "thickness" : depth},
					{"name":"output", "background" : outIndex}
				]


		params = {
					"layers" : layers,
					# "materials" : materials,
					"pitch" : pitch,
					"geometry" : geometry,
					"wavelength" : wavelength,
					"polarization" : 'h',
					"grid" : grid
		}

		super().__init__(**params)


class SimpleCircleLibrary(SimpleLibrary):

	def __init__(self,pitch,diameter,depth,wavelength,inIndex=3.4699,outIndex=1.0,atomIndex=3.4699,surroundIndex=1.0,grid="square"):

		args = {"diameter" : diameter}
		func = aag.setCircleGeometry
		geometry = {"func" : func, "layer" : "slab", "material" : atomIndex, "args" : args}

		super().__init__(pitch=pitch,
							geometry=geometry,
							depth=depth,
							wavelength=wavelength,
							inIndex=inIndex,
							outIndex=outIndex,
							atomIndex=atomIndex,
							surroundIndex=surroundIndex,
							grid=grid)


class SimpleSquareLibrary(SimpleLibrary):

	def __init__(self,pitch,width,depth,wavelength,inIndex=3.4699,outIndex=1.0,atomIndex=3.4699,surroundIndex=1.0,grid="square"):

		args = {"width" : width}
		func = aag.setSquareGeometry
		geometry = {"func" : func, "layer" : "slab", "material" : atomIndex, "args" : args}

		super().__init__(pitch=pitch,
							geometry=geometry,
							depth=depth,
							wavelength=wavelength,
							inIndex=inIndex,
							outIndex=outIndex,
							atomIndex=atomIndex,
							surroundIndex=surroundIndex,
							grid=grid)