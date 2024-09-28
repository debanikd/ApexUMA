import numpy as np
import json
from apexatoms.materials import Material

# https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

# Loads a ApexAtoms library file and returns a python dict
def loadLibrary(fileName):
	if fileName.endswith(".json"):
		f = open(fileName)
		# print(f)
		loaded_json = json.load(f)

		# Turn lists back into numpy arrays
		def evalDict(cur_dict):
			for key,value in cur_dict.items():
				if isinstance(value,dict):
					evalDict(value)
				elif isinstance(value,list):
					value = np.asarray(value)

		evalDict(loaded_json)

		return loaded_json

	elif fileName.endswith(".npy"):
		return np.load(fileName,allow_pickle=True).item()

def setupLayers(layer_list):

	if isinstance(layer_list,list):
		layers = {}
		for i, layer in enumerate(layer_list):
		# for i, layer in enumerate(layers):
			# if isinstance(layer,Layer)
			layers[layer["name"]] = {} # Add new entry to Layers
			# Set layer thickness
			if (i==0) or (i==len(layer_list)-1): # If it's the first or last entry, set thickness to 0 (semi-infinite)
				layers[layer["name"]]["thickness"] = 0
			else: # Otherwise set to 'thickness' from input
				layers[layer["name"]]["thickness"] = layer["thickness"]
			# Set layer background material
			# Handling different inputs is now done in _setupMaterials()
			layers[layer["name"]]["background"] = layer["background"]
		return layers
	elif isinstance(layer_list,dict):
		return layer_list


# Set up dict of Material objects based on input
# Pulls materials out of layers & geometries and generates them if needed
def setupMaterials(materials,layers,geometry):
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
