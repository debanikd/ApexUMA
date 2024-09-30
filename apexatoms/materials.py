import numpy as np
# Material object to store refractive index information
# Currently just supports static refractive indices
# Laying groundwork for future expansion
# Get index by calling the class itself
# To Do: add Sellmeier materials
# To Do: add ability to import n,k lists
# To Do: add built in materials for ease of use
# To Do: add handling to automatically use InkStone's 'vacuum' material
# To Do: add support for non-isotropic materials

# Material super class
class Material:

	def __init__(self,name):
		self.name = name

	@classmethod 
	def newMat(cls,*args,**kwargs):
		if not args:
			# check to see which **kwargs have been passed
			if "index" in kwargs:
				return StaticMaterial(**kwargs)
			elif "coeffs" in kwargs:
				return SellmeierMaterial(**kwargs)
			else:
				raise Exception("Missing \"index\" or \"coeffs\" arguments")

		else:
			if not isinstance(args[0],str):
				raise Exception("First argument must be material name.")
			if len(args)==1:
				# check to see which **kwargs have been passed
				if "index" in kwargs:
					return StaticMaterial(args[0],**kwargs)
				elif "coeffs" in kwargs:
					return SellmeierMaterial(args[0],**kwargs)
				else:
					raise Exception("Missing \"index\" or \"coeffs\" arguments")
			else:
				if isinstance(args[1],(int,float,complex)):
					return StaticMaterial(*args)
				elif isinstance(args[1],(list)):
					return SellmeierMaterial(*args)
				else:
					raise Exception("Second argument must be either static index value or list of sellmeier coefficients.")


	def __call__(self,wavelength):
		if np.ndim(wavelength)>0:
			return np.vectorize(self.getIndex)(wavelength)
		else:
			return self.getIndex(wavelength)

	def getData(self):
		data_dict = {}
		for v in vars(self):
			if not v.startswith('_'): 
				data_dict[v] = vars(self)[v]
		return data_dict


# Simple static index material
class StaticMaterial(Material):
	mat_type = "static"

	def __init__(self,name,index):
		self.name = name
		self.index = index

	# def getData(self):
	# 	return {"name":self.name,"type":"static","index":self.index}

	def getIndex(self,wavelength):
		return self.index

	def __str__(self):
		return self.name + ": static material with index " + str(self.index)

# Material defined by a list of pairs of coefficients for the Sellmeier equation
# Each pair is B and C for 
# n^2 = 1 + SUM(B lambda^2/(lambda^2 - C))
class SellmeierMaterial(Material):
	mat_type = "sellmeier"

	def __init__(self,name,coeffs):
		self.name = name
		self.coeffs = coeffs

	def getIndex(self,wavelength):
		return np.sqrt(1+np.sum([B*wavelength**2/(wavelength**2 - C**2) for B,C in self.coeffs]))

	def __str__(self):
		coeff_string = ""
		for i,(b,c) in enumerate(self.coeffs):
			coeff_string += f"\n\tB{i+1}: {b}\n"
			coeff_string += f"\tC{i+1}: {c}"

		return self.name + ": Sellmeier material with coefficients:" + coeff_string


# Material defined by imported n,k lists or file
# File should be a CSV or tab delimited file where the first column is wavelength in microns, then n, then k
# To Do: add support for different wavelength units
# class NKMaterial(Material):

# 	def __init__(self,name,file=None,waves=None,n=None,k=None):




# class Silicon(Material):

# 	def __init__(self):
# 		self.name="silicon"



# class Vacuum(Material):

# 	def __init__(self):
# 		self.name="vacuum"
# 		self.index=1
# 		self.type="static"