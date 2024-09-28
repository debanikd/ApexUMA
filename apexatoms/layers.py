from apexatoms.materials import Material
# Layer object to help construct simulations
# Laying groundwork for future expansion
# Not currently used
class Layer:

	def __init__(self,name,thickness,material):
		self.name=name
		self.thickness=thickness
