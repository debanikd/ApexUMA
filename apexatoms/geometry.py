import numpy as np
# from inkstone import Inkstone

# Method to unpack and apply geometry dicts to a simulation
# returns Inkstone simulation object with geometry applied
# To do: handling if geometry is a list (multiple geometries added together)
def geometryHandler(sim,geometry_dict):
	layer = geometry_dict["layer"]
	material = geometry_dict["material"]
	func = globals()[geometry_dict["func"]] if isinstance(geometry_dict["func"],str) else geometry_dict["func"]
	args = geometry_dict["args"]

	if np.ndim(layer)>0:
		material = np.full(len(layer),material)
		for l,m in zip(layer,material):
			sim = func(sim=sim,layer=l,material=m.name,**args)
	else:
		sim = func(sim=sim,layer=layer,material=material.name,**args)

	return sim


	# check to see if multiple layers
	# check to see if multiple materials
	# Try to use something like np.full(layer.shape,material) to automatically expand?



# Method to add a circle geometry to a sim
# Returns the sim with added circle geometry
def addCircle(sim,layer,material,diameter):
	sim.AddPatternDisk(
		layer = layer,
		material = material,
		center = [0,0],
		radius = diameter/2
	)

	return sim

# Method to add a ellipse geometry to a sim
# Returns the sim with added ellipse geometry
# a is full width in X direction
# b is full width in Y direction
# theta is positive rotation from X axis in radians
def addEllipse(sim,layer,material,a,b,theta):
	sim.AddPatternEllipse(
		layer = layer,
		material = material,
		center = [0,0],
		half_lengths = (a/2,b/2),
		angle = theta
	)

	return sim

# Method to add a squarre geometry to a sim
# Calls the addRectangle function with 0 angle, and equal side lengths
# Returns the sim with added squarre geometry
def addSquare(sim,layer,material,width):
	return addRectangle(sim,width,width,0,layer,material)

# Method to add a rectangle geometry to a sim
# Returns the sim with added rectangle geometry
# a is full width in X direction
# b is full width in Y direction
# theta is positive rotation from X axis in radians
def addRectangle(sim,layer,material,a,b,theta,shift_x=0,shift_y=0):
	sim.AddPatternRectangle(
		layer = layer,
		material = material,
		center = [shift_x,shift_y],
		side_lengths = (a,b),
		angle = theta
	)

	return sim

# Method to add a polygon geometry to a sim
# Returns the sim with added polygon geometry
# points is a list of tuples of points
def addPolygon(sim,layer,material,points):
	sim.AddPatternPolygon(
		layer = layer,
		material = material,
		vertices = points
	)

	return sim

# Method to add a rectangle geometry to a sim
# Returns the sim with added rectangle geometry
# length_a/b is the total length of the cross bar in the X/Y direction
# width_a/b is the width of the width parallel to the X/Y direction
#       
#          __      ^^ width_b
#          |       ||
#          |       ||
# length_b |  ============ } width_a
#          |  ============ }
#          |       ||
#          __      ||
#            |--length_a--|
def addCrossBar(sim,layer,material,length_a,length_b,width_a,width_b):
	points = [(width_b/2,length_b/2),
				(width_b/2,width_a/2),
				(length_a/2,width_a/2),
				(length_a/2,-width_a/2),
				(width_b/2,-width_a/2),
				(width_b/2,-length_b/2),
				(-width_b/2,-length_b/2),
				(-width_b/2,-width_a/2),
				(-length_a/2,-width_a/2),
				(-length_a/2,width_a/2),
				(-width_b/2,width_a/2),
				(-width_b/2,length_b/2),
				(width_b/2,length_b/2)]
	points.reverse()
	sim = addPolygon(sim,layer,material,points)

	# sim = addRectangle(sim,length_a,width_a,0,layer,material)

	partial_length = (length_b-width_a)/2
	offset = partial_length/2+width_a/2

	# sim = addRectangle(sim,width_b,partial_length,0,layer,material,shift_y=-offset)

	# sim = addRectangle(sim,width_b,partial_length,0,layer,material,shift_y=offset)
	# sim = addRectangle(sim,length_a,width_a,0,layer,material)
	# sim = addRectangle(sim,width_b,partial_length,0,layer,material,shift_y=offset,shift_x=offset)
	# sim = addRectangle(sim,width_b,partial_length,0,layer,material,shift_y=-offset,shift_x=offset)
	# sim = addRectangle(sim,width_b,partial_length,0,layer,material,shift_y=offset,shift_x=-offset)
	# sim = addRectangle(sim,width_b,partial_length,0,layer,material,shift_y=-offset,shift_x=-offset)

	# sim = addRectangle(sim,width_b,length_b,0,layer,material)
	return sim

# Creates annular circle geometry
# If multiple diameters are added, adds them as concentric rings, alternating passed material & layer background material
def setCircleGeometry(sim,layer,material,diameter):
	if hasattr(diameter,"__len__"):
		bg_material = sim.layers[layer].material_bg
		for i,d in enumerate(diameter):
			if i % 2 == 0:
				if d:
					sim = addCircle(sim,layer,material,d)

			else:
				if d:
					sim = addCircle(sim,layer,bg_material,d)
	else:
		sim = addCircle(sim,layer,material,diameter)

	return sim


# Creates nested square geometry
# If multiple widths are added, adds them as nested sqaures, alternating atom and surround indices
def setSquareGeometry(sim,layer,material,width):
	if hasattr(width,"__len__"):
		bg_material = sim.layers[layer].material_bg
		for i,w in enumerate(width):
			if i % 2 == 0:
				if w:
					sim = addSquare(sim,layer,material,w)

			else:
				if w:
					sim = addSquare(sim,layer,bg_material,w)
	else:
		sim = addSquare(sim,layers,material,width)

	return sim