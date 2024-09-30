import os

Nthread = 1
os.environ["OMP_NUM_THREADS"] = str(Nthread) # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = str(Nthread) # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = str(Nthread) # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = str(Nthread) # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = str(Nthread) # export NUMEXPR_NUM_THREADS=1

from apexatoms.atom import Atom, SimpleCircleAtom, SimpleSquareAtom
from apexatoms.libraries import Library, SimpleCircleLibrary, SimpleSquareLibrary
from apexatoms.materials import Material
from apexatoms.layers import Layer
from apexatoms.geometry import *
from apexatoms.helper import *

__version__ = "1.1.0"
__author__ = "David Lombardo"
__company__ = "Apex MicroDevices"