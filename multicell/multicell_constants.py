# lattice setting for the graph of cells
GRIDSIZE = 4
SEARCH_RADIUS_CELL = 1

# update order of cells and spins requires a seed, this fixed it at zero
DYNAMICS_FIXED_UPDATE_ORDER = True

# lattice dynamics
BLOCK_UPDATE_LATTICE = True
SEND_01 = False

# lattice initial condition
VALID_BUILDSTRINGS = ["mono", "dual", "memory_sequence", "random", "explicit"]
BUILDSTRING = "dual"

# simulation specific settings
NUM_LATTICE_STEPS = 20
LATTICE_PLOT_PERIOD = 10

# 'exosome' specific settings
# TODO: implement no_ext_field runtime savings
VALID_EXOSOME_STRINGS = ["on", "off", "all", "no_exo_field"]
EXOSTRING = "no_exo_field"
EXOSOME_REMOVE_RATIO = 0.0

# global signal range, all-connected cell-cell neighbour graph (except self-interactions)
MEANFIELD = False

# cells also signal with themselves (adjacency diagonals are "1")
# TODO implement autocrine in meanfield and non-parallel updating;
#  maybe in lastline get_surroundings_square (method of SpatialCell)
AUTOCRINE = False
