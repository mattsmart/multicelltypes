import numpy as np

from multicell.graph_adjacency import lattice_square_loc_to_int
from multicell.multicell_constants import VALID_EXOSOME_STRINGS, EXOSTRING
from singlecell.singlecell_constants import BETA, FIELD_SIGNAL_STRENGTH, FIELD_APPLIED_STRENGTH
from singlecell.singlecell_class import Cell
from singlecell.singlecell_functions import state_subsample, state_only_on, state_only_off


class SpatialCell(Cell):
    def __init__(self, state, label, location, simsetup, state_array=None, steps=None):
        memories_list = simsetup['CELLTYPE_LABELS']
        gene_list = simsetup['GENE_LABELS']
        Cell.__init__(self, state, label, memories_list, gene_list, state_array=state_array, steps=steps)
        self.location = location

    def get_spatial_location_int(self, n):
        # n is sqrt(num_cells), the edge length of the lattice
        return lattice_square_loc_to_int(self.location, n)

    def get_surroundings_square(self, search_radius, gridsize):
        """Specifies the location of the top left corner of the search square
        Args:
            search_radius: half-edge length of the square
        Returns:
            list of locations; length should be (2 * search_radius + 1) ** 2 (- 1 remove self?)
        Notes:
            - periodic BCs apply, so search boxes wrap around at boundaries
            - note that we assert that search_radius be less than half the grid size
            - may have different search radius depending om context (neighboring bacteria/empty cell)
            - currently DOES NOT remove the original location
        """
        row = self.location[0]
        col = self.location[1]
        surroundings = [[row_to_search % gridsize, col_to_search % gridsize]
                        for row_to_search in range(row - search_radius, row + search_radius + 1)
                        for col_to_search in range(col - search_radius, col + search_radius + 1)]
        surroundings.remove(self.location)
        return surroundings

    def get_local_paracrine_field(self, lattice, neighbour_locs, simsetup, flag_01=False):
        sent_signals = np.zeros(self.N)
        for loc in neighbour_locs:
            nbr_cell_state = lattice[loc[0]][loc[1]].get_current_state()
            if flag_01:
                # convert to 0, 1 rep for biological dot product below
                nbr_cell_state_sent = (nbr_cell_state + 1) / 2.0
            else:
                nbr_cell_state_sent = nbr_cell_state
            sent_signals += np.dot(simsetup['FIELD_SEND'], nbr_cell_state_sent)
        return sent_signals

    def get_local_exosome_field(self, lattice, search_radius, gridsize, exosome_string=EXOSTRING,
                                exosome_remove_ratio=0.0, neighbours=None):
        """
        # TODO: try other methods, currently sample from on genes in nearby states
        A - sample from only 'on' genes
        B - sample from whole gene state vector
        """
        if neighbours is None:
            neighbours = self.get_surroundings_square(search_radius, gridsize)
        field_state = np.zeros(self.N)
        if exosome_string == "on":
            for loc in neighbours:
                nbr_cell_state = np.zeros(self.N)
                nbr_cell_state[:] = lattice[loc[0]][loc[1]].get_current_state()[:]
                nbr_state_only_on = state_only_on(nbr_cell_state)
                if exosome_remove_ratio == 0.0:
                    field_state += nbr_state_only_on
                else:
                    nbr_state_only_on = state_subsample(
                        nbr_state_only_on, ratio_to_remove=exosome_remove_ratio)
                    field_state += nbr_state_only_on
        elif exosome_string == "all":
            for loc in neighbours:
                nbr_cell_state = np.zeros(self.N)
                nbr_cell_state[:] = lattice[loc[0]][loc[1]].get_current_state()[:]
                if exosome_remove_ratio == 0.0:
                    field_state += nbr_cell_state
                else:
                    nbr_state_subsample = state_subsample(
                        nbr_cell_state, ratio_to_remove=exosome_remove_ratio)
                    field_state += nbr_state_subsample
        elif exosome_string == "off":
            for loc in neighbours:
                nbr_cell_state = np.zeros(self.N)
                nbr_cell_state[:] = lattice[loc[0]][loc[1]].get_current_state()[:]
                nbr_state_only_off = state_only_off(nbr_cell_state)
                if exosome_remove_ratio == 0.0:
                    field_state += nbr_state_only_off
                else:
                    nbr_state_only_off = state_subsample(
                        nbr_state_only_off, ratio_to_remove=exosome_remove_ratio)
                    field_state += nbr_state_only_off
        else:
            if exosome_string != "no_exo_field":
                raise ValueError("exosome_string arg invalid, must be one of %s" % VALID_EXOSOME_STRINGS)
        return field_state, neighbours

    def update_with_signal_field(
            self, lattice, search_radius, gridsize, intxn_matrix, simsetup, beta=BETA,
            exosome_string=EXOSTRING, exosome_remove_ratio=0.0,
            field_signal_strength=FIELD_SIGNAL_STRENGTH,
            field_app=None, field_app_strength=FIELD_APPLIED_STRENGTH):
        ext_field, neighbours = self.get_local_exosome_field(lattice, search_radius, gridsize, exosome_string=exosome_string, exosome_remove_ratio=exosome_remove_ratio)
        if simsetup['FIELD_SEND'] is not None:
            ext_field += self.get_local_paracrine_field(lattice, neighbours, simsetup)
        self.update_state(beta=beta, intxn_matrix=intxn_matrix, field_signal=ext_field, field_signal_strength=field_signal_strength, field_applied=field_app, field_applied_strength=field_app_strength)

    def update_with_meanfield(self, intxn_matrix, field_signal_mean, beta=BETA, app_field=None,
                              field_signal_strength=FIELD_SIGNAL_STRENGTH,
                              field_app_strength=FIELD_APPLIED_STRENGTH):
        self.update_state(beta=beta, intxn_matrix=intxn_matrix, field_signal=field_signal_mean, field_signal_strength=field_signal_strength, field_applied=app_field, field_applied_strength=field_app_strength)
