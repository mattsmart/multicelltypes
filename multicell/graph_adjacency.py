import matplotlib.pyplot as plt
import numpy as np

from multicell.multicell_constants import VALID_EXOSOME_STRINGS, EXOSTRING
from singlecell.singlecell_functions import state_subsample, state_only_on, state_only_off


def lattice_square_loc_to_int(loc, sidelength):
    # maps a two-tuple, for the location of a cell on square grid, to a unique integer
    # sidelength is sqrt(num_cells), the edge length of the lattice
    assert 0 <= loc[0] < sidelength
    assert 0 <= loc[1] < sidelength
    x, y = loc[0], loc[1]
    return x * sidelength + y


def lattice_square_int_to_loc(node_idx, sidelength):
    # maps node_idx, the unique int rep of a cell location on the grid, to corresponding two-tuple
    # sidelength is sqrt(num_cells), the edge length of the lattice
    y = node_idx % sidelength              # remainder from the division mod n
    x = int((node_idx - y) / sidelength)   # solve for x
    return x, y


def adjacency_lattice_square(sidelength, num_cells, search_radius, periodic_bc=False):
    """
    periodic_bc: wrap around boundary condition (False default)
    """
    assert num_cells == sidelength ** 2
    adjacency_arr_uptri = np.zeros((num_cells, num_cells))
    # build only upper diagonal part of A
    for a in range(num_cells):
        grid_loc_a = lattice_square_int_to_loc(a, sidelength)  # map cell id to grid loc (i, j)
        arow, acol = grid_loc_a[0], grid_loc_a[1]
        arow_low = arow - search_radius
        arow_high = arow + search_radius
        acol_low = acol - search_radius
        acol_high = acol + search_radius
        for b in range(a + 1, num_cells):
            grid_loc_b = lattice_square_int_to_loc(b, sidelength)  # map cell id to grid loc (i, j)
            # is the cell a neighbor?
            if (arow_low <= grid_loc_b[0] <= arow_high) and \
                    (acol_low <= grid_loc_b[1] <= acol_high):
                adjacency_arr_uptri[a, b] = 1

    if periodic_bc:

        def build_nbr_ints(arow, acol):
            nbr_locs = []
            search_ints = range(-search_radius, search_radius + 1)
            for i in search_ints:
                x = (arow + i) % sidelength
                for j in search_ints:
                    y = (acol + j) % sidelength
                    if not (i == 0 and j == 0):
                        loc = (x, y)
                        grid_int = lattice_square_loc_to_int(loc, sidelength)
                        nbr_locs.append(grid_int)
            return nbr_locs

        for a in range(num_cells):
            grid_loc_a = lattice_square_int_to_loc(a, sidelength)  # map cell id to grid loc (i, j)
            arow, acol = grid_loc_a[0], grid_loc_a[1]
            nbr_ints = build_nbr_ints(arow, acol)
            for b in range(a + 1, num_cells):
                if b in nbr_ints:
                    adjacency_arr_uptri[a, b] = 1

    adjacency_arr_lowtri = adjacency_arr_uptri.T
    adjacency_arr = adjacency_arr_lowtri + adjacency_arr_uptri
    return adjacency_arr


def adjacency_general(num_cells):
    # TODO implement
    return None


def general_paracrine_field(multicell, receiver_idx, step, flag_01=False, neighbours=None):
    if neighbours is None:
        graph_neighbours_col = multicell.matrix_A[:, receiver_idx]
        neighbours = [idx for idx, i in enumerate(graph_neighbours_col) if i == 1]

    sent_signals = np.zeros(multicell.num_genes)
    for loc in neighbours:
        nbr_cell_state = multicell.get_cell_state(loc, step)
        if flag_01:
            # convert to 0, 1 rep for biological dot product below
            nbr_cell_state_sent = (nbr_cell_state + 1) / 2.0
        else:
            nbr_cell_state_sent = nbr_cell_state
        sent_signals += np.dot(multicell.matrix_W, nbr_cell_state_sent)
    return sent_signals


def general_exosome_field(multicell, receiver_idx, step, neighbours=None):
    """
    Generalization of `get_local_exosome_field(self, ...)` in multicell_spatialcell.py
        A - sample from only 'on' genes (similar options of 'off', 'all')
        B - 'no_exo_field' will return all np.zeros(num_genes) for the field
    Returns:
        (unscaled) exosome field, neighbours
    """
    exosome_string = multicell.exosome_string
    exosome_remove_ratio = multicell.exosome_remove_ratio

    if neighbours is None:
        graph_neighbours_col = multicell.matrix_A[:, receiver_idx]
        neighbours = [idx for idx, i in enumerate(graph_neighbours_col) if i == 1]

    field_state = np.zeros(multicell.num_genes)
    if exosome_string == "on":
        for loc in neighbours:
            nbr_cell_state = np.zeros(multicell.num_genes)
            nbr_cell_state[:] = multicell.get_cell_state(loc, step)[:]
            nbr_state_only_on = state_only_on(nbr_cell_state)
            if exosome_remove_ratio == 0.0:
                field_state += nbr_state_only_on
            else:
                nbr_state_only_on = state_subsample(
                    nbr_state_only_on, ratio_to_remove=exosome_remove_ratio)
                field_state += nbr_state_only_on
    elif exosome_string == "all":
        for loc in neighbours:
            nbr_cell_state = np.zeros(multicell.num_genes)
            nbr_cell_state[:] = multicell.get_cell_state(loc, step)[:]
            if exosome_remove_ratio == 0.0:
                field_state += nbr_cell_state
            else:
                nbr_state_subsample = state_subsample(
                    nbr_cell_state, ratio_to_remove=exosome_remove_ratio)
                field_state += nbr_state_subsample
    elif exosome_string == "off":
        for loc in neighbours:
            nbr_cell_state = np.zeros(multicell.num_genes)
            nbr_cell_state[:] = multicell.get_cell_state(loc, step)[:]
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


if __name__ == '__main__':
    sidelength = 10
    num_cells = sidelength ** 2
    search_radius = 1
    A = adjacency_lattice_square(sidelength, num_cells, search_radius, periodic_bc=True)
    plt.imshow(A)
    plt.show()
    for i in range(num_cells):
        print(np.sum(A[i,:]))

        nbr_grid = np.ones((sidelength, sidelength)) - 2
        ax, ay = lattice_square_int_to_loc(i, sidelength)
        print(ax, ay)
        for j in range(num_cells):
            bx, by = lattice_square_int_to_loc(j, sidelength)
            nbr_grid[bx, by] = A[i, j]
        nbr_grid[ax, ay] = -1

        plt.imshow(nbr_grid)
        plt.colorbar()
        plt.show()



