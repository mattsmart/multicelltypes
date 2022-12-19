import numpy as np
import os

from singlecell.singlecell_functions import hamiltonian
from multicell.graph_adjacency import general_paracrine_field, general_exosome_field


def calc_lattice_energy(lattice, simsetup, field, fs, gamma, search_radius, exosome_remove_ratio,
                        exosome_string, meanfield, norm=True):
    """
    Lattice energy is the multicell hamiltonian
        H_multi = [Sum (H_internal)] - gamma * [Sum(interactions)] - fs * [app_field dot Sum(state)]
    Returns total energy and the two main terms
    """
    M1 = len(lattice)
    M2 = len(lattice[0])
    num_cells = M1 * M2
    assert M1 == M2  # TODO relax
    H_multi = 0
    H_self = 0
    H_pairwise = 0
    H_app = 0
    # compute self energies and applied field contribution separately
    for i in range(M1):
        for j in range(M2):
            cell = lattice[i][j]
            H_self += hamiltonian(cell.get_current_state(), simsetup['J'], field=None, fs=0.0)
            if field is not None:
                H_app -= fs * np.dot(cell.get_current_state().T, field)
    # compute interactions
    # meanfield case
    if meanfield:
        mf_search_radius = None
        mf_neighbours = [[a, b] for a in range(M2) for b in range(M1)]
    else:
        assert search_radius is not None
    for i in range(M1):
        for j in range(M2):
            cell = lattice[i][j]
            if meanfield:
                nbr_states_sent, neighbours = cell.get_local_exosome_field(
                    lattice, mf_search_radius, M1, exosome_string=exosome_string,
                    exosome_remove_ratio=exosome_remove_ratio, neighbours=mf_neighbours)
                if simsetup['FIELD_SEND'] is not None:
                    nbr_states_sent += cell.get_local_paracrine_field(lattice, neighbours, simsetup)
            else:
                nbr_states_sent, neighbours = cell.get_local_exosome_field(
                    lattice, search_radius, M1, exosome_string=exosome_string,
                    exosome_remove_ratio=exosome_remove_ratio, neighbours=None)
                if simsetup['FIELD_SEND'] is not None:
                    nbr_states_sent += cell.get_local_paracrine_field(lattice, neighbours, simsetup)
            """
            nbr_states_sent_01 = (nbr_states_sent + len(neighbours)) / 2.0
            field_neighbours = np.dot(simsetup['FIELD_SEND'], nbr_states_sent_01)

            print 'Hpair:', i,j, 'adding', np.dot(field_neighbours, cell.get_current_state())
            print 'neighbours are', neighbours
            print cell.get_current_label(), 'receiving from', \ 
               [lattice[p[0]][p[1]].get_current_label() for p in neighbours]
            print 'cell state', cell.get_current_state()
            print 'nbr field', nbr_states_sent
            print 'nbr field 01', nbr_states_sent_01
            print 'field_neighbours', field_neighbours
            """
            H_pairwise += np.dot(nbr_states_sent, cell.get_current_state())
    H_pairwise_scaled = - H_pairwise * gamma / 2  # divide by two because double-counting neighbours
    if norm:
        H_self = H_self / num_cells
        H_app = H_app / num_cells
        H_pairwise_scaled = H_pairwise_scaled / num_cells
    H_multi = H_self + H_app + H_pairwise_scaled
    return H_multi, H_self, H_app, H_pairwise_scaled


def calc_graph_energy(multicell, step, norm=True):
    """
    Graph energy is the multicell hamiltonian
        H_multi = [Sum (H_internal)] - gamma * [Sum(interactions)] - fs * [app_field dot Sum(state)]
    - Only valid for symmetric J, W, A matrices
    - Field should be 1D arr of size num_cells * num_genes, already scaled by field strength kappa
    - Returns total energy and the three main terms
         H_quadratic_form, H_multi, H_self, H_app, H_pairwise_scaled
    - Expect H_quadratic_form to equal H_multi when no_exo_field is used
    """
    num_cells = multicell.num_cells
    H_self = 0
    H_pairwise = 0
    H_app = 0

    H_quadratic_form = -0.5 * np.dot(np.dot(multicell.matrix_J_multicell,
                                            multicell.graph_state_arr[:, step]),
                                     multicell.graph_state_arr[:, step]) \
                       - np.dot(multicell.field_applied[:, step], multicell.graph_state_arr[:, step])

    for a in range(num_cells):
        cell_state = multicell.get_cell_state(a, step)

        # compute self energies and applied field contribution separately
        H_self += hamiltonian(cell_state, multicell.matrix_J, field=None, fs=0.0)

        # compute applied field term on that cell
        field_on_cell = multicell.get_field_on_cell(a, step)
        H_app -= np.dot(cell_state.T, field_on_cell)

        # compute interactions
        #  note that a cells neighboursa are the ones which 'send' to it
        #  if A_ij = 1, then there is a connection from i to j
        #  to get all the senders to cell i, we need to look at col i
        graph_neighbours_col = multicell.matrix_A[:, a]
        graph_neighbours = [idx for idx, i in enumerate(graph_neighbours_col) if i == 1]
        field_signal_exo, _ = general_exosome_field(multicell, a, step, neighbours=graph_neighbours)
        field_signal_W = general_paracrine_field(
            multicell, a, step, flag_01=False, neighbours=graph_neighbours)

        field_signal_unscaled = field_signal_exo + field_signal_W
        H_pairwise -= np.dot(field_signal_unscaled, cell_state)

    # divide by two because double-counting neighbours
    H_pairwise_scaled = H_pairwise * multicell.gamma / 2

    if norm:
        H_quadratic_form = H_quadratic_form / num_cells
        H_self = H_self / num_cells
        H_app = H_app / num_cells
        H_pairwise_scaled = H_pairwise_scaled / num_cells

    H_multi = H_self + H_app + H_pairwise_scaled
    return H_quadratic_form, H_multi, H_self, H_app, H_pairwise_scaled


def get_state_of_lattice(lattice, simsetup, datatype='full'):
    M1 = len(lattice)
    M2 = len(lattice[0])
    if datatype == 'full':
        x = np.zeros((M1, M2, simsetup['N']), dtype=int)
        for i in range(M1):
            for j in range(M2):
                cell = lattice[i][j]
                x[i,j,:] = (1 + cell.get_current_state()) / 2.0  # note 01 rep
    return x


def calc_compression_ratio(x, eta_0=None, datatype='full', elemtype=np.bool, method='manual'):
    """
    TODO add an eta_min ref point as all zeros of np.int with shape x.shape
    x - the data object (assume lies between -1 and 1)
    eta_0 is meant to be a rough upper bound on eta(x)
        - compute via 'maximally disordered' input x (random data)
    Returns eta(x)/eta_0 ranging between 0 and 1 +- eps
    """
    assert method in ['manual', 'package']
    assert datatype in ['full', 'custom']
    assert elemtype in [np.bool, np.int, np.float]

    def foo(x_to_compress):
        fname = 'tmp.npz'
        np.savez_compressed(fname, a=x_to_compress)
        fsize = os.path.getsize(fname)
        os.remove(fname)
        return fsize

    if datatype == 'full':
        if eta_0 is None:
            x_random = np.random.randint(0, high=2, size=x.shape, dtype=np.int)
            eta_0 = foo(x_random)  # consider max over few realizations?
        if x.dtype != elemtype:
            print('NOTE: Recasting x as elemtype', elemtype, 'from', x.dtype)
            x = x.astype(dtype=elemtype)
        eta = foo(x)
    else:
        assert datatype == 'custom'
        if eta_0 is None:
            assert -1 <= np.min(x) <= np.max(x) <= 1
            #x_random = np.random.rand(*(x.shape))*2 - 1  # TODO flag ref as float or bool
            if elemtype==np.bool:
                if x.dtype!=np.bool:
                    print('NOTE: Recasting x as np.float from', x.dtype)
                    x = x.astype(dtype=np.bool)
                x_random = np.random.randint(0, high=2, size=x.shape, dtype=np.bool)
            elif elemtype==np.int:
                assert np.issubdtype(x.dtype, np.int)
                nint = len(set(x))
                x_random = np.random.randint(0, high=nint+1, size=x.shape, dtype=np.int)
            else:
                assert elemtype==np.float
                if x.dtype!=np.float:
                    print('NOTE: Recasting x as np.float from', x.dtype)
                    x = x.astype(dtype=np.float)
                x_random = np.random.rand(*(x.shape)) * 2 - 1
            eta_0 = foo(x_random)  # consider max over few realizations?
        eta = foo(x)
    return float(eta)/eta_0, eta, eta_0


def test_compression_ratio():
    nn = 10000
    print("test_compression_ratio for x: %dx1 array..." % nn)
    x1 = np.ones(nn, dtype=np.int)  #[1, 1, 1, 1]
    x2 = np.zeros(nn, dtype=np.int) #[-1, -1, -1, -1]
    x3 = np.random.randint(0, high=2, size=nn)
    eta_ratio_1, eta_1, eta_0_1 = \
        calc_compression_ratio(x1, eta_0=None, datatype='custom', method='manual', elemtype=np.bool)
    eta_ratio_2, eta_2, eta_0_2 = \
        calc_compression_ratio(x2, eta_0=None, datatype='custom', method='manual', elemtype=np.bool)
    eta_ratio_3, eta_3, eta_0_3 = \
        calc_compression_ratio(x3, eta_0=None, datatype='custom', method='manual', elemtype=np.bool)
    print('x1', 'gives', eta_ratio_1, eta_1, eta_0_1)
    print('x2', 'gives', eta_ratio_2, eta_2, eta_0_2)
    print('x3', 'gives', eta_ratio_3, eta_3, eta_0_3)

    xshape = (1000, 500)
    print("test_compression_ratio for x: %d x %d array..." % (xshape[0], xshape[1]))
    x1 = np.ones(xshape)
    x2 = -np.ones(xshape)
    x3 = np.zeros(xshape)
    x4 = np.zeros(xshape)
    x4[:,0] = 1
    x5 = np.random.rand(*xshape)*2 - 1
    print(x5.shape)
    x6 = np.random.randint(-1, high=2, size=xshape)
    x7 = np.random.randint(0, high=2, size=xshape) * 2 - 1
    x_dict ={1: {'data': x1, 'label': 'all +1', 'dtype': np.bool},
             2: {'data': x2, 'label': 'all -1', 'dtype': np.bool},
             3: {'data': x3, 'label': 'all 0',  'dtype': np.bool},
             4: {'data': x4, 'label': 'all 0 except all 1 first col', 'dtype': np.bool},
             5: {'data': x5, 'label': 'rand floats -1 to 1', 'dtype': np.float},
             6: {'data': x6, 'label': 'rand ints -1, 0, 1', 'dtype': np.float},
             7: {'data': x7, 'label': 'rand ints -1, 1', 'dtype': np.float}}
    for idx in range(1, len(list(x_dict.keys()))+1):
        elem = x_dict[idx]['data']
        elemtype = x_dict[idx]['dtype']
        eta_ratio, eta, eta_0 = calc_compression_ratio(
            elem, eta_0=None, datatype='custom', method='manual', elemtype=elemtype)
        print(x_dict[idx]['label'], 'gives', eta_ratio, eta, eta_0)
    return None


if __name__ == '__main__':
    test_compression_ratio()
