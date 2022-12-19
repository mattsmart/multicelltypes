import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path

from singlecell.singlecell_functions import single_memory_projection


axis_buffer = 20.0
axis_length = 100.0
axis_tick_length = int(axis_length + axis_buffer)
memory_keys = [5,24]
memory_colour_dict = {5: 'blue', 24: 'red'}
fast_flag = False  # True - fast / simple plotting
#nutrient_text_flag = False  # True - plot nutrient quantity at each grid location (slow)  TODO: plot scalar at each location?


def polygon_manual(sides, radius=1, rotation=0, translation=None):
    # simple, see https://stackoverflow.com/questions/23411688/drawing-polygon-with-n-number-of-sides-in-python-3-2
    one_segment = np.pi * 2 / sides
    points = [
        (np.sin(one_segment * i + rotation) * radius,
         np.cos(one_segment * i + rotation) * radius)
        for i in range(sides)]
    if translation:
        points = [[sum(pair) for pair in zip(point, translation)]
                  for point in points]
    return points


def polygon_celltypes_example(labels, colours):
    # TODO delete
    # see https://stackoverflow.com/questions/3654289/scipy-create-2d-polygon-mask
    # https://stackoverflow.com/questions/23411688/drawing-polygon-with-n-number-of-sides-in-python-3-2
    assert len(labels) == len(colours)
    p = len(labels)

    width, height = 1000, 1000
    """
    polygon = [(0.1 * width, 0.1 * height), (0.15 * width, 0.7 * height), (0.8 * width, 0.75 * height),
               (0.72 * width, 0.15 * height)]
    """
    polygon_corners = polygon_manual(p, radius=400, rotation=-np.pi/2, translation=[width/2, height/2])
    from matplotlib.path import Path

    poly_path = Path(polygon_corners)

    x, y = np.mgrid[:height, :width]
    coors = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))  # coors.shape is (4000000,2)
    mask = poly_path.contains_points(coors)
    print(coors.shape, mask.shape)
    plt.imshow(mask.reshape(height, width))
    plt.show()


def polygon_celltypes_legend(labels, colours, ax=None):
    # see https://stackoverflow.com/questions/3654289/scipy-create-2d-polygon-mask
    # https://stackoverflow.com/questions/23411688/drawing-polygon-with-n-number-of-sides-in-python-3-2
    # colours = rgb list of lists
    # TODO don't think this is a good rep for p>3 since can't see blends of more than 2 cell types
    assert len(labels) == len(colours)
    colours = np.array(colours)  # this has each row as an anchor colour
    colours_t = np.transpose(colours)  # this has each column as an anchor colour
    p = len(labels)
    assert p == 3  # TODO should switch to generalized / higher order venn diagram for p > 4 cell types

    figsize = (4.0, 4.0)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()

    # create polygon and trace path between corners
    npts = 200
    width, height = npts, npts
    radius = npts / 2.0
    angle = -np.pi/2
    center = [width / 2.0, height / 2.0]
    center_arr = np.array(center)
    polygon_corners = polygon_manual(p, radius=radius, rotation=angle, translation=center)
    poly_path = Path(polygon_corners)

    def map_xy_rgba_OLD(xval, yval):
        # rule 1: alpha = distance from center point (corners have NO alpha i.e. 1.0 opaque)
        # rule 2: get distance to each corner; take weighted average (sqrt of sum of sqrs) of the p anchor colours
        loc = np.array([xval, yval])
        #alpha = np.max([0.5, np.linalg.norm(loc-center_arr)/radius])  # i.e. get_dist_from_center(loc) -> scalar
        alpha = 1.0
        weights = np.zeros(p)                          # i.e. get_corner_weights(loc) -> p array, weight = 1 on respective corner, 0.0 at or beyond center
        for idx, corner in enumerate(polygon_corners):
            weights[idx] = np.max([0, 1 - np.linalg.norm(loc - corner) / radius])
        # get weighted rgb as sum of squares of weighted colours
        weight_sqrt_sqr = np.sqrt(np.sum(np.square(weights)))
        colours_t_weighted = np.multiply(colours_t, weights)
        rgb = np.sum(np.square(colours_t_weighted), axis=1) / weight_sqrt_sqr
        rgba = [rgb[0], rgb[1], rgb[2], alpha]
        return rgba

    def map_xy_rgba(xval, yval):
        # rule 1: alpha = distance from center point (corners have NO alpha i.e. 1.0 opaque)
        # rule 2: get distance to each corner; take weighted average (sqrt of sum of sqrs) of the p anchor colours
        loc = np.array([xval, yval])

        #alpha = np.max([0.5, np.linalg.norm(loc-center_arr)/radius])  # i.e. get_dist_from_center(loc) -> scalar
        alpha = np.max([0.0, np.linalg.norm(loc - center_arr) / radius])  # i.e. get_dist_from_center(loc) -> scalar
        #alpha = 1.0

        weights = np.zeros(p)                          # i.e. get_corner_weights(loc) -> p array, weight = 1 on respective corner, 0.0 at or beyond center
        for idx, corner in enumerate(polygon_corners):
            weights[idx] = np.max([0, 2 - np.linalg.norm(loc - corner) / radius])
        # get weighted rgb as sum of squares of weighted colours
        weight_sqrt_sqr = np.sqrt(np.sum(np.square(weights)))
        colours_t_weighted = np.multiply(colours_t, weights)
        rgb = np.sum(np.square(colours_t_weighted), axis=1) / weight_sqrt_sqr
        rgba = [rgb[0], rgb[1], rgb[2], alpha]
        return rgba

    # get x y grid, coords, and mask for inside/outside polygon
    x, y = np.mgrid[:height, :width]
    coors = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))          # coors.shape is (width*height, 2)
    #print x.reshape(-1, 1).shape
    #print y.reshape(-1, 1).shape
    #coors_reshape = coors.reshape(height, width)
    mask = poly_path.contains_points(coors).reshape(height, width)   # get true false on bounding rectangle
    print(x.shape, y.shape)
    print(mask.shape)
    #print mask[0,0], type(mask[0,0])
    #print mask_reshape[npts/2, npts/2], type(mask_reshape[npts/2, npts/2])
    #rint coors.shape, mask.shape

    # build imshow colour array
    imshow_array = np.zeros((len(x), len(y), 4))  # last slow is RGB + transparency
    for i in range(npts):
        print(i)
        for j in range(npts):
            xval = x[i, j]
            yval = y[i, j]
            if mask[i, j]:
                imshow_array[i, j] = map_xy_rgba(xval, yval)
            else:
                imshow_array[i, j] = None
    ax.imshow(imshow_array)

    # TODO plot celltype labels
    for idx, corner in enumerate(polygon_corners):
        angle_polar = angle + 2 * np.pi / p * idx
        ax.text(np.cos(angle_polar) * radius*1.15 + center[0] - npts*0.05,
                np.sin(angle_polar) * radius*1.15 + center[1] + npts*0.05,
                labels[idx], fontsize=28)

    # remove all axis
    plt.axis('off')
    # TODO save to file option
    plt.show()
    return ax

'''
def polygon_celltypes_legend(labels, colors, smallfig=False, ax=None):
    """
    Plots a regular polygon with each corner representing a cell type and its anchor color
    Num of vertices = num cell types = len(labels)
    Interior of the polygon is interpolated using ???????????????????????
    Center is white by default
    TODO - 2p polygon with anti vertex = anti cell type?
    TODO - short term, simpler, triangle versionb for poster?
    """
    assert len(labels) == len(colors)
    p = len(labels)

    if smallfig:
        figsize = (2.0, 1.6)
        text_fs = 20
        ms = 10
        stlw = 0.5
        nn = 20
        ylim_mod = 0.08
    else:
        figsize=(4, 3)
        text_fs = 20
        ms = 10
        stlw = 0.5
        nn = 100
        ylim_mod = 0.04

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()

    X = np.array([[0.0, 0.0], [N, 0.0], [N / 2.0, N]])

    if smallfig:
        t1 = plt.Polygon(X[:3, :], color=(0.902, 0.902, 0.902), alpha=1.0, ec=(0.14, 0.14, 0.14), lw=1, zorder=1)
        ax.add_patch(t1)
        ax.text(-params.N*0.12, -params.N*0.08, r'$x$', fontsize=text_fs)
        ax.text(params.N*1.045, -params.N*0.08, r'$y$', fontsize=text_fs)
        ax.text(params.N/2.0*0.93, params.N*1.115, r'$z$', fontsize=text_fs)
    else:
        t1 = plt.Polygon(X[:3, :], color='k', alpha=0.1, zorder=1)
        ax.add_patch(t1)
        ax.text(-params.N * 0.07, -params.N * 0.05, r'$x$', fontsize=text_fs)
        ax.text(params.N * 1.03, -params.N * 0.05, r'$y$', fontsize=text_fs)
        ax.text(params.N / 2.0 * 0.96, params.N * 1.07, r'$z$', fontsize=text_fs)

    if streamlines:
        B, A = np.mgrid[0:N:nn*1j, 0:N:nn*1j]
        # need to mask outside of simplex
        ADOT = np.zeros(np.shape(A))
        BDOT = np.zeros(np.shape(A))
        SPEEDS = np.zeros(np.shape(A))
        for i in xrange(nn):
            for j in xrange(nn):
                a = A[i, j]
                b = B[i, j]
                z = b
                x = N - a - b/2.0  # TODO check
                y = N - x - z
                if b > 2.0*a or b > 2.0*(N-a) or b == 0:  # check if outside simplex
                    ADOT[i, j] = np.nan
                    BDOT[i, j] = np.nan
                else:
                    dxvecdt = params.ode_system_vector([x,y,z], None)
                    SPEEDS[i, j] = np.sqrt(dxvecdt[0]**2 + dxvecdt[1]**2 + dxvecdt[2]**2)
                    ADOT[i, j] = (-dxvecdt[0] + dxvecdt[1])/2.0  # (- xdot + ydot) / 2
                    BDOT[i, j] = dxvecdt[2]                      # zdot
        if smallfig:
            strm = ax.streamplot(A, B, ADOT, BDOT, color=(0.34, 0.34, 0.34), linewidth=stlw)
        else:
            # this will color lines
            """
            strm = ax.streamplot(A, B, ADOT, BDOT, color=SPEEDS, linewidth=stlw, cmap=plt.cm.coolwarm)
            if cbar:
                plt.colorbar(strm.lines)
            """
            # this will change line thickness
            stlw_low = stlw
            stlw_high = 1.0
            speeds_low = np.min(SPEEDS)
            speeds_high = np.max(SPEEDS)
            speeds_conv = 0.3 + SPEEDS / speeds_high
            strm = ax.streamplot(A, B, ADOT, BDOT, color=(0.34, 0.34, 0.34), linewidth=speeds_conv)

    if fp:
        stable_fps = []
        unstable_fps = []
        all_fps = fp_location_fsolve(params, check_near_traj_endpt=True, gridsteps=35, tol=10e-1, buffer=True)
        for fp in all_fps:
            J = jacobian_numerical_2d(params, fp[0:2])
            eigenvalues, V = np.linalg.eig(J)
            if eigenvalues[0] < 0 and eigenvalues[1] < 0:
                stable_fps.append(fp)
            else:
                unstable_fps.append(fp)
        for fp in stable_fps:
            fp_x = (N + fp[1] - fp[0]) / 2.0
            #plt.plot(fp_x, fp[2], marker='o', markersize=ms, markeredgecolor='black', linewidth='3', color='k')
            ax.plot(fp_x, fp[2], marker='o', markersize=ms, markeredgecolor='black', linewidth='3', color=(0.212, 0.271, 0.31), zorder=10)  # #b88c8c is pastel reddish, (0.212, 0.271, 0.31) blueish
        for fp in unstable_fps:
            fp_x = (N + fp[1] - fp[0]) / 2.0
            #plt.plot(fp_x, fp[2], marker='o', markersize=ms, markeredgecolor='black', linewidth='3', markerfacecolor="None")
            ax.plot(fp_x, fp[2], marker='o', markersize=ms, markeredgecolor='black', linewidth='3', color=(0.902, 0.902, 0.902), zorder=10)

    ax.set_ylim(-N*ylim_mod, N*(1+ylim_mod))
    ax.axis('off')
    return ax
'''

if __name__ == '__main__':
    turquoise = [30, 223, 214]
    soft_yellow = [237, 209, 112]
    soft_brick = [192, 86, 64]

    soft_blue = [148, 210, 226]
    soft_blue_alt1 = [58, 128, 191]

    soft_red = [192, 86, 64]
    soft_red_alt1 = [240, 166, 144]
    soft_red_alt2 = [255, 134, 113]

    soft_yellow = [237, 209, 112]

    colours_A = np.divide([soft_blue_alt1, soft_red, soft_yellow], 255.0)
    #polygon_celltypes_example(['A','B','C'], ['red', 'blue', 'green'])
    polygon_celltypes_legend(['A','B','C'], colours_A)
