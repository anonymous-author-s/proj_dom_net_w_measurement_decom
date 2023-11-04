import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import radon, iradon, rescale, resize
from scipy import interpolate

from multiprocessing import Pool, Process

# import multiprocess as mp

##
def Z_prj(src, delta):
    ny, nx = src.shape

    dst = np.zeros_like(src)

    idx_src = np.linspace(-(ny - 1)/2, (ny - 1)/2, ny)

    def _Z_prj(i):
        idx_dst = idx_src - delta[i]

        f = interpolate.interp1d(idx_src, src[:, i], kind='linear', bounds_error=False, fill_value=0)
        dst_ = f(idx_dst)
        # dst_[np.isnan(dst_)] = 0

        dst[:, i] = dst_

    # pool = Pool(16)
    # pool.map(interp1d, range(len(delta)))

    for i in range(len(delta)):
        _Z_prj(i)
        # idx_dst = idx_src - delta[i]
        #
        # f = interpolate.interp1d(idx_src, src[:, i], kind='linear', bounds_error=False, fill_value=0)
        # dst_ = f(idx_dst)
        # # dst_[np.isnan(dst_)] = 0
        #
        # dst[:, i] = dst_
    return dst

##
def K_prj (src, nstage):
    [ny, _] = src.shape

    ny_dec = ny // 2 ** nstage

    margin = int((ny - ny_dec) // 2)

    dst = src[margin:-margin, :]

    return dst

##
def decom_prj(prj, nstage, params):

    if nstage == 0:
        return np.transpose(prj, (1, 0))

    nx = params['nImgX']
    ny = params['nImgY']
    nview = params['nView']

    ndctx = params['nDctX']

    theta = np.linspace(0, 2 * np.pi, nview, endpoint=False)


    dx = nx / (2 ** (nstage + 1))
    dy = ny / (2 ** (nstage + 1))

    dctx = ndctx / (2 ** nstage)

    delta_y = np.arange((-ny / 2 + dy), (ny / 2 - dy) + 1, 2*dy)
    delta_x = np.arange((-nx / 2 + dx), (nx / 2 - dx) + 1, 2*dx)

    delta_x_set, delta_y_set = np.meshgrid(delta_x, delta_y)

    delta_set = np.hstack((delta_y_set.reshape(-1)[:, np.newaxis],
                           delta_x_set.reshape(-1)[:, np.newaxis]))
    theta_set = np.hstack((np.cos(theta).reshape(-1)[:, np.newaxis],
                           np.sin(theta).reshape(-1)[:, np.newaxis]))

    # Angular filter
    # phi = np.ones((4 * nstage - 1, 1))
    # phi[0] = 0.5
    # phi[-1] = 0.5
    #
    # shift = np.arange(-int(len(phi) / 2), +int(len(phi) / 2) + 1)

    # prj_set = []
    prj_set = np.zeros((len(delta_set), int(nview), int(dctx)), dtype=np.float32)

    def _decom_prj(i):
        delta_cur = delta_set[i, :]
        v = np.dot(theta_set, delta_cur[:, np.newaxis], )

        prj_dec = Z_prj(prj, v)
        prj_dec = K_prj(prj_dec, nstage)

        prj_set[i] = np.transpose(prj_dec, (1, 0))

    procs = []

    for i in range(len(delta_set)):
        proc = Process(target=_decom_prj, args=(i,))
        procs.append(proc)
        proc.start()

        # _decom_prj(i)

        # delta_cur = delta_set[i, :]
        # v = np.dot(theta_set, delta_cur[:, np.newaxis], )
        #
        # prj_dec = Z_prj(prj, v)
        # prj_dec = K_prj(prj_dec, nstage)
        #
        # prj_set[i] = np.transpose(prj_dec, (1, 0))

    for proc in procs:
        proc.join()

    # return np.stack(prj_set, axis=2)
    return prj_set
