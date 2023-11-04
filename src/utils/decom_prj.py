import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import radon, iradon, rescale, resize
from scipy import interpolate

##
def Z_prj(src, delta):
    ny, nx = src.shape

    dst = np.zeros_like(src)

    idx_src = np.linspace(-(ny - 1)/2, (ny - 1)/2, ny)

    for i in range(len(delta)):
        idx_dst = idx_src - delta[i]

        f = interpolate.interp1d(idx_src, src[:, i], kind='linear', bounds_error=False, fill_value=0)
        dst_ = f(idx_dst)
        # dst_[np.isnan(dst_)] = 0

        dst[:, i] = dst_
    return dst

##
# def K_prj (src, N, R1=1, R2=1, T=1):
#     [ny, _] = src.shape
#
#     # margin = int((ny - 2 * np.ceil((((R1 + R2) + N / np.sqrt(2)) / T + 1/2))) // 2)
#     margin = int((ny - 2 * np.ceil((((R1 + R2) + N / np.sqrt(2)) / T + 1/2))) // 2)
#     # my_pre = int(my // 2)
#     # my_post = int(my - my_pre)
#
#     dst = src[margin:-margin, :]
#
#     return dst

def K_prj (src, nStage):
    [ny, _] = src.shape

    ny_dec = ny // 2 ** nStage

    margin = int((ny - ny_dec) // 2)

    dst = src[margin:-margin, :]

    return dst

##
# def decom_prj(prj, N, theta, nStage, T=1, R1=1, R2=1, L=0):
#     N_cur = int(N / (2 ** nStage))
#
#     delta = N / (2 ** (nStage + 1))
#     delta_y = np.arange((-N / 2 + delta), (N / 2 - delta) + 1, 2*delta)
#     delta_x = np.arange((-N / 2 + delta), (N / 2 - delta) + 1, 2*delta)
#
#     delta_x_set, delta_y_set = np.meshgrid(delta_x, delta_y)
#
#     delta_set = np.hstack((delta_y_set.reshape(-1)[:, np.newaxis],
#                            delta_x_set.reshape(-1)[:, np.newaxis]))
#     theta_set = np.hstack((np.cos(theta * np.pi / 180).reshape(-1)[:, np.newaxis],
#                            np.sin(theta * np.pi / 180).reshape(-1)[:, np.newaxis]))
#
#     # Angular filter
#     phi = np.ones((4 * nStage - 1, 1))
#     phi[0] = 0.5
#     phi[-1] = 0.5
#
#     bnd = np.arange(-int(len(phi)/2), +int(len(phi)/2) + 1)
#
#     prj_set = []
#     rec_set = []
#
#     for i in range(len(delta_set)):
#         delta_cur = delta_set[i, :]
#         v = np.dot(theta_set, delta_cur[:, np.newaxis], ) / T
#
#         prj_dec = Z_prj(prj, v)
#         # prj_dec = K_prj(prj_dec, N_cur, R1, R2, T)
#         prj_dec = K_prj(prj_dec, nStage)
#
#         # rec_dec = iradon(prj_dec, theta=theta, circle=False, filter_name=None)
#         # rec_dec = iradon(prj_dec, theta=theta, circle=False, output_size=N_cur, filter_name=None)
#
#         # # Angular decimation
#         # prj_dec_ = np.zeros_like(prj_dec)
#         # for j in range(len(phi)):
#         #     shift_ = shift[j]
#         #     phi_ = phi[j]
#         #
#         #     prj_dec_ = prj_dec_ + phi_ * np.roll(prj_dec, shift=shift_, axis=0)
#         #
#         # prj_dec = prj_dec_[:, 0::2**nStage] / np.sum(phi)
#         #
#         # # rec_dec = iradon(prj_dec, theta=theta[0::2**nStage], circle=False, output_size=N_dec, filter_name=None)
#         # rec_dec = iradon(prj_dec, theta=theta[0::2**nStage], circle=False, filter_name=None)
#
#         prj_set.append(prj_dec)
#         # rec_set.append(rec_dec)
#
#     # return np.stack(prj_set, axis=2), np.stack(rec_set, axis=2)
#     return np.stack(prj_set, axis=2)


# def decom_prj(prj, nStage, nsz, theta):
def decom_prj(prj, nStage, params):

    if nStage == 0:
        return prj
    else:
        prj = np.transpose(prj, (1, 0)).copy()

    nx = params['nImgX']
    ny = params['nImgY']
    nview = params['nView']

    ndctx = params['nDctX']

    theta = np.linspace(0, 2 * np.pi, nview, endpoint=False)


    dx = nx / (2 ** (nStage + 1))
    dy = ny / (2 ** (nStage + 1))

    dctx = ndctx / (2 ** nStage)

    delta_y = np.arange((-ny / 2 + dy), (ny / 2 - dy) + 1, 2*dy)
    delta_x = np.arange((-nx / 2 + dx), (nx / 2 - dx) + 1, 2*dx)

    delta_x_set, delta_y_set = np.meshgrid(delta_x, delta_y)

    delta_set = np.hstack((delta_y_set.reshape(-1)[:, np.newaxis],
                           delta_x_set.reshape(-1)[:, np.newaxis]))
    theta_set = np.hstack((np.cos(theta).reshape(-1)[:, np.newaxis],
                           np.sin(theta).reshape(-1)[:, np.newaxis]))

    # Angular filter
    # phi = np.ones((4 * nStage - 1, 1))
    # phi[0] = 0.5
    # phi[-1] = 0.5
    #
    # shift = np.arange(-int(len(phi) / 2), +int(len(phi) / 2) + 1)

    # prj_set = []
    prj_set = np.zeros((len(delta_set), int(nview), int(dctx)), dtype=np.float32)

    # for i in range(len(delta_set)):
    for iy in range(len(delta_y)):
        for ix in range(len(delta_x)):
            iprj = len(delta_y)*ix + iy
            iprj = len(delta_x)*len(delta_y) - iprj - 1

            iset = len(delta_x) * iy + ix
            # iset = len(delta_x) * len(delta_y) - iset - 1

            delta_cur = delta_set[iset, :]
            v = np.dot(theta_set, delta_cur[:, np.newaxis], )

            prj_dec = Z_prj(prj, v)
            _prj_dec = K_prj(prj_dec, nStage)

            # # Angular decimation
            # prj_dec_ = np.zeros_like(prj_dec)
            # for j in range(len(phi)):
            #     shift_ = shift[j]
            #     phi_ = phi[j]
            #
            #     prj_dec_ = prj_dec_ + phi_ * np.roll(prj_dec, shift=shift_, axis=0)
            #
            # prj_dec = prj_dec_[:, 0::2**nStage] / np.sum(phi)

            # prj_set.append(prj_dec)
            prj_set[iprj] = np.transpose(_prj_dec, (1, 0))

    # return np.stack(prj_set, axis=2)
    return prj_set


def decom_prj_gpu(src, nStage, params):
    ndctx = int(params['nDctX']/2**nStage)
    ndctz = int(2 ** (2*nStage))
    nview = params['nView']

    src = src.transpose((0, 2, 1)).astype(np.float32).copy()
    dst = np.zeros((ndctz, nview, ndctx), dtype=np.float32)
    params['decomposition'](dst, src)
    # dst = np.transpose(dst, (0, 2, 1))
    return dst.copy()