import numpy as np
import matplotlib.pyplot as plt

def decom_img(img, nStage, params):
    if nStage == 0:
        return img

    nx = params['nImgX']
    ny = params['nImgY']

    ms = 2 ** nStage

    mx = nx // ms
    my = ny // ms


    # img_dec = np.zeros((nset ** 2, my, mx), dtype=np.float32)
    # 
    # for i in range(nset):
    #     for j in range(nset):
    #         idx = nset * j + i
    #         img_dec[idx] = img[0, my * (nset - 1 - i): my * (nset - 1 - i + 1), mx * (nset - 1 - j): mx * (nset - 1 - j + 1)]
    # return img_dec

    # img_dec = img.reshape((-1, ms, my, ms, mx))
    # img_dec = img_dec.transpose((0, 1, 3, 2, 4))
    # img_dec = img_dec.reshape((-1, ms * ms, my, mx))

    img_dec = img.reshape((ms, my, ms, mx))
    img_dec = img_dec.transpose((0, 2, 1, 3))
    img_dec = img_dec.reshape((ms * ms, my, mx))

    return img_dec
