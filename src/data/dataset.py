import os
import numpy as np
import torch
from glob import glob

from skimage.util import pad
from skimage import transform

from src.utils.params import get_params, get_decom

from src.utils.decom_img import decom_img
from src.utils.decom_prj import decom_prj_gpu as decom_prj
from src.utils.comp_img import comp_img

from src.models import op_ct

from scipy.stats import poisson

import matplotlib.pyplot as plt


class Dataset(torch.utils.data.Dataset):
    """
    datasets of image files of the form
       stuff<number>_trans.pt
       stuff<number>_density.pt
    """

    def __init__(self, data_dir,
                 transform=None, idx=None,
                 params=None, nStage=None, downsample=None,
                 noise_range=None, downsample_range=None,
                 input_type='input', label_type='label',
                 flip=0.5, weight=5e3, mode='train'):
        if noise_range is None:
            noise_range = [4, 8]
        if downsample_range is None:
            downsample_range = [3, 4, 6, 8, 12]
        self.data_dir = data_dir
        self.transform = transform
        self.params = params
        self.nStage = nStage
        self.noise_range = noise_range
        self.downsample_range = downsample_range
        self.downample = downsample
        self.input_type = input_type
        self.label_type = label_type
        self.flip = flip
        self.weight = weight
        self.mode = mode

        ##
        self.params = get_params(name_project=params['name_project'], dir_project=params['dir_project'], dir_dll=params['dir_dll'], nStage=0, downsample=1, nView=params['nView'])
        self.params_ds = get_params(name_project=params['name_project'], dir_project=params['dir_project'], dir_dll=params['dir_dll'], nStage=0, downsample=1, nView=params['nView'])
        self.params_dec = get_params(name_project=params['name_project'], dir_project=params['dir_project'], dir_dll=params['dir_dll'], nStage=nStage, downsample=1, nView=params['nView'])

        self.params = get_decom(path_dll='lib_decom/', nStage=nStage, params=self.params)

        ##
        lst_input = glob(os.path.join(data_dir, f'{self.input_type}_*.npy'))
        lst_input.sort()

        lst_label = glob(os.path.join(data_dir, f'{self.label_type}_*.npy'))
        lst_label.sort()

        if idx is None:
            self.lst_input = lst_input
            self.lst_label = lst_label
        else:
            self.lst_input = lst_input[idx]
            self.lst_label = lst_label[idx]

        ##
    def __getitem__(self, index):
        prj = np.load(self.lst_input[index])
        # fbp = np.load(self.lst_label[index])

        # prj = self.weight * prj
        
        if np.random.rand() < self.flip:
            prj = np.flip(prj, axis=1)

        if np.random.rand() < self.flip:
            prj = np.flip(prj, axis=2)

        if np.random.rand() < self.flip:
            prj = np.roll(prj, shift=int(np.random.randint(self.params['nView'])), axis=1)

        label_prj = prj.copy()
        # label_fbp = fbp.copy()

        # if self.noise_range[-1]:
        #     i0 = 10 ** np.random.uniform(self.noise_range[0], self.noise_range[-1])
        #     input_prj, input_noise = add_poisson(label_prj.copy(), i0)

        downsample = self.downsample_range[np.random.randint(0, len(self.downsample_range))]

        self.params_ds = get_params(name_project=self.params['name_project'], dir_project=self.params['dir_project'],
                                    dir_dll=self.params['dir_dll'], nStage=0, downsample=downsample, nView=self.params['nView'] // downsample)


        label_flt = op_ct.Filtration(label_prj.copy(), self.params)
        # input_flt = op_ct.Filtration(input_prj.copy(), self.params)
        input_flt = label_flt[:, ::downsample, :].copy()


        label_fbp = op_ct.Backprojection(label_flt.copy(), self.params)
        input_fbp = op_ct.Backprojection(input_flt.copy(), self.params_ds)
        
        input_prj = op_ct.Projection(input_fbp.copy(), self.params)
        input_flt = op_ct.Filtration(input_prj.copy(), self.params)
        input_mask = np.zeros_like(input_flt)
        input_mask[:, ::downsample, :] = 1

        # input_flt[:, ::downsample, :] = label_flt[:, ::downsample, :].copy()
        input_flt = (input_mask * label_flt + (1 - input_mask) * input_flt).copy()



        # label_flt_dec = decom_prj(label_flt, self.nStage, self.params)
        # label_fbp_dec = op_ct.Backprojection(label_flt_dec.copy(), self.params_dec)

        label_flt_dec = decom_prj(label_flt, self.nStage, self.params)
        input_flt_dec = decom_prj(input_flt, self.nStage, self.params)
        input_mask_dec = np.zeros_like(input_flt_dec)
        input_mask_dec[:, ::downsample, :] = 1

        # input_noise_dec = (input_flt_dec - label_flt_dec).copy()

        label_fbp_dec = decom_img(label_fbp, self.nStage, self.params)
        # label_fbp_dec = op_ct.Backprojection(label_flt_dec, self.params_dec)
        input_fbp_dec = op_ct.Backprojection(input_flt_dec, self.params_dec)

        

        ## Apply FOV - Projection
        data = {'label_fbp': label_fbp_dec.copy(), 'label_flt': label_flt_dec.copy(),
                'input_fbp': input_fbp_dec.copy(), 'input_flt': input_flt_dec.copy(),
                'input_mask': input_mask_dec.copy()}

        if self.transform:
            data = self.transform(data)

        return data

    def __len__(self):
        return len(self.lst_label)



class Normalize(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        for key, value in data.items():
            data[key] = (value - self.mean) / self.std
        return data


class UnNormalize(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        for key, value in data.items():
            data[key] = (value * self.std) + self.mean
        return data


class Weighting(object):
    def __init__(self, key):
        self.key = key

    def __call__(self, data):
        wgt = data[self.key]

        for key, value in data.items():
            if not key == self.key:
                data[key] = wgt * value
        return data


class Converter(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, dir='numpy2tensor'):
        self.dir = dir

    def __call__(self, sample):
        if self.dir == 'numpy2tensor':
            for key, value in sample.items():
                sample[key] = torch.from_numpy(value.copy())  # .permute((2, 0, 1))
        elif self.dir == 'tensor2numpy':
            # for key, value in sample.items():
            #     sample[key] = value.numpy()  # .transpose(1, 2, 0)
            sample = sample.cpu().detach().numpy()  # .transpose(1, 2, 0)

        return sample


class RandomFlip(object):
    def __call__(self, data):
        # Random Left or Right Flip
        if np.random.rand() > 0.5:
            for key, value in data.items():
                data[key] = np.flip(value, axis=0)

        if np.random.rand() > 0.5:
            for key, value in data.items():
                data[key] = np.flip(value, axis=1)

        return data


class ZeroPad(object):
    """Rescale the image in a sample to a given size

    Args:
      output_size (tuple or int): Desired output size.
                                  If tuple, output is matched to output_size.
                                  If int, smaller of image edges is matched
                                  to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, data):
        h, w = data['label'].shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        l = (new_w - w) // 2
        r = (new_w - w) - l

        u = (new_h - h) // 2
        b = (new_h - h) - u

        for key, value in data.items():
            data[key] = pad(value, pad_width=((u, b), (l, r), (0, 0)))

        return data


class Rescale(object):
    """Rescale the image in a sample to a given size

    Args:
      output_size (tuple or int): Desired output size.
                                  If tuple, output is matched to output_size.
                                  If int, smaller of image edges is matched
                                  to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, data):
        h, w = data['label'].shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        for key, value in data.items():
            data[key] = transform.resize(value, (new_h, new_w), mode=0)

        return data


class RandomCrop(object):
    """Crop randomly the image in a sample

    Args:
      output_size (tuple or int): Desired output size.
                                  If int, square crop is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, data):
        h, w = data['label'].shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        id_y = np.arange(top, top + new_h, 1)[:, np.newaxis]
        id_x = np.arange(left, left + new_w, 1)

        for key, value in data.items():
            data[key] = value[id_y, id_x]

        return data


class CenterCrop(object):
    """Crop randomly the image in a sample

    Args:
    output_size (tuple or int): Desired output size.
                                If int, square crop is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, data):
        h, w = data['label'].shape[:2]
        new_h, new_w = self.output_size

        top = (h - new_h) // 2
        left = (w - new_w) // 2

        for key, value in data.items():
            data[key] = value[top: top + new_h, left: left + new_w]

        return data

class UnifromSample(object):
    """Crop randomly the image in a sample

    Args:
      output_size (tuple or int): Desired output size.
                                  If int, square crop is made.
    """

    def __init__(self, stride):
        assert isinstance(stride, (int, tuple))
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            assert len(stride) == 2
            self.stride = stride

    def __call__(self, data):
        h, w = data['label'].shape[:2]
        stride_h, stride_w = self.stride
        new_h = h // stride_h
        new_w = w // stride_w

        top = np.random.randint(0, stride_h + (h - new_h * stride_h))
        left = np.random.randint(0, stride_w + (w - new_w * stride_w))

        id_h = np.arange(top, h, stride_h)[:, np.newaxis]
        id_w = np.arange(left, w, stride_w)

        for key, value in data.items():
            data[key] = value[id_h, id_w]

        return data


def generate_fov(params, ratio, nker=7, sgm=3):
    # def gaussian_kernel(size, sigma=1, verbose=False):
    #     def dnorm(x, mu, sd):
    #         return 1 / (np.sqrt(2 * np.pi) * sd) * np.exp(-np.power((x - mu) / sd, 2) / 2)
    #
    #     kernel_1D = np.linspace(-(size // 2), size // 2, size)
    #     for i in range(size):
    #         kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    #     kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
    #
    #     # kernel_2D *= 1.0 / kernel_2D.max()
    #     kernel_2D *= 1.0 / np.sum(kernel_2D)
    #
    #     if verbose:
    #         plt.imshow(kernel_2D, interpolation='none', cmap='gray')
    #         plt.title("Image")
    #         plt.show()
    #
    #     return kernel_2D

    if params['CT_NAME'] == 'parallel':
        dDct = params['dDct']
    elif params['CT_NAME'] == 'fan':
        dDct = params['dDct'] * params['dDSO'] / params['dDSD']

    # FOV - Projection
    radius_dct = dDct * params['nDct'] / 2.0
    radius_fov = ratio * radius_dct

    fov_prj = np.linspace(-radius_dct, radius_dct, params['nDct'])
    fov_prj = 1.0 * (np.abs(fov_prj) <= radius_fov)
    fov_prj = np.tile(fov_prj[np.newaxis, :], (params['nView'], 1)).astype(np.float32)

    # FOV - Image
    radius_img_x = params['dImgX'] * (params['nImgX'] - 1) / 2.0
    radius_img_y = params['dImgY'] * (params['nImgY'] - 1) / 2.0

    fov_img_x = np.linspace(-radius_img_x, radius_img_x, params['nImgX'])
    fov_img_y = np.linspace(-radius_img_y, radius_img_y, params['nImgY'])

    ms_img_x, ms_img_y = np.meshgrid(fov_img_x, fov_img_y)

    # FOV with tight radius
    fov_img = (1.0 * (((ms_img_x ** 2) + (ms_img_y ** 2)) < (radius_fov ** 2))).astype(np.float32)

    return fov_prj, fov_img

# def add_poisson(x0, i0):
#     x = np.exp(-x0)
#     x = poisson.rvs(i0 * x)
#     x[x < 1] = 1
#     x = -np.log(x / i0)
#     x[x < 0] = 0
#     x = x.astype(np.float32)
#
#     noise = x - x0

# def add_poisson(x0, i0, a):
#
#     # Development and Validation of a Practical Lower-Dose-Simulation Tool for Optimizing Computed Tomography Scan Protocols
#     sz = x0.shape
#
#     noise = np.sqrt((1.0 - a)/a * np.exp(x0)/i0) * np.random.randn(sz[0], sz[1])
#     x = x0 + noise
#
#     noise = noise.astype(np.float32)
#     x = x.astype(np.float32)
#
#     return x, noise

# Simple Lower-Dose-Simulation
def add_poisson(x0, i0):
    x = np.exp(-x0)
    x = poisson.rvs(i0 * x).reshape(x0.shape)
    x[x < 1] = 1
    x = -np.log(x / i0)
    x[x < 0] = 0
    x = x.astype(np.float32)

    noise = x - x0

    return x, noise