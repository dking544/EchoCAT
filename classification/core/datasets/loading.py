import os.path as osp

import cv2
import numpy as np

from core.datasets.io import imfrombytes
from .build import PIPELINES


@PIPELINES.register_module()
class LoadImageFromFile(object):


    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type

    def get(self,filepath):


        with open(filepath, 'rb') as f:
            value_buf = f.read()
        return value_buf

    def __call__(self, results):
        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img_bytes = self.get(filename)
        img = imfrombytes(img_bytes, flag=self.color_type)

        if not osp.exists(filename):
            raise FileNotFoundError(f"The file {filename} does not exist.")

        img_bytes = self.get(filename)

        img_array = np.frombuffer(img_bytes, np.uint8)

        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Image decoding failed")

        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results
