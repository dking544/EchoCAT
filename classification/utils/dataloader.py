from random import shuffle
from PIL import Image
import copy
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms
import torch

from core.datasets.compose import Compose

class Mydataset(Dataset):
    def __init__(self, gt_labels, cfg):
        self.gt_labels = gt_labels
        self.cfg = cfg
        self.pipeline = Compose(self.cfg)
        self.data_infos = self.load_annotations()

    def __len__(self):
        return len(self.gt_labels)

    def __getitem__(self, index):

        results = copy.deepcopy(self.data_infos[index])


        results = self.pipeline(results)


        if 'gt_label' not in results:
             results['gt_label'] = self.data_infos[index]['gt_label']

        if 'filename' not in results:
             results['filename'] = self.data_infos[index]['filename']

        return results['img'], int(results['gt_label']), results['filename']

    def load_annotations(self):

        if len(self.gt_labels) == 0:
            raise TypeError('ann_file is None')


        samples = [x.strip().rsplit(' ', 1) for x in self.gt_labels]

        data_infos = []
        for filename, gt_label in samples:
            info = {


                'filename': filename,
                'ori_filename': filename,
                'img_prefix': None,


                'img_info': {'filename': filename},


                'gt_label': int(gt_label)
            }
            data_infos.append(info)
        return data_infos

def collate(batches):
    images, gts, image_path = tuple(zip(*batches))
    images = torch.stack(images, dim=0)
    gts = torch.as_tensor(gts)

    return images, gts, image_path
