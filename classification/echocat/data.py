from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from echocat.runtime import recipe
from echocat.transforms import ForegroundPercentileWindowThreeChannel
from utils.dataloader import Mydataset


class ExternalDataset(Dataset):
    def __init__(self,lines,task):
        self.samples=[line.strip().rsplit(' ',1) for line in lines]
        self.task=task
        self.geometry=transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(256,interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224)])
        self.tensorize=transforms.Compose([transforms.ToTensor(),
            transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
        self.window=ForegroundPercentileWindowThreeChannel()

    def __len__(self):return len(self.samples)

    def __getitem__(self,index):
        path,label=self.samples[index]
        with Image.open(path) as image:
            image=self.geometry(image.convert('RGB'))
            if self.task=='view':
                image=Image.fromarray(self.window({'img':np.asarray(image)})['img'])
            tensor=self.tensorize(image)
        return tensor,int(label),path


def evaluation_dataset(lines,task,protocol):
    if protocol=='external':return ExternalDataset(lines,task)
    if protocol=='validation':return Mydataset(lines,recipe(task)['val_pipeline'])
    raise ValueError('Unknown input protocol')
