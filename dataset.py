import os
import os.path

import torch
from torchvision import transforms
import torch.utils.data as data
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg',
    '.JPG',
    '.jpeg',
    '.JPEG',
    '.png',
    '.PNG',
    '.ppm',
    '.PPM',
    '.bmp',
    '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return Image.open(path).convert('RGB')


def portion_models(dataset_portion, category_path):
    '''
    Load category, model names from a shapenet dataset.
    '''

    def model_names(model_path):
        """ Return model names"""
        model_names = [name for name in os.listdir(model_path)
                       if os.path.isdir(os.path.join(model_path, name))]
        return sorted(model_names)

    model_path = os.path.join(category_path)
    models = model_names(model_path)
    num_models = len(models)

    portioned_models = models[int(num_models * dataset_portion[0]):int(num_models * dataset_portion[1])]

    return portioned_models

class Dataset(data.Dataset):

    def __init__(self, root, transform=None, loader=default_loader, model_portion=[0, 0.8]):
        image_dict = {}        
        image_list = []     
            
        for directory in os.listdir(root): # loop over model-categories
            image_dict[directory] = {}
            for subdirectory in portion_models(model_portion, os.path.join(root,directory)): # loop over models
                image_dict[directory][subdirectory] = []                
                for filename in os.listdir(os.path.join(root,directory,subdirectory,'rendering')): # loop over image files
                    if is_image_file(filename):
                        image_list.append('{}'.format(os.path.join(directory,subdirectory,'rendering',filename)))
                        image_dict[directory][subdirectory].append('{}'.format(filename))
## Simple image folder case:        
#        for filename in os.listdir(root):
#            if is_image_file(filename):
#                images.append('{}'.format(filename))

        self.root = root
        self.image_list = image_list
        self.image_dict = image_dict
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        filename = self.imgs[index]
        try:
            img = self.loader(os.path.join(self.root, filename))
        except:
            return torch.zeros((3, 32, 32))

        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgs)
