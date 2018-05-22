import os
import os.path

import torch
from torchvision import transforms
import torch.utils.data as data
from PIL import Image
from numpy import random

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

    def __init__(self, root, transform=None, loader=default_loader, model_portion=[0, 0.8], max_views=5):
        image_dict = {}        
        image_list = []   
        cat_model_list = []
        im_list = []
            
        for directory in os.listdir(root): # loop over model-categories
            image_dict[directory] = {}
            for subdirectory in portion_models(model_portion, os.path.join(root,directory)): # loop over models
                image_dict[directory][subdirectory] = []        
                cat_model_list.append([directory, subdirectory])
                im_list_cur = []
                for filename in os.listdir(os.path.join(root,directory,subdirectory,'rendering')): # loop over image files
                    if is_image_file(filename):
                        image_list.append('{}'.format(os.path.join(directory,subdirectory,'rendering',filename)))
                        image_dict[directory][subdirectory].append('{}'.format(filename))
                        im_list_cur.append(filename)
                im_list.append(im_list_cur)
## Simple image folder case:        
#        for filename in os.listdir(root):
#            if is_image_file(filename):
#                images.append('{}'.format(filename))        
        combined = list(zip(cat_model_list, im_list))
        random.shuffle(combined)        
        cat_model_list[:], im_list[:] = zip(*combined)
        
        self.max_views = max_views
        self.cur_idx = 0
        self.root = root
        self.image_list = image_list
        self.image_dict = image_dict
        self.cat_model_list = cat_model_list
        self.im_list = im_list
        self.transform = transform
        self.loader = loader

    def __getitem__(self):
        cur_n_views = random.randint(self.max_views) + 1
        filenames = random.choice(self.im_list[self.cur_idx], cur_n_views, replace=False)
        imgs = torch.zeros(cur_n_views, 3, 128, 128)        
        try:
            for view in range(cur_n_views):
                imgs[view,:,:,:] = self.loader(os.path.join(self.root, self.cat_model_list[self.cur_idx][0], self.cat_model_list[self.cur_idx][1], filenames[view]))
                if self.transform is not None:
                    imgs[view,:,:,:] = self.transform(imgs[view,:,:,:])
        except:
            return imgs

        return imgs

    def __len__(self):
        return len(self.im_list)
        
#    def fetch_batch(self):
#        cur_n_views = random.randint(self.max_views) + 1
#        image_files = random.choice(self.im_list[self.cur_idx], cur_n_views, replace=False)
#        batch_img = torch.zeros(cur_n_views, self.batch_size, 3, 128, 128)
#        for view in range(cur_n_views):
            
        
    
        

#class renderedViewsSampler(data.Sampler):
    