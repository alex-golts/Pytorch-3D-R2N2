import os
import os.path

import torch
from torchvision import transforms
import torch.utils.data as data
from PIL import Image
from numpy import random
from binvox_rw import read_as_3d_array

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


def loader_image(path):
    return Image.open(path).convert('RGB')
    
def loader_label(path):
    with open(path, 'rb') as f:
        voxel = read_as_3d_array(f)
    return voxel


def portion_models(dataset_portion, category_path):
    '''
    Load category, model names from a shapenet dataset.
    '''

    def model_names(model_path):
        """ Return model names"""
        model_dir_list = os.listdir(model_path)
        model_names = [name for name in model_dir_list
                       if os.path.isdir(os.path.join(model_path, name))]
        return sorted(model_names)

    model_path = os.path.join(category_path)
    models = model_names(model_path)
    num_models = len(models)

    portioned_models = models[int(num_models * dataset_portion[0]):int(num_models * dataset_portion[1])]

    return portioned_models

class Dataset(data.Dataset):

    def __init__(self, root, transform=None, loader_image=loader_image, loader_label=loader_label, model_portion=[0, 0.8], min_views=1, max_views=5, batch_size=24):
        image_dict = {}        
        image_list = []   
        cat_model_list = []
        im_list = []
        image_root = os.path.join(root, 'ShapeNetRendering')
        label_root = os.path.join(root, 'ShapeNetVox32')
        main_dir_list = os.listdir(image_root)
        for directory in main_dir_list: # loop over model-categories
            image_dict[directory] = {}
            model_list = portion_models(model_portion, os.path.join(image_root,directory))
            print('Directory: ' + directory + ', # of models: ' + str(len(model_list)))
            for subdirectory in model_list: # loop over models
                image_dict[directory][subdirectory] = []        
                cat_model_list.append([directory, subdirectory])
                im_list_cur = []
                sub_dir_list = [f for f in os.listdir(os.path.join(image_root,directory,subdirectory,'rendering')) if is_image_file(f)]
                for filename in sub_dir_list: # loop over image files
                    image_list.append('{}'.format(os.path.join(directory,subdirectory,'rendering',filename)))
                    image_dict[directory][subdirectory].append('{}'.format(filename))
                    im_list_cur.append(filename)
                im_list.append(im_list_cur)
## Simple image folder case:        
#        for filename in os.listdir(root):
#            if is_image_file(filename):
#                images.append('{}'.format(filename))        
        #combined = list(zip(cat_model_list, im_list))
        #random.shuffle(combined)        
        #cat_model_list[:], im_list[:] = zip(*combined)
        
        self.min_views = min_views
        self.max_views = max_views
        #self.cur_idx = 0
        self.image_root = image_root
        self.label_root = label_root
        self.image_list = image_list
        self.image_dict = image_dict
        self.cat_model_list = cat_model_list
        self.im_list = im_list
        self.transform = transform
        self.loader_image = loader_image
        self.loader_label = loader_label
        self.batch_size = batch_size
        self.cur_index_within_batch = self.batch_size

    def __getitem__(self, index):  
        # index indicates the model id (model id's are randomly shuffled)
        if self.cur_index_within_batch == self.batch_size:
            self.cur_index_within_batch = 0
            self.cur_n_views = random.randint(self.min_views, self.max_views+1)
        self.cur_index_within_batch += 1
        # the specific images within the chosen model are chosen at random
        filenames = random.choice(self.im_list[index], self.cur_n_views, replace=False)
        imgs = torch.zeros(self.cur_n_views, 3, 128, 128)  
        label = torch.zeros(32,32,32, dtype=torch.long)
        try:
            labeltmp = self.loader_label(os.path.join(self.label_root, self.cat_model_list[index][0], self.cat_model_list[index][1], 'model.binvox'))                      
            label = torch.from_numpy(labeltmp.data.astype('uint8')).long()
            for view in range(self.cur_n_views):
                imgtmp = self.loader_image(os.path.join(self.image_root, self.cat_model_list[index][0], self.cat_model_list[index][1], 'rendering', filenames[view]))                
                if self.transform is not None:
                    imgs[view,:,:,:] = self.transform(imgtmp)
                
        except:
            print('PROBLEM WITH LOADING A BATCH')
            pass

        return {'imgs': imgs, 'label': label}

    def __len__(self):
        return len(self.im_list)
        
