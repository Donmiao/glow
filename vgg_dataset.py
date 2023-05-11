# vgg_dataset.py

import os
import tensorflow as tf
import numpy as np
from PIL import Image
from torch import tensor
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import rasterio
from rasterio.plot import show

class MyDataset(Dataset):
    def __init__(self, type, img_size, data_dir):
        #self.name2label = {"neutral": 0, "angry": 1, "depressed": 2, "drunk": 3, "happy": 4, "heavy": 5, "hurried": 6, "lazy" : 7, "old": 8, "proud": 9, "robot": 10, "sneaky": 11, "soldier": 12, "strutting": 13, "zombie": 14}
        self.name2label = {"neutral": 0, "angry": 1, "depressed": 2, "happy": 3, "heavy": 4, "old": 5, "proud": 6, "strutting": 7}
        self.img_size = img_size
        self.data_dir = data_dir
        self.data_list = list()
        for file in os.listdir(self.data_dir):
            if(".tiff" in file):
                self.data_list.append(os.path.join(self.data_dir, file))
            else:
                print(f"Not valid tiff file! {file}")
        print("Load {} Data Successfully!".format(type))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        file = self.data_list[item]
        #img = Image.open(file)
        with rasterio.open(file) as image:
            image_array = image.read()
        img = ToTensor()(image_array)
        img = np.array(img, np.float32).transpose(1, 2, 0)
        #img = np.array(img, np.float32).transpose(2, 0, 1)
        # if len(img.split()) == 1:
        #     img = img.convert("RGB")
        #img = img.resize((self.img_size,self.img_size))
        label = self.name2label[os.path.basename(file).split('-')[0]]
        # image_to_tensor = ToTensor()
        # img = image_to_tensor(img)
        #print(f"tensor: {label}")
        label = tensor(label)
        label_name = os.path.basename(file).split('-')[0]
        #print(f"tensor tensor: {label}")
        #label_ = [np.array(label)]
        label_ = [int(label)]
        #label_ =(tf.cast(tf.reshape(label_name, shape=[]), dtype=tf.int32))
        #print(f"label_ : {label_}")
        return img, label_ #label#, label_name #label
