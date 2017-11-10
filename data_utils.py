import json
import os

import mxnet as mx
import numpy as np
from mxnet import gluon as gl
from mxnet import nd


class SceneDataSet(gl.data.Dataset):
    def __init__(self, json_file, img_path, transform):
        self._img_path = img_path
        self._transform = transform
        with open(json_file, 'r') as f:
            annotation_list = json.load(f)
        self._img_list = [[i['image_id'], i['label_id']]
                          for i in annotation_list]

    def __getitem__(self, idx):
        img_name = self._img_list[idx][0]
        label = np.float32(self._img_list[idx][1])
        img = mx.image.imread(os.path.join(self._img_path, img_name))
        img = self._transform(img)
        return img, label

    def __len__(self):
        return len(self._img_list)
