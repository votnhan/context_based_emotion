import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast
from PIL import Image

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

def read_json(filename):
    with open(filename, 'r') as fp:
        content =  json.load(fp)

    return content

def write_json(filename, content_dict, log=True):
    with open(filename, 'w') as fp:
        json.dump(content_dict, fp)

    if log:
        print('Write json file {}'.format(filename))

def create_folder(path):
    path = str(path)
    if not os.path.exists(path):
        os.makedirs(path)

def save_pandas_df(data, filename, columns, index=None, use_index=True):
    df = pd.DataFrame(data=data, index=index, columns=columns)
    df.to_csv(filename, index=use_index)

def read_image(image_path):
    image = Image.open(image_path)
    return image

def append_log_to_file(file_path, list_items):
    with open(file_path, 'a') as opened_file:
        line_items = ','.join(list_items)
        opened_file.write(line_items+'\n')
        opened_file.close()

def plot_train_val_loss(log_file, out_file):
    df = pd.read_csv(log_file, index_col='Epoch')
    plt.plot(df['Train_loss'].values, label='Training loss')
    plt.plot(df['Validation_loss'].values, label='Validation loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.savefig(out_file)
    print('Plot train and val loss to {}'.format(out_file))

def get_weight_of_samples(train_df, weight_class_file, cat2idx):
    weight_class = np.load(weight_class_file)
    sample_weights = []
    for labels_str in train_df['Categorical_Labels']:
        labels = ast.literal_eval(labels_str)
        weights = 0.0
        for label in labels:
            weights += weight_class[cat2idx[label]]
        sample_weights.append(weights)
    
    return sample_weights

