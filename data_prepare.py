import os
import json
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset


"""
idx_to_label = {
    0: 'anger',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'suprise',
}
labels_en = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
"""


class MMSAATBaselineDataset(Dataset):
    def __init__(self, stage, dataset_dir):
        self.stage = stage
        self.dataset_dir = dataset_dir
        self.dataset_path = self.dataset_dir + self.stage + '.json'
        self.filename_label_list = []
        with open(self.dataset_path) as f:
            for example in json.load(f):
                a = example['audio_file'].replace('.wav', '')
                v = example['video_file']
                self.filename_label_list.append((a, v, example['txt_label'], example['audio_label'], example['visual_label'], example['video_label']))
        self.labels_ch = ['愤怒', '厌恶', '恐惧', '高兴', '平静', '悲伤', '惊奇']

    def __len__(self):
        return len(self.filename_label_list)

    def __getitem__(self, idx):
        current_filename, current_filename_v, label_t, label_a, label_v, label_m = self.filename_label_list[idx]
        text_vector = np.load(self.dataset_dir + 'text/' + self.stage + '/' + current_filename + '.npy')
        text_vector = torch.from_numpy(text_vector)
        video_vector = np.load(self.dataset_dir + 'visual/' + self.stage + '/' + current_filename + '.mp4.npy')
        video_vector = torch.from_numpy(video_vector)
        audio_vector = np.load(self.dataset_dir + 'audio/' + self.stage + '/' + current_filename + '.npy') 
        audio_vector = torch.from_numpy(audio_vector)
        return  text_vector, audio_vector, video_vector, \
            self.labels_ch.index(label_t), self.labels_ch.index(label_a), \
            self.labels_ch.index(label_v), self.labels_ch.index(label_m)
        
        
class CHSIMSDataset(Dataset):
    def __init__(self, stage, dataset_dir):
        self.stage = stage
        self.dataset_dir = dataset_dir
        self.dataset_path = self.dataset_dir + 'unaligned_39.pkl'
        with open(self.dataset_path, 'rb') as f:
            data = pickle.load(f)
        self.text = data[self.stage]['text'].astype(np.float32)
        self.video = data[self.stage]['vision'].astype(np.float32)
        self.audio = data[self.stage]['audio'].astype(np.float32)
        self.labels = {
            'M': np.array(data[self.stage]['regression_labels']).astype(np.float32)
        }
        for m in "TAV":
            self.labels[m] = data[self.stage]['regression_labels_' + m].astype(np.float32)
        for m in "TAVM":
            self.labels[m] = np.select([self.labels[m] > 0.6, np.logical_and(self.labels[m] >= 0.2, self.labels[m] <= 0.6), self.labels[m] == 0., np.logical_and(self.labels[m] >= -0.6, self.labels[m] <= -0.2), self.labels[m] < -0.6], [4, 3, 2, 1, 0]) # acc5
           
    def __len__(self):
        return self.labels['M'].shape[0]
       
    def __getitem__(self, idx):
        text_vector = self.text[idx]
        text_vector = torch.from_numpy(text_vector)
        video_vector = self.video[idx]
        video_vector = torch.from_numpy(video_vector)
        audio_vector = self.audio[idx]
        audio_vector = torch.from_numpy(audio_vector)
        return text_vector, audio_vector, video_vector, self.labels['T'][idx], self.labels['A'][idx], self.labels['V'][idx], self.labels['M'][idx] 

        
        
        