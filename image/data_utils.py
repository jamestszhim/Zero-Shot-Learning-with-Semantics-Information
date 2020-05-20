import os
import sys
import torch
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from utility import unit_norm

class AwaDataset(Dataset):

    def __init__(self, dataroot, attribute_embedding='bin', split=1):
        assert attribute_embedding in ['bin', 'cont', 'text', 'weighted_text', 'label', 'bert_text', 'bert_wiki', 'random']
        assert split in [1, 2, 3]
        attribute_dir = 'extracted_attributes'

        self.embedding = attribute_embedding
        self.featurs = pd.read_csv(os.path.join(
            dataroot, 'Features/ResNet101/AwA2-features.txt'), sep=' ', header=None)
        self.labels = np.loadtxt(os.path.join(
            dataroot, 'Features/ResNet101/AwA2-labels.txt'))
        if attribute_embedding == 'bin':
            self.text_labels = np.loadtxt(os.path.join(
                dataroot, 'predicate-matrix-binary.txt'))
        elif attribute_embedding == 'cont':
            self.text_labels = np.loadtxt(os.path.join(
                dataroot, 'predicate-matrix-continuous.txt'))
            self.text_labels = MinMaxScaler().fit_transform(self.text_labels)
        elif attribute_embedding == 'text':
            self.text_labels = np.loadtxt(os.path.join(
                dataroot, attribute_dir, 'predicate-embedding.txt'))
            self.text_labels = MinMaxScaler().fit_transform(self.text_labels)
        elif attribute_embedding == 'weighted_text':
            text_weights = np.loadtxt(os.path.join(
                dataroot, 'predicate-matrix-continuous.txt'))
            text_weights = MinMaxScaler().fit_transform(text_weights)
            text_vec = np.loadtxt(os.path.join(dataroot, attribute_dir, 'predicates2vec.txt'))
            self.text_labels = unit_norm(text_weights.dot(text_vec))
        elif attribute_embedding == 'label':
            self.text_labels = np.loadtxt(os.path.join(
                dataroot, attribute_dir, 'label-vector.txt'))
            self.text_labels = unit_norm(self.text_labels)
        elif attribute_embedding == 'bert_text':
            self.text_labels = np.loadtxt(os.path.join(
                dataroot, attribute_dir, 'bert-predicates-text.txt'))
            self.text_labels = unit_norm(self.text_labels)
            # self.text_labels = MinMaxScaler().fit_transform(self.text_labels)
        elif attribute_embedding == 'bert_wiki':
            self.text_labels = np.loadtxt(os.path.join(
                dataroot, attribute_dir, 'wiki_embedding_5.txt'))
            self.text_labels = unit_norm(self.text_labels)
            # self.text_labels = MinMaxScaler().fit_transform(self.text_labels)
        elif attribute_embedding == 'random':
            self.text_labels = np.random.uniform(-1,1,(50,300))
            self.text_labels = unit_norm(self.text_labels)
            
        self.class2label = dict(np.genfromtxt(
            os.path.join(dataroot, 'classes.txt'), dtype='str'))
        self.label2class = {v: k for k, v in self.class2label.items()}

        train_labels = np.loadtxt(os.path.join(dataroot, f'Split/SS/trainclasses{split}.txt'), dtype=str)
        valid_labels = np.loadtxt(os.path.join(dataroot, f'Split/SS/valclasses{split}.txt'), dtype=str)
        test_labels = np.loadtxt(os.path.join(dataroot, f'Split/SS/testclasses.txt'), dtype=str)
        self.support_classes = [int(self.label2class.get(key)) for key in train_labels]
        self.novel_classes = [int(self.label2class.get(key)) for key in valid_labels]
        self.test_classes = [int(self.label2class.get(key)) for key in test_labels]
        self.mix_classes = self.support_classes + self.novel_classes
        support_idx = np.where(np.isin(self.labels, self.support_classes))[0]
        self.test_idx = np.where(np.isin(self.labels, self.test_classes))[0]
        self.novel_idx = np.where(np.isin(self.labels, self.novel_classes))[0]
        
        self.support_train_idx, self.support_test_idx = train_test_split(
                                                    support_idx,
                                                    test_size=0.2,
                                                    shuffle=True,
                                                    stratify=self.labels[support_idx])
        
        self.mix_idx = np.concatenate((self.support_test_idx, self.novel_idx), axis=None)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        X = np.array(self.featurs.iloc[idx, :]).astype(np.float32)
        y = int(self.labels[idx]) # remember y starts from 1 instead of 0
        z = self.text_labels[y-1].astype(np.float32)

        return X, z, y

class ReducedAwaDataset(Dataset):

    def __init__(self, dataroot, attribute_embedding='bin'):
        self.featurs = pd.read_csv(os.path.join(
            dataroot, 'Features/ResNet101/reduced_AwA2-features.txt'), sep=' ', header=None)
        self.labels = np.loadtxt(os.path.join(
            dataroot, 'Features/ResNet101/reduced_AwA2-labels.txt'))
        if attribute_embedding == 'bin':
            self.text_labels = np.loadtxt(os.path.join(
                dataroot, 'predicate-matrix-binary.txt'))
        elif attribute_embedding == 'cont':
            self.text_labels = np.loadtxt(os.path.join(
                dataroot, 'predicate-matrix-continuous.txt'))
            self.text_labels = MinMaxScaler().fit_transform(self.text_labels)
            
        
        ###### This is set manually by inspecting the data file #####
        support_split = 6982
        self.support_classes = [1, 30,  4,  9, 41, 37, 33, 25, 46, 49]
        self.novel_classes = [5, 40, 50, 19, 22]
        self.mix_classes = self.support_classes + self.novel_classes
        #############################################################
        
        self.support_train_idx, self.support_test_idx = train_test_split(
                                                    np.arange(support_split),
                                                    test_size=0.2,
                                                    shuffle=True,
                                                    stratify=self.labels[:support_split])
        
        self.novel_idx = np.arange(support_split, len(self.labels))
        self.mix_idx = np.arange(len(self.labels))
        self.class_map = dict(np.genfromtxt(
            os.path.join(dataroot, 'classes.txt'), dtype='str'))

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        X = np.array(self.featurs.iloc[idx, :]).astype(np.float32)
        y = int(self.labels[idx]) # remember y starts from 1 instead of 0
        z = self.text_labels[y-1].astype(np.float32)

        return X, z, y

def get_data_snapshot(dataroot, dataset='awa', attribute_embedding='bin', split=1, bs=128):
    data = AwaDataset(dataroot, attribute_embedding, split)
    feature, text_feature, label = data[0]

    train_sampler = SubsetRandomSampler(data.support_train_idx)
    test_sampler = SubsetRandomSampler(data.support_test_idx)
    bm_sampler = SubsetRandomSampler(data.test_idx)
    novel_sampler = SubsetRandomSampler(data.novel_idx)
    mix_sampler = SubsetRandomSampler(data.mix_idx) # predict in joint space

    train_loader = DataLoader(data, batch_size=bs,
                              sampler=train_sampler,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(data, batch_size=bs,
                             sampler=test_sampler,
                             num_workers=4, pin_memory=True)
    novel_loader = DataLoader(data, batch_size=bs,
                             sampler=novel_sampler,
                              num_workers=4, pin_memory=True)
    bm_loader = DataLoader(data, batch_size=bs,
                             sampler=bm_sampler,
                              num_workers=4, pin_memory=True)
    mix_loader = DataLoader(data, batch_size=bs,
                            sampler = mix_sampler,
                            num_workers=4, pin_memory=True)

    print('AWA2 Dataset')
    print(f'|Train:\t\t{len(data.support_train_idx)} ({len(data.support_classes)} classes)')
    print(f'|Test:\t\t{len(data.support_test_idx)}')
    print(f'|Novel:\t\t{len(data.novel_idx)} ({len(data.novel_classes)} classes)')

    snapshot = {'train_loader': train_loader,
                'test_loader': test_loader,
                'novel_loader': novel_loader,
                'test_loader': test_loader,
                'bm_loader': bm_loader,
                'mix_loader': mix_loader,
                'd_attribute': len(text_feature),
                'd_input': len(feature),
                'embedding': data.embedding,
                'text_labels': data.text_labels,
                'support_classes': np.array(data.support_classes),
                'novel_classes': np.array(data.novel_classes),
                'test_classes': np.array(data.test_classes),
                'mix_classes': np.array(data.mix_classes),
                'class2label': data.class2label}

    return snapshot