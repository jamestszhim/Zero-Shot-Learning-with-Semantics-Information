import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
import pickle
import numpy as np
import pandas as pd

from torchtext import *
from torchtext.data import *

import nltk
nltk.download('punkt')
from nltk import word_tokenize


def load_cls_dataset():
    txt_field = data.Field(tokenize=word_tokenize, lower=True, include_lengths=True, batch_first=True)
    label_field = data.Field(sequential=False, unk_token=None)
    source_field = data.Field(sequential=False, unk_token=None)

    # make splits for data
    train, test= TabularDataset.splits(path='./Dataset/emotion', train='train_full.csv', test='test_full.csv',format='csv', 
                                      fields=[('label', label_field),('source', source_field), ('sentence', txt_field)], skip_header=True)
        
    # build the vocabulary on the training set only
    txt_field.build_vocab(train, min_freq=3)
    label_field.build_vocab(train)
    source_field.build_vocab(train)
    # make iterator for splits
    train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=32, sort_key=lambda x: len(x.sentence),sort_within_batch=True)
    
    print(f'Number of vocab: {len(txt_field.vocab)}')
    print(f'Number of training samples: {len(train.examples)}')
    print(f'Number of testing samples: {len(test.examples)}')

    print(f'Example of training data:\n {vars(train.examples[0])}\n')
    print(f'Example of testing data:\n {vars(test.examples[1])}\n')
    
    return train_iter, test_iter, txt_field, label_field






class CustomDataset(Dataset):

    def __init__(self, df):
        self.dataset_df = df

    def __len__(self):
        return len(self.dataset_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.dataset_df.iloc[idx]
        sentvec_tensor = torch.tensor(row['sentvec'])
        label = torch.tensor(row['label'])
        sample = {'label': label, 'sentvec': sentvec_tensor}
        return sample
    
    def get_class_labels(self):
        return list(self.dataset_df['label'].unique())


class DatasetIterator(object):
    def __init__(self, dataset_iter_dict, label2idx):
        self.dataset_iter_dict = dataset_iter_dict
        self.label2idx = label2idx

    def __iter__(self):
        
        # train_0 dataset
        if len(self.dataset_iter_dict.keys()) == 5:
            
            for batch_fear, batch_shame,  batch_sadness, batch_anger, batch_love in zip(iter(self.dataset_iter_dict[self.label2idx['joy']]), 
                                                                                       iter(self.dataset_iter_dict[self.label2idx['fear']]),
                                                                                       iter(self.dataset_iter_dict[self.label2idx['sadness']]), 
                                                                                       iter(self.dataset_iter_dict[self.label2idx['anger']]),
                                                                                       iter(self.dataset_iter_dict[self.label2idx['love']])):
                batch_size = len(batch_fear['sentvec'])
                labels = [self.label2idx['joy']]*batch_size + [self.label2idx['fear']]*batch_size + [self.label2idx['sadness']]*batch_size + [self.label2idx['anger']]*batch_size + [self.label2idx['love']]*batch_size 
                sentences = torch.cat([batch_fear['sentvec'],batch_shame['sentvec'],batch_sadness['sentvec'],batch_anger['sentvec'],batch_love['sentvec']], 0)
                if len(batch_fear['sentvec']) == len(batch_shame['sentvec']) ==len(batch_sadness['sentvec']) ==len(batch_anger['sentvec']) ==len(batch_love['sentvec']):
                    yield  {'label':torch.LongTensor(labels), 'sentvec': sentences}
         
        # full train dataset
        elif len(self.dataset_iter_dict.keys()) == 9:
            labels = list(self.dataset_iter_dict.keys())
            for batch_0, batch_1,  batch_2, batch_3, batch_4, batch_5,  batch_6, batch_7, batch_8 in zip(iter(self.dataset_iter_dict[labels[0]]), iter(self.dataset_iter_dict[labels[1]]), 
                                                                                                         iter(self.dataset_iter_dict[labels[2]]), iter(self.dataset_iter_dict[labels[3]]), 
                                                                                                         iter(self.dataset_iter_dict[labels[4]]), iter(self.dataset_iter_dict[labels[5]]), 
                                                                                                         iter(self.dataset_iter_dict[labels[6]]), iter(self.dataset_iter_dict[labels[7]]), 
                                                                                                         iter(self.dataset_iter_dict[labels[8]])):
                batch_size = len(batch_0['sentvec'])
                labels = torch.cat([batch_0['label'],batch_1['label'],batch_2['label'],batch_3['label'],batch_4['label'],batch_5['label']
                                       ,batch_6['label'],batch_7['label'],batch_8['label']],0)
                
                sentences = torch.cat([batch_0['sentvec'],batch_1['sentvec'],batch_2['sentvec'],batch_3['sentvec'],batch_4['sentvec'],batch_5['sentvec']
                                       ,batch_6['sentvec'],batch_7['sentvec'],batch_8['sentvec']],0)
                
                if (len(batch_0['sentvec'])==len(batch_1['sentvec'])==len(batch_1['sentvec'])==len(batch_2['sentvec'])==len(batch_3['sentvec'])==len(batch_4['sentvec'])==len(batch_5['sentvec'])==len(batch_6['sentvec'])==len(batch_7['sentvec'])==len(batch_8['sentvec'])):
                    yield  {'label':torch.LongTensor(labels), 'sentvec': sentences}




class EmotionDataset(object):
    def __init__(self, batch_size):
        self.root = 'Dataset/emotion/'
        self.batch_size = batch_size
        with open(self.root + 'idx2label.pkl', 'rb') as f:
            self.idx2label = pickle.load(f)

        with open(self.root + 'label2idx.pkl', 'rb') as f:
            self.label2idx = pickle.load(f)

        with open(self.root + 'labelvec.pkl', 'rb') as f:
            self.labelvec = pickle.load(f)
    
        self.train_0_df = pd.read_pickle(self.root+'train_0.pkl') 
        self.train_df = pd.read_pickle(self.root+'train_full.pkl') 
        self.test_df = pd.read_pickle(self.root+'test_full.pkl') 
        self.sup_df = pd.read_pickle(self.root+'support.pkl') 
        self.noval_df = pd.read_pickle(self.root+'noval.pkl') 
        
        
        self.sup_labels = torch.LongTensor(self.sup_df['label'].unique())
        self.noval_labels = torch.LongTensor(self.noval_df['label'].unique())
    
        print('classees of train_0:', ', '.join(np.sort([self.idx2label[i] for i in self.train_0_df['label'].unique()])))
        print('classees of support:', ', '.join(np.sort([self.idx2label[i] for i in self.sup_df['label'].unique()])))
        print('classees of noval:   ', ', '.join(np.sort([self.idx2label[i] for i in self.noval_df['label'].unique()])))
        
        
    def load_datasets(self, dataset_df, train):
        if train:
            class_labels = dataset_df['label'].unique()
            dataset_dict = {}
            for i in range(len(class_labels)):
                dataset_dict[class_labels[i]] = CustomDataset(dataset_df[dataset_df['label']==class_labels[i]])

            dataset_iter_dict = {}
            for label in dataset_dict:
                dataset_iter_dict[label] = DataLoader(dataset=dataset_dict[label], 
                                                      sampler=RandomSampler(data_source=dataset_dict[label],replacement=False) , 
                                                      batch_size=self.batch_size)
                
            iters = DatasetIterator(dataset_iter_dict, self.label2idx)
        else:
            dataset = CustomDataset(dataset_df)
            iters = DataLoader(dataset,shuffle=False,batch_size=60)
        return iters

       
    def get_iters(self):                                                                                                             
        train0_iter = self.load_datasets(self.train_0_df, True) 
        train_iter = self.load_datasets(self.train_df, True) 
        test_iter = self.load_datasets(self.test_df, False) 
        sup_iter = self.load_datasets(self.sup_df, False)
        noval_iter = self.load_datasets(self.noval_df, False)

        return train_iter, test_iter, train0_iter, sup_iter, noval_iter, self.idx2label, self.label2idx, torch.tensor(self.labelvec), self.sup_labels, self.noval_labels