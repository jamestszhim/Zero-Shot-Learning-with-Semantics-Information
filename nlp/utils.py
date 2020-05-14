from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from tqdm.notebook import tqdm

import torch
from torch import nn
import torch.nn.functional as F


    
    
def plot_confusion_matrix(y_true, y_pred, title, saved_name, labels):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    df_cm = pd.DataFrame(cm, index = labels,
                      columns = labels)
    plt.figure(figsize = (10,7))
    fig = sn.heatmap(df_cm, annot=True, cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted label')
    plt.xticks(rotation=-45)
    plt.yticks(rotation=0)
    fig.get_figure().savefig('results/' + saved_name + '.png', dpi=400)
    return cm
    
    
    
def count_parameters(model):
    temp = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model architecture:\n\n', model)
    print(f'\nThe model has {temp:,} trainable parameters')
    
    
def get_acc(true, predict, k):
    correct = 0.0
    total = 0.0
    for i in range(len(true)):
        if true[i] in predict[i,:k]:
            correct+=1
        total+=1
    return correct/total


def plot_loss(record):
    plot_set = list(record.TRAIN_LOSS.keys())
    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'lime', 'pink', 'skyblue', 'peru', 'orange', 'darkkahaki', 'plum' ]
    plt.figure(figsize=(15,10))
    for i in range(len(plot_set)):
        x = np.arange(len(record.TRAIN_LOSS[plot_set[i]]))
        plt.plot(x, record.TRAIN_LOSS[plot_set[i]], label=str(plot_set[i])+' train' , c=color[i])
        plt.plot(x, record.NOVAL_LOSS[plot_set[i]], '-.', label=str(plot_set[i])+' noval', c=color[i])
        plt.plot(x, record.SUP_LOSS[plot_set[i]], '--', label=str(plot_set[i])+' sup', c=color[i])

    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.legend()
    plt.show()
    
    
def plot_acc(record):

    plot_set = list(record.TRAIN_ACC_1.keys())
    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'lime', 'pink', 'skyblue', 'peru', 'orange', 'darkkahaki', 'plum' ]
    plt.figure(figsize=(15,10))
    for i in range(len(plot_set)):
        x = np.arange(len(record.TRAIN_LOSS[plot_set[i]]))
        plt.plot(x, record.TRAIN_ACC_1[plot_set[i]], label=str(plot_set[i])+' train' , c=color[i])
        plt.plot(x, record.NOVAL_ACC_1[plot_set[i]], '-.', label=str(plot_set[i])+' noval', c=color[i])
        plt.plot(x, record.SUP_ACC_1[plot_set[i]], '--', label=str(plot_set[i])+' sup', c=color[i])
    plt.title('top1 acc')
    plt.xlabel('Epoch')
    plt.ylabel('Average ACC')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(15,10))
    for i in range(len(plot_set)):

        plt.plot(x, record.TRAIN_ACC_2[plot_set[i]], label=str(plot_set[i])+' train' , c=color[i])
        plt.plot(x, record.NOVAL_ACC_2[plot_set[i]], '-.', label=str(plot_set[i])+' noval', c=color[i])
        plt.plot(x, record.SUP_ACC_2[plot_set[i]], '--', label=str(plot_set[i])+' sup', c=color[i])
    plt.title('top2 acc')
    plt.xlabel('Epoch')
    plt.ylabel('Average ACC')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(15,10))
    for i in range(len(plot_set)):

        plt.plot(x, record.TRAIN_ACC_3[plot_set[i]], label=str(plot_set[i])+' train' , c=color[i])
        plt.plot(x, record.NOVAL_ACC_3[plot_set[i]], '-.', label=str(plot_set[i])+' noval', c=color[i])
        plt.plot(x, record.SUP_ACC_3[plot_set[i]], '--', label=str(plot_set[i])+' sup', c=color[i])
    plt.title('top3 acc')
    plt.xlabel('Epoch')
    plt.ylabel('Average ACC')
    plt.legend()
    plt.show()
    
    
    
    
class Record(object):
    def __init__(self):
        self.TRAIN_LOSS ={}
        self.TRAIN_True_Labels={}
        self.TRAIN_Predicted_Labels={}


        self.NOVAL_LOSS={}
        self.NOVAL_True_Labels={}
        self.NOVAL_Predicted_Labels={}

        self.SUP_LOSS={}
        self.SUP_True_Labels={}
        self.SUP_Predicted_Labels={}

        self.TRAIN_ACC_1 = {}
        self.NOVAL_ACC_1 = {}
        self.SUP_ACC_1 = {}
        
        self.TRAIN_ACC_2 = {}
        self.NOVAL_ACC_2 = {}
        self.SUP_ACC_2 = {}
        
        self.TRAIN_ACC_3 = {}
        self.NOVAL_ACC_3 = {}
        self.SUP_ACC_3 = {}
        
    def config_new_trial(self, model_name):
        self.TRAIN_LOSS[model_name] = [] 
        self.TRAIN_True_Labels[model_name] = [] 
        self.TRAIN_Predicted_Labels[model_name] = [] 
        
        self.NOVAL_LOSS[model_name] = [] 
        self.NOVAL_True_Labels[model_name] = [] 
        self.NOVAL_Predicted_Labels[model_name] = [] 

        self.SUP_LOSS[model_name] = [] 
        self.SUP_True_Labels[model_name] = [] 
        self.SUP_Predicted_Labels[model_name] = [] 

        self.TRAIN_ACC_1[model_name] = [] 
        self.NOVAL_ACC_1[model_name] = [] 
        self.SUP_ACC_1[model_name] = [] 
        
        self.TRAIN_ACC_2[model_name] = [] 
        self.NOVAL_ACC_2[model_name] = [] 
        self.SUP_ACC_2[model_name] = [] 
        
        self.TRAIN_ACC_3[model_name] = [] 
        self.NOVAL_ACC_3[model_name] = [] 
        self.SUP_ACC_3[model_name] = [] 
        
    def gen_acc(self, model_name):
        num_epoch = len(self.TRAIN_LOSS[model_name])
        for i in tqdm(range(num_epoch)):
            self.TRAIN_ACC_1[model_name].append(get_acc(self.TRAIN_True_Labels[model_name][i], self.TRAIN_Predicted_Labels[model_name][i], 1))
            self.NOVAL_ACC_1[model_name].append(get_acc(self.NOVAL_True_Labels[model_name][i], self.NOVAL_Predicted_Labels[model_name][i], 1))
            self.SUP_ACC_1[model_name].append(get_acc(self.SUP_True_Labels[model_name][i], self.SUP_Predicted_Labels[model_name][i], 1))
            
            self.TRAIN_ACC_2[model_name].append(get_acc(self.TRAIN_True_Labels[model_name][i], self.TRAIN_Predicted_Labels[model_name][i], 2))
            self.NOVAL_ACC_2[model_name].append(get_acc(self.NOVAL_True_Labels[model_name][i], self.NOVAL_Predicted_Labels[model_name][i], 2))
            self.SUP_ACC_2[model_name].append(get_acc(self.SUP_True_Labels[model_name][i], self.SUP_Predicted_Labels[model_name][i], 2))
            
            self.TRAIN_ACC_3[model_name].append(get_acc(self.TRAIN_True_Labels[model_name][i], self.TRAIN_Predicted_Labels[model_name][i], 3))
            self.NOVAL_ACC_3[model_name].append(get_acc(self.NOVAL_True_Labels[model_name][i], self.NOVAL_Predicted_Labels[model_name][i], 3))
            self.SUP_ACC_3[model_name].append(get_acc(self.SUP_True_Labels[model_name][i], self.SUP_Predicted_Labels[model_name][i], 3))
            
            

class Trainer(nn.Module):
    def __init__(self, labelvec, support_class, noval_class, device):
        super(Trainer, self).__init__()
        self.criterion = nn.CosineEmbeddingLoss(margin=0.5)
        self.label_embedder = nn.Embedding.from_pretrained(labelvec, freeze=True).to(device)
        self.num_class = len(labelvec)
        self.labelvec = labelvec.to(device)
        self.device = device
        self.labelvec_0 = labelvec.clone().to(device)
        self.labelvec_0[noval_class] = 0.0
        self.labelvec_1 = labelvec.clone().to(device)
        self.labelvec_1[support_class] = 0.0
        
        
    def get_cosine_dist(self, output, mode):
        # output: batch, 300
        batch_size = len(output)
        if mode=='train':
            labelvec_matrix = self.labelvec.unsqueeze(0).repeat(batch_size, 1, 1)
            
        elif mode=='support':
            labelvec_matrix = self.labelvec_0.unsqueeze(0).repeat(batch_size, 1, 1)
        
        elif mode=='noval':
            labelvec_matrix = self.labelvec_1.unsqueeze(0).repeat(batch_size, 1, 1)
            
        output_matrix = output.unsqueeze(1).repeat(1, self.num_class, 1)
        cosine_dist = F.cosine_similarity(output_matrix, labelvec_matrix, -1)
        return 1- cosine_dist
        
    
    def train(self, model, iterator, optimizer, num_negative_sampling=1, clip=1.0):
        model.train()
        epoch_loss = 0.0
        cumulated_num = 0
        predicted_labels = []
        true_labels = []

        for i, batch in enumerate(iterator):
            optimizer.zero_grad()

            src= batch['sentvec']
            label = batch['label']
            batch_size = len(label)

            src, label = src.to(self.device), label.long().to(self.device)

            temp = torch.rand(batch_size, self.num_class).to(self.device)
            temp[torch.arange(batch_size), label] = 1

            true_label_vec = self.label_embedder(label)
            negative_label = torch.topk(temp, dim=-1, k=8, largest=False)[1].to(self.device)
            

            output = model.forward(src)

            positive_loss = self.criterion(output, true_label_vec, torch.ones(batch_size).to(self.device))
            negative_loss = 0.0
            for j in range(num_negative_sampling):
                negative_label_vec = self.label_embedder(negative_label[:,j])
                negative_loss += self.criterion(output, negative_label_vec, -torch.ones(batch_size).to(self.device))
            loss = positive_loss + negative_loss/num_negative_sampling
            loss.backward()       

            if clip>=0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            distancs = self.get_cosine_dist(output.detach(), 'train')
            predicted_labels.append(torch.topk(distancs.detach(), k=5,dim=-1, largest=False)[1].cpu().numpy())
            true_labels.append(label.cpu().numpy())

            epoch_loss += loss.item()
            cumulated_num += batch_size

            print(f'\r{loss.item()} ', end='')

        return epoch_loss/cumulated_num, np.concatenate(true_labels), np.concatenate(predicted_labels, axis=0)


    def evaluate(self, model, iterator, topk, mode):

        model.eval()
        epoch_loss = 0.0
        cumulated_num = 0

        true_labels = []
        predicted_labels = []    

        with torch.no_grad():
            for i, batch in enumerate(iterator):

                src= batch['sentvec']
                label = batch['label']
                batch_size = len(label)
                src, label = src.to(self.device), label.long().to(self.device)
                true_label_vec = self.label_embedder(label)
                output = model.forward(src)

                loss = self.criterion(output, true_label_vec, torch.ones(batch_size).to(self.device))
                
                distancs = self.get_cosine_dist(output.detach(), mode)
                predicted_labels.append(torch.topk(distancs.detach(), k=topk, dim=-1, largest=False)[1].cpu().numpy())
                true_labels.append(label.cpu().numpy())
                
                epoch_loss += loss.item()
                cumulated_num += batch_size

        return epoch_loss/cumulated_num, np.concatenate(true_labels), np.concatenate(predicted_labels, axis=0)
    
    def run(self, model_name, num_epoch, record, model, optimizer, num_negative_sampling, train0_iter, sup_iter, noval_iter):
    
        record.config_new_trial(model_name)
        for i in tqdm(range(num_epoch)):

            loss, true_laebls, predicted_laebls = self.train(model, train0_iter, optimizer, num_negative_sampling=num_negative_sampling)
            record.TRAIN_LOSS[model_name].append(loss)
            record.TRAIN_True_Labels[model_name].append(true_laebls)
            record.TRAIN_Predicted_Labels[model_name].append(predicted_laebls)

            loss, true_laebls, predicted_laebls =  self.evaluate(model, noval_iter, 3, 'noval')
            record.NOVAL_LOSS[model_name].append(loss)
            record.NOVAL_True_Labels[model_name].append(true_laebls)
            record.NOVAL_Predicted_Labels[model_name].append(predicted_laebls)

            loss, true_laebls, predicted_laebls =  self.evaluate(model,sup_iter , 3, 'support')
            record.SUP_LOSS[model_name].append(loss)
            record.SUP_True_Labels[model_name].append(true_laebls)
            record.SUP_Predicted_Labels[model_name].append(predicted_laebls)
        record.gen_acc(model_name)