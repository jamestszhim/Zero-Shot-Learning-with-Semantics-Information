import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from model.mlp import MLP
from utility import *
from data_utils import get_data_snapshot

class ModelTrainer(object):
    def __init__(self, data_snapshot, gpu=False, topk=1, distance='cos'):
        self.bin_attribute = True if data_snapshot['embedding'] == 'bin' else False
        self.embedding = data_snapshot['embedding']
        model_config = {'d_input': data_snapshot['d_input'],
                        'n_class': data_snapshot['d_attribute']}
        self.classes = {'support': data_snapshot['support_classes'],
                        'novel': data_snapshot['novel_classes'],
                        'mix': data_snapshot['mix_classes'],
                        'test': data_snapshot['test_classes']}
        self.class2label = data_snapshot['class2label']
        self.train_loader = data_snapshot['train_loader']
        self.test_loader = data_snapshot['test_loader']
        self.novel_loader = data_snapshot['novel_loader']
        self.bm_loader = data_snapshot['bm_loader']
        self.mix_loader = data_snapshot['mix_loader']
        self.text_labels = data_snapshot['text_labels']
        self.device = torch.device('cuda') if gpu else torch.device('cpu')
        self.net = MLP(model_config).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters())
        self.rank_loss = False
        if self.embedding == 'bin':
            self.criterion = nn.BCELoss()
        elif self.embedding == 'cont':
            self.criterion = nn.MSELoss()
        else:
            self.rank_loss = True
            self.criterion = TripletLoss(torch.from_numpy(self.text_labels).float().to(self.device), margin=0.5)
        self.topk = topk
        self.dist = distance
    
    def attribute_to_class_prob(self, pred_labels, target_set='support'):
        assert target_set in ['support', 'novel', 'mix', 'test']
        target_classes = self.classes[target_set]
        probs = []
        known_labels = self.text_labels[target_classes-1] # -1 is needed because the target class start with 1 
        for pred_label_tensor in pred_labels:
            pred_label = pred_label_tensor.detach().cpu().numpy()
            if self.dist == 'l2':
                class_dist = np.linalg.norm(known_labels-pred_label, 2, axis=1)
                class_dist = 1.0/(class_dist+1e-8)
            elif self.dist == 'cos':
                class_dist = cosine_similarity(known_labels, pred_label)
            class_prob = softmax(class_dist)
            probs.append(class_prob.tolist())

        return np.array(probs)
    
    def _train(self, cur_epoch):
        self.net.train()
        running_loss = 0.0
        total, correct = 0, 0
        for i, (inputs, text_labels, labels) in enumerate(self.train_loader):
            inputs, text_labels = inputs.to(self.device), text_labels.to(self.device)
            outputs = self.net(inputs)
            if self.bin_attribute:
                outputs = torch.sigmoid(outputs)
            if self.rank_loss:
                loss = self.criterion(outputs, labels-1)
            else:
                loss = self.criterion(outputs, text_labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            
            # Acc
            prob = self.attribute_to_class_prob(outputs, 'support')
            predicted = self.classes['support'][np.argsort(-1*prob, axis=1)[:,:self.topk]]
            np_label = labels.cpu().numpy()
            for idx, pred in enumerate(predicted):
                correct += 1 if np_label[idx] in pred else 0
            total += labels.size(0)

        train_loss = running_loss/len(self.train_loader)
        print(f'==> Epoch {cur_epoch} loss: {train_loss:.2f} acc: {correct/total:.2f}')
        return train_loss

    def _test(self, target_set='support', feat_only=False):
        if target_set == 'support':
            data_loader = self.test_loader 
        elif target_set == 'novel':
            data_loader = self.novel_loader
        elif target_set == 'test':
            data_loader = self.bm_loader
        else:
            data_loader = self.mix_loader
        self.net.eval()
        total, correct = 0, 0
        with torch.no_grad():
            y_score = []
            y_pred = []
            y_test = []
            running_loss = 0.0
            for i, (inputs, text_labels, labels) in enumerate(data_loader):
                inputs, text_labels = inputs.to(self.device), text_labels.to(self.device)
                outputs = self.net(inputs)
                np_label = labels.cpu().numpy()
                if not feat_only:
                    if self.bin_attribute:
                        outputs = torch.sigmoid(outputs)
                    if self.rank_loss:
                        loss = self.criterion(outputs, labels-1)
                    else:
                        loss = self.criterion(outputs, text_labels)
                    running_loss += loss.item()
                    
                    # Acc
                    prob = self.attribute_to_class_prob(outputs, target_set)
                    predicted = self.classes[target_set][np.argsort(-1*prob, axis=1)[:,:self.topk]]
                    for idx, pred in enumerate(predicted):
                        correct += 1 if np_label[idx] in pred else 0
                    
                    y_pred.extend(predicted.tolist())
                
                total += labels.size(0)
                y_score.extend(outputs.detach().cpu().numpy())
                y_test.extend(np_label.tolist())
                
                
        test_loss = running_loss/len(data_loader)
        print(f'| ({target_set}) loss: {test_loss:.2f} acc: {correct/total:.2f}')
        return correct/total, (np.array(y_score), np.array(y_pred), np.array(y_test))

    def run_model(self, num_epochs):
        for i in range(num_epochs):
            train_loss = self._train(i)
            test_acc, _ = self._test('support')
            novel_acc, _ = self._test('novel')
            bm_acc, _ = self._test('test')
            mix_acc, _ = self._test('mix')
        return [test_acc, novel_acc, mix_acc, bm_acc]
    
    def plot_confusion_matrix(self, target_set='support'):
        _, (_, y_pred, y_test) = self._test(target_set)
        target_classes = self.classes[target_set]
        labels = [self.class2label.get(key) for key in list(map(str, np.sort(target_classes)))]
        n_class = len(target_classes)
        cmatrix = np.array(confusion_matrix(y_test, y_pred[:,0], normalize='true'))
        disp = ConfusionMatrixDisplay(cmatrix, labels)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title(f'Confusion Matrix for {target_set} class')   
        disp.plot(xticks_rotation=-45.0, ax=ax, cmap='Blues')
