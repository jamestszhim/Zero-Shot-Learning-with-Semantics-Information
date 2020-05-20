# from sentence_transformers import SentenceTransformer
# import wikipedia
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def unit_norm(x):
    u = []
    for xi in x:
        u.append(xi/np.linalg.norm(x))
    return np.array(u)

# def get_wiki_text(dataroot, sentence=3):
#     label_text = np.loadtxt(dataroot+'/classes.txt', str)
#     animals = label_text.T[1]
#     sentences = []
#     for animal in animals:
#         def disambigous(x):
#             if x == 'blue whale':
#                 return 'blue whales'
#             elif x == 'rat':
#                 return 'rats'
#             elif x == 'buffalo':
#                 return 'water buffalo'
#             elif x == 'pig':
#                 return 'pigs'
#             elif x == 'dolphin':
#                 return 'oceanic dolphins'
#             else:
#                 return x
#         s = wikipedia.summary(disambigous(animal), sentences=sentence)
#         sentences.append([animal, s])
#     return sentences

# def get_embeddings(sentences):
#     model = SentenceTransformer('bert-base-nli-mean-tokens')
#     sentence_embeddings = model.encode(sentences)
#     sentence_embeddings = np.array(sentence_embeddings)
#     print(f'Encoded shape: {sentence_embeddings.shape}')
#     return sentence_embeddings

def cosine_similarity(X, y):
    Xm = np.dot(X, y)
    norm_p = np.linalg.norm(X, axis=1) * np.linalg.norm(y)
    distances = Xm/norm_p
    distances = (distances - min(distances)) / (max(distances) - min(distances))
    return distances

def adjusted_cosine_similarity(X, y):
    # X: known labels
    # y: query label
    mu = X.T.mean(axis=-1)
    Xhat = X - mu
    yhat = y - mu
    Xm = np.dot(Xhat, yhat)
    norm_p = np.linalg.norm(Xhat, axis=1) * np.linalg.norm(yhat)
    distances = Xm/norm_p
    distances = (distances - min(distances)) / (max(distances) - min(distances))
    return distances

def cos_loss(input, target):
    return 1 - F.cosine_similarity(input, target).mean()

class TripletLoss(nn.Module):
    def __init__(self, Y, margin=0.1):
        super(TripletLoss, self).__init__()
        self.Y = Y # known embeddings
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, y_idxs):

        n = inputs.size(0)  # batch_size
        d = len(self.Y) # number of known embeddings

        dist_ap, dist_an = [], []
        for i in range(n):
            y = y_idxs[i]
            ij = torch.randperm(d)
            for iter in range(d):
                j = ij[iter]
                if j == y:
                    continue
                else:
                    dist_ap.append(self.Y[y].dot(inputs[i]).unsqueeze(0))  # positive anchor
                    dist_an.append(self.Y[j].dot(inputs[i]).unsqueeze(0))  # negative anchor
                if self.Y[y].dot(inputs[i]) - self.Y[j].dot(inputs[i]) > self.margin:
                    break
        dist_ap = torch.cat(dist_ap) 
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_ap, dist_an, y)
        return loss

def get_euclidean_dist(curr_labels, class_labels):
    return torch.sqrt(torch.sum((curr_labels - class_labels)**2))

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res