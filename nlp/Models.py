from torch import nn
import torch.nn.functional as F



class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(768, 500)
        self.fc2 = nn.Linear(500, 300)
        self.bn1 = nn.BatchNorm1d(768)
        self.bn2 = nn.BatchNorm1d(500)
        self.drops = nn.Dropout(0.1)

    def forward(self, x):
        x = self.bn1(x)
        x = self.drops(x)
        x = F.relu(self.fc1(x))
        x = self.bn2(x)
        x = self.drops(x)
        x = self.fc2(x)
        return x
    
    
class LSTM_CLS(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, dropout, padding_idx, num_layer):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hid_dim*2*num_layer, 9)
        self.rnn = nn.GRU(input_size=emb_dim, hidden_size=self.hid_dim, num_layers=num_layer, dropout=dropout, bidirectional=True, batch_first=True)  


    def forward(self, src, src_len):
        #src: src sent len, batch size;  src_len: batch size

        embedded = self.dropout(self.embedding(src))
        #embedded, [src sent len, batch size, emb dim]
                    
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len, batch_first=True)
        
        hidden = self.rnn(packed_embedded)[1]
        #hidden: num_dir * num_layer, batch size, hid dim
        
        hidden_flat = hidden.permute(1, 0, 2).reshape(len(src_len), -1)
        out = self.out(hidden_flat)        
        
        return out