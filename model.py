import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers.modeling_bert import BertModel


class Encoder(nn.Module):
    def __init__(self, num_classes, num_support_per_class,
                 vocab_size, embed_size, hidden_size,
                 output_dim, weights):
        super(Encoder, self).__init__()
        self.num_support = num_classes * num_support_per_class
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        if weights is not None:
            self.embedding.weight = nn.Parameter(weights)
            self.embedding.requires_grad = False

        self.bilstm = nn.LSTM(embed_size, hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(2 * hidden_size, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def attention(self, x):
        weights = torch.tanh(self.fc1(x))
        weights = self.fc2(weights)  # (batch=k*c, seq_len, d_a)
        batch, seq_len, d_a = weights.shape
        weights = weights.transpose(1, 2)  # (batch=k*c, d_a, seq_len)
        weights = weights.contiguous().view(-1, seq_len)
        weights = F.softmax(weights, dim=1).view(batch, d_a, seq_len)
        sentence_embeddings = torch.bmm(weights, x)  # (batch=k*c, d_a, 2*hidden)
        # sentence_embeddings = weights @ x  # batch_size*r*lstm_hid_dim @表示矩阵-向量乘法

        avg_sentence_embeddings = torch.mean(sentence_embeddings, dim=1)  # (batch, 2*hidden)
        return avg_sentence_embeddings

    def forward(self, x, hidden=None):
        batch_size, _ = x.shape
        if hidden is None:
            h = x.data.new(2, batch_size, self.hidden_size).fill_(0).float()
            c = x.data.new(2, batch_size, self.hidden_size).fill_(0).float()
        else:
            h, c = hidden
        x = self.embedding(x)
        outputs, _ = self.bilstm(x, (h, c))  # (batch=k*c,seq_len,2*hidden)
        outputs = self.attention(outputs)  # (batch=k*c, 2*hidden)
        # (c*s, 2*hidden_size), (c*q, 2*hidden_size)
        support, query = outputs[0: self.num_support], outputs[self.num_support:]
        return support, query

class BertEncoder(BertModel):
    def __init__(self,config,num_classes, num_support_per_class,output_dim):
        super(BertEncoder,self).__init__(config)
        self.num_classes = num_classes
        self.num_support_per_class = num_support_per_class
        self.output_dim = output_dim
        self.num_support = num_classes * num_support_per_class
        # self.fc1 = nn.Linear(self.config.hidden_size, output_dim)
        # self.fc2 = nn.Linear(output_dim, output_dim)
        # torch.nn.init.xavier_normal_(self.fc1.weight)
        # torch.nn.init.xavier_normal_(self.fc2.weight)
        self.init_weights()
    
    def attention(self, x):
        weights = torch.tanh(self.fc1(x))
        weights = self.fc2(weights)  # (batch=k*c, seq_len, d_a)
        batch, seq_len, d_a = weights.shape
        weights = weights.transpose(1, 2)  # (batch=k*c, d_a, seq_len)
        weights = weights.contiguous().view(-1, seq_len)
        weights = F.softmax(weights, dim=1).view(batch, d_a, seq_len)
        sentence_embeddings = torch.bmm(weights, x)  # (batch=k*c, d_a, hidden)
        # sentence_embeddings = weights @ x  # batch_size*r*lstm_hid_dim @表示矩阵-向量乘法
        avg_sentence_embeddings = torch.mean(sentence_embeddings, dim=1)  # (batch, hidden)
        return avg_sentence_embeddings


    def forward(self,x):
        sequence_output,pooled_output = super().forward(input_ids=x)
        # outputs = self.attention(sequence_output)  # (batch=k*c, hidden)
        outputs = pooled_output
        # (c*s, 2*hidden_size), (c*q, hidden_size)
        support, query = outputs[0: self.num_support], outputs[self.num_support:]
        return support, query


class Induction(nn.Module):
    def __init__(self, C, S, hidden_size, iterations):
        super(Induction, self).__init__()
        self.C = C
        self.S = S
        self.H = hidden_size
        self.iterations = iterations
        # self.W = torch.nn.Parameter(torch.randn(H, H))
        self.W = nn.Linear(self.H,self.H)
        torch.nn.init.xavier_normal_(self.W.weight)

    def forward(self, x):
        dim = x.shape[-1]
        x = x.reshape(self.C,self.S,dim)# (C,S,H)
        b_ij = torch.zeros(self.C, self.S).to(x)
        e_ij = self.W(x)# (C,S,H)
        e_ij = self.squash(e_ij)

        for _ in range(self.iterations):
            d_i = F.softmax(b_ij, dim=1).unsqueeze(-1) # (C,S,1)
            
            c_i = torch.sum(torch.mul(d_i, e_ij), dim=1)  # (C,H)
            # squash
            c_i = self.squash(c_i)
            # c_produce_e = torch.bmm(e_ij, c_i.unsqueeze(2))  # (C,S,1)
            c_produce_e =torch.matmul(e_ij,c_i.unsqueeze(-1)).squeeze(-1) # (C,S)
            b_ij = b_ij + c_produce_e

        return c_i

    def squash_(self,x):
            squared = torch.norm(x)**2
            coeff = squared / (1 + squared) / torch.sqrt(squared + 1e-9)
            x = coeff * x
            return x
            
    def squash(self,input_tensor, dim=-1, epsilon=1e-7):
            squared_norm = (input_tensor ** 2).sum(dim=dim, keepdim=True)
            safe_norm = torch.sqrt(squared_norm + epsilon)
            scale = squared_norm / (1 + squared_norm)
            unit_vector = input_tensor / safe_norm
            return scale * unit_vector

class Relation(nn.Module):
    def __init__(self, C, H, out_size):
        super(Relation, self).__init__()
        self.out_size = out_size
        self.M = torch.nn.Parameter(torch.randn(H, H, out_size))
        self.W = nn.Linear(C * out_size,C)
        torch.nn.init.xavier_normal_(self.W.weight)

    def forward(self, class_vector, query_encoder):  # (C,H) (Q,H)
        mid_pro = []
        for slice in range(self.out_size):
            slice_inter = torch.matmul(torch.matmul(class_vector, self.M[:, :, slice]), query_encoder.transpose(1, 0))  # (C,Q)
            mid_pro.append(slice_inter)
        mid_pro = torch.cat(mid_pro, dim=0)  # (C*out_size,Q)
        V = F.relu(mid_pro.transpose(0, 1))  # (Q,C*out_size)
        probs = torch.sigmoid(self.W(V))  # (Q,C)
        return probs


class FewShotInduction(nn.Module):
    def __init__(self, C, S, vocab_size,config,weights=None):
        """
        C: number class
        S: support set
        """
        super(FewShotInduction, self).__init__()
        embed_size=int(config['model']['embed_dim'])
        hidden_size=int(config['model']['hidden_dim'])
        d_a=int(config['model']['d_a'])
        iterations=int(config['model']['iterations'])
        outsize=int(config['model']['relation_dim'])
        pretrain_path = config['data']['pretrain_path']
        if config['data']['encoder'] == 'lstm':
            self.encoder = Encoder(C, S, vocab_size, embed_size, hidden_size, d_a, weights)
            hidden_size = 2*hidden_size
        else:
            self.encoder = BertEncoder.from_pretrained(pretrain_path,num_classes=C, num_support_per_class=S,output_dim=d_a)
            hidden_size = self.encoder.config.hidden_size

        self.induction = Induction(C, S, hidden_size, iterations)
        self.relation = Relation(C,  hidden_size, outsize)

    def forward(self, x):
        support_encoder, query_encoder = self.encoder(x)  # (k*c, 2*hidden_size)
        class_vector = self.induction(support_encoder)
        probs = self.relation(class_vector, query_encoder)
        return probs

