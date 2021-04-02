# coding:utf-8

from transformers.modeling_bert import BertModel
from transformers.tokenization_bert import BertTokenizer
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import Dataset,DataLoader
from tensorboardX import SummaryWriter
import torch
from torch.optim import Adam
import torch.nn.functional as f
import numpy as np
from sklearn import metrics
import configparser
import random

seed = 2021
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
#最低训练样本数量
MIN_TRAIN_NUMBER = 6

def read_data(files):
        with open(files,'r',encoding="utf-8") as f:
                lines = f.readlines()
        label_text = defaultdict(list)
        
        for line  in lines:
            l = line.strip().split("\t")
            if len(l)==2:
                label_text[l[0]].append(l[1])
            else:
                print(l)

        val_data = {}
        train_data = {}
        labels = set()
        for label,text in label_text.items():
            # random.shuffle(text)
            number = len(text)
            # train_number = min(MIN_TRAIN_NUMBER,int(number*0.5))
            train_number = MIN_TRAIN_NUMBER
            train_data[label] = text[:train_number]
            val_data[label] = text[train_number:]
            labels.add(label)
        labels = sorted(labels)
        labels = dict(zip(labels,range(len(labels))))
        return val_data,train_data,labels


class dataLoader(Dataset):
    def __init__(self,data,tokenizer:BertTokenizer,max_length,label_dict,):
        super(dataLoader,self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        self.label = []
        self.label_dict = label_dict
        for l,d in data.items():
            self.data.extend(d)
            self.label.extend([l]*len(d))

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        data = self.data[index]
        label = self.label_dict[self.label[index]]
        inputs_id = self.tokenizer.encode(data,max_length=self.max_length,pad_to_max_length=True)
        inputs_id = torch.tensor(inputs_id,dtype=torch.long)
        label = torch.tensor(label,dtype=torch.long)
        return inputs_id,label

class bertClassifyModel(BertModel):
    def __init__(self,config,labels):
        super(bertClassifyModel,self).__init__(config)
        self.linear = nn.Linear(self.config.hidden_size,labels)
        self.init_weights()

    def  forward(self,x):
        _,pooled_output = super().forward(input_ids=x)
        logit = self.linear(pooled_output)
        return logit

def get_loss(output,target):
        loss = f.cross_entropy(output,target)
        return loss

def acc_and_f1(preds, labels):
    acc = (preds.argmax(-1) == labels).mean()
    f1 = metrics.f1_score(y_true=labels, y_pred=preds.argmax(-1),average="macro")
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
        "recall": metrics.recall_score(labels,preds.argmax(-1),average="macro")
    }


def main(config):
    writer = SummaryWriter("./classify_log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 20
    val_data,train_data,label_dict = read_data("./data/midea_dev.txt")
    n_class = len(label_dict)

    pretrain_path = 'bert_chinese_base'
    tokenizer = BertTokenizer.from_pretrained(pretrain_path)
    
    train_dataset = dataLoader(train_data,tokenizer,int(config["data"]['window']),label_dict)
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

    val_dataset = dataLoader(val_data,tokenizer,int(config['data']['window']),label_dict)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=True)
    model = bertClassifyModel.from_pretrained(pretrain_path,labels=n_class)

    model.to(device)
    opt = Adam(model.parameters(),lr=float(config["model"]["lr"]))
    best_score = 0
    best_result = {}
    for epoch in range(100):
        for index,data in enumerate(train_loader):
            model.train()
            model.zero_grad()
            opt.zero_grad()
            input_id,target = data
            input_id = input_id.to(device)
            target = target.to(device)
            logit = model(input_id)
            loss = get_loss(logit,target)
            loss.backward()
            opt.step()
            writer.add_scalar("train_loss",loss,index)

            if index %100 == 0:
                eval_loss = 0.0
                nb_eval_steps = 0
                preds = None
                out_label_ids = None
                for eval_data  in val_loader:
                        model.eval()
                        input_id,target = eval_data
                        input_id = input_id.to(device)
                        target = target.to(device)
                        with torch.no_grad():
                            logits = model(input_id)
                            loss = get_loss(logits,target)
                            eval_loss += loss.mean().item()
                            nb_eval_steps += 1
                            if preds is None:
                                preds = logits.detach().cpu().numpy()
                                out_label_ids =target.detach().cpu().numpy()
                            else:
                                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                                out_label_ids = np.append(out_label_ids, target.detach().cpu().numpy(), axis=0)
                eval_loss = eval_loss / nb_eval_steps

                result = acc_and_f1(preds, out_label_ids)
                
                for k,v in result.items():
                    writer.add_scalar(k,v,index)
                    print("epoch:%s"%epoch,k,"==",v)
                if result['f1']>best_score:
                    torch.save(model.state_dict,"./best_bert.pt")
                    best_score = result['f1']
                    best_result = result
    
    print("best score:",best_score)
    print("best result:",best_result)


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config.ini")
    main(config)


