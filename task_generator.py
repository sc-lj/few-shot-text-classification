# code is based on https://github.com/katerakelly/pytorch-maml
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, Dataset
import random
import os
import numpy as np
from torch.utils.data.sampler import Sampler
import copy
import Constants
from Util import count_doc, counter2dict, load_weights, sentence2indices
from transformers.tokenization_bert import BertTokenizer

# omniglot_character_folders
def get_data(path = "data/midea.txt"):
    doc = open(path, "r", encoding="utf-8").read().splitlines()
    random.seed(1)
    random.shuffle(doc)

    counter = count_doc(doc)
    word2index, index2word = counter2dict(counter=counter, min_freq=2)
    print(word2index, index2word)
    dict_data = {}
    for line in doc:
        words = line.split("\t")
        if len(words) != 2:
            print(line)
            continue
        y, x = words[0], words[1]
        if y in dict_data:
            dict_data[y].append(x)
        else:
            dict_data[y] = [x]
    keys = list(dict_data.keys())
    labels = {key: i for i, key in enumerate(keys)}
    # labels = {}
    print()
    for k,v in dict_data.items():
        print('类别',k,'样本数量',len(v), "平均长度", sum([ len(line) for line in v ])/len(v) )
    print("标签类别数量", len(labels))
    # test_classes = random.sample(keys, int(len(keys) * 0.2))
    dev_data = {k:v for k,v in dict_data.items() if 10<=len(v)<25}
    train_data = {k:v for k,v in dict_data.items() if len(v)>=25}
    test_data = {k:v for k,v in dict_data.items() if len(v)<10}

    return train_data, dev_data, test_data, word2index, labels


class ClassifyTask(object):
    # This class is for task generation for both meta training and meta testing.
    # For meta training, we use all 20 samples without valid set (empty here).
    # For meta testing, we use 1 or 5 shot samples for training, while using the same number of samples for validation.
    # If set num_samples = 20 and chracter_folders = metatrain_character_folders, we generate tasks for meta training
    # If set num_samples = 1 or 5 and chracter_folders = metatest_chracter_folders, we generate tasks for meta testing
    def __init__(self, dict_data, num_classes, train_num, test_num):
        self.character_folders = list(dict_data.keys())
        self.num_classes = int(num_classes)
        self.train_num = train_num
        self.test_num = test_num

        class_folders = random.sample(self.character_folders, self.num_classes)
        labels = np.array(range(len(class_folders)))
        labels = dict(zip(class_folders, labels))
        self.train_labels =[]
        self.test_labels =[]

        self.train_roots = []
        self.test_roots = []
        self.class_number = {"train":[],"test":[]}
        for c in class_folders:
            examples = dict_data[c]
            random.shuffle(examples)
            train = examples[:self.train_num]
            self.train_roots += train
            test = examples[self.train_num:self.train_num+self.test_num ]
            self.test_roots += test
            self.train_labels += [labels[c]] * len(train)
            self.test_labels += [labels[c]] * len(test)
            self.class_number['train'].append(len(train))
            self.class_number['test'].append(len(test))



class FewShotDataset(Dataset):

    def __init__(self, task, word2index,config):
        # self.transform = transform # Torch operations on the input image
        # self.target_transform = target_transform
        self.task = task
        self.config = config
        # self.split = split
        self.word2index = word2index
        self.max_len = int(config["data"]["window"])
        # self.image_roots = self.task.train_roots if self.split == 'train' else self.task.test_roots
        # self.labels = self.task.train_labels if self.split == 'train' else self.task.test_labels
        self.image_roots = self.task.train_roots+self.task.test_roots
        self.labels = self.task.train_labels+self.task.test_labels
        self.tokenizer = BertTokenizer.from_pretrained(config['data']['pretrain_path'])

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")


class Omniglot(FewShotDataset):

    def __init__(self, *args, **kwargs):
        super(Omniglot, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        line = self.image_roots[idx]
        if self.config['data']['encoder'] == 'lstm':
            image = sentence2indices(line, self.word2index, self.max_len, Constants.PAD)
        else:
            image = self.tokenizer.encode(line,max_length=self.max_len,pad_to_max_length=True)
        label = self.labels[idx]
        return torch.tensor(image), label


class ClassBalancedSampler(Sampler):
    ''' Samples 'num_inst' examples each from 'num_cl' pools
        of examples of size 'num_per_class' '''

    def __init__(self, num_per_class, num_cl, num_inst,class_number, shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle
        self.class_number = class_number

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batch = [[i + j * self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
            # batch = []
            # for j in range(self.num_cl):
            #     single = []
            #     number = list(range(self.class_number[j]))
            #     random.shuffle(number)
            #     while len(number)<self.num_per_class:
            #         a  = copy.deepcopy(number)
            #         random.shuffle(a)
            #         number.extend(a)
            #     random_number = number[:self.num_per_class]
            #     for i in random_number:
            #         single.append(i + sum(self.class_number[:j]))
            #     batch.append(single)
        else:
            batch = [[i + j * self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in  range(self.num_cl)]
            # batch = [[i + sum(self.class_number[:j])  for i in range(self.class_number[j])[:self.num_per_class]] for j in  range(self.num_cl)]
            # batch = []
            # for j in range(self.num_cl):
            #     single = []
            #     number = list(range(self.class_number[j]))
            #     while len(number)<self.num_per_class:
            #         number.extend(number)
            #     random_number = number[:self.num_per_class]
            #     for i in random_number:
            #         single.append(i + sum(self.class_number[:j]))
            #     batch.append(single)

        batch = [item for sublist in batch for item in sublist ]

        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1


def get_data_loader(task,word2index,config, shuffle=True):
    # NOTE: batch size here is # instances PER CLASS  split, word2index, max_len
    dataset = Omniglot(task, word2index=word2index,config=config)
    num_per_class=int(config["model"]["support"])+int(config["model"]["query"])
    per_class_number = task.class_number['train']
    # sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.train_num,class_number=per_class_number, shuffle=shuffle)
    loader = DataLoader(dataset, batch_size=num_per_class * task.num_classes,
    #  sampler=sampler,
    # collate_fn=lambda x:collate_fn(x,support=int(config['model']["support"]),class_number = int(config['model']["class"]))
    )

    return loader


def collate_fn(batchs,**kwargs):
    support = kwargs.get('support')
    class_number = kwargs.get('class_number')

    datas = []
    targets = []
    for data,target  in batchs:
        datas.append(data)
        targets.append(target)
    
    datas = torch.stack(datas)
    targets = torch.tensor(targets,dtype=torch.long)
    seq_len = datas.shape[-1]
    datas = datas.reshape(class_number,-1,seq_len).transpose(1,0)
    targets = targets.reshape(class_number,-1).transpose(1,0)
    support_data = datas[:support,:,:].transpose(1,0).reshape(-1,seq_len)
    support_target = targets[:support,:].transpose(1,0).reshape(-1)

    query_data = datas[support:,:,:].transpose(1,0).reshape(-1,seq_len)
    query_target = targets[support:,:].transpose(1,0).reshape(-1)
    data = torch.cat([support_data,query_data])
    target = torch.cat([support_target,query_target])
    return data,target




