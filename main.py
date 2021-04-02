import configparser
import pickle
import os
import torch
from torch import optim
import random
import numpy as np
from model import FewShotInduction
from criterion import Criterion
from tensorboardX import SummaryWriter
from task_generator import get_data,ClassifyTask, get_data_loader
from Util import load_weights
from utils import save_pkl

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

def train(episode):
    model.train()
    task = ClassifyTask(train_data, config["model"]["class"], int(config["model"]["support"]),int(config["model"]["query"]))
    sample_dataloader = get_data_loader(task,word2index, config, num_per_class=int(config["model"]["support"])+int(config["model"]["query"]),shuffle=False)
    data, target = sample_dataloader.__iter__().next()
    data = data.to(device)
    target = target.to(device)
    optimizer.zero_grad()
    model.zero_grad()
    predict = model(data)
    loss, acc = criterion(predict, target)
    loss.backward()
    optimizer.step()

    writer.add_scalar('train_loss', loss.item(), episode)
    writer.add_scalar('train_acc', acc, episode)
    if episode % log_interval == 0:
        print('Train Episode: {} Loss: {} Acc: {}'.format(episode, loss.item(), acc))


def dev(episode):
    model.eval()
    correct = 0.
    count = 0.
    task = ClassifyTask(dev_data, config["model"]["class"], int(config["model"]["support"]), int(config["model"]["query"]))
    dev_loader = get_data_loader(task,word2index, config, num_per_class=int(config["model"]["support"])+int(config["model"]["query"]),shuffle=False)
    for data, target in dev_loader:
        with torch.no_grad():
            data = data.to(device)
            target = target.to(device)
            predict = model(data)
            _, acc = criterion(predict, target)
            amount = len(target) - support * 2
            correct += acc * amount
            count += amount
    acc = correct / count
    writer.add_scalar('dev_acc', acc, episode)
    print('Dev Episode: {} Acc: {}'.format(episode, acc))
    return acc


def test():
    model.eval()
    correct = 0.
    count = 0.
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        predict = model(data)
        _, acc = criterion(predict, target)
        amount = len(target) - support * 2
        correct += acc * amount
        count += amount
    acc = correct / count
    writer.add_scalar('test_acc', acc)
    print('Test Acc: {}'.format(acc))
    return acc


def main():
    best_episode, best_acc = 0, 0.
    episodes = int(config['model']['episodes'])
    early_stop = int(config['model']['early_stop']) * dev_interval
    for episode in range(1, episodes + 1):
        train(episode)
        if episode % dev_interval == 0:
            acc = dev(episode)
            if acc > best_acc:
                print('Better acc! Saving model!')
                torch.save(model.state_dict(), config['model']['model_path'])
                best_episode, best_acc = episode, acc
            # if episode - best_episode >= early_stop:
            #     print('Early stop at episode', episode)
            #     break

    print('Reload the best model on episode', best_episode, 'with best acc', best_acc.item())
    ckpt = torch.load(config['model']['model_path'])
    model.load_state_dict(ckpt)
    # test()


if __name__ == "__main__":
    # config
    config = configparser.ConfigParser()
    config.read("config.ini")

    # seed
    seed = int(config['model']['seed'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # log_interval
    log_interval = int(config['model']['log_interval'])
    dev_interval = int(config['model']['dev_interval'])

    # data loaders
    train_data, dev_data, test_loader, word2index, labels = get_data()
    
    # word2vec weights
    # weights = pickle.load(open(os.path.join(config['data']['path'], config['data']['weights']), 'rb'))    
    weights = load_weights(word2index,"data/char_vector.txt")
    
    save_pkl("./word2index_weight.pkl",(word2index,weights))
    
    # model & optimizer & criterion
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    support = int(config['model']['support'])
    model = FewShotInduction(C=int(config['model']['class']),
                             S=support,
                             vocab_size=len(word2index),
                             embed_size=int(config['model']['embed_dim']),
                             hidden_size=int(config['model']['hidden_dim']),
                             d_a=int(config['model']['d_a']),
                             iterations=int(config['model']['iterations']),
                             outsize=int(config['model']['relation_dim']),
                             weights=weights).to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(config['model']['lr']))
    criterion = Criterion(way=int(config['model']['class']), shot=int(config['model']['support']))
    

    # writer
    os.makedirs(config['model']['log_path'], exist_ok=True)
    writer = SummaryWriter(config['model']['log_path'])
    main()
    writer.close()
