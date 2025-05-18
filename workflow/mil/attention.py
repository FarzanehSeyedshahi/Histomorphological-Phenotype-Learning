
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

from torch import nn
import torch
import torch.nn.functional as F
import h5py
from tqdm import tqdm
import pandas as pd
import numpy as np
from skimage.transform import resize

import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from ignite.handlers import EarlyStopping
from torch.utils.data import WeightedRandomSampler
import warnings
warnings.filterwarnings("ignore")

# Images
main_path = '/mnt/cephfs/sharedscratch/users/fshahi/Projects/Histomorphological-Phenotype-Learning'
mil_path = '{}/workflow/mil'.format(main_path)
import sys
import os
sys.path.append(main_path)
from data_manipulation.data import Data
from models.clustering.correlations import *
from models.clustering.data_processing import *


# Attention MIL
def get_scores_simple(labels, y_prob):
    scores = dict()
    imb_thr = 0.3
    labels = np.array(labels)
    y_prob = np.array(y_prob)
    scores['ROC AUC'] = roc_auc_score(y_true=list(labels), y_score=list(y_prob))
    scores['Avg. Precision'] = average_precision_score(y_true=list(labels), y_score=list(y_prob))
    tn, fp, fn, tp = confusion_matrix(labels, (y_prob>imb_thr)*1.0).ravel()
    scores['Sensitivity'] = tp / (tp+fn)
    scores['Specificity'] = tn / (tn+fp)
    scores['Accuracy'] = accuracy_score(labels, (y_prob>imb_thr)*1.0)
    scores['F1'] = (2*tp) / (2*tp + fp + fn)
    scores['Balanced Accuracy'] = (scores['Sensitivity'] + scores['Specificity']) / 2
    return scores

class CsvBags(torch.utils.data.Dataset):
    def __init__(self, csv):
        self.csv = csv
        self.bags, self.labels = self._create_bag()

    def __len__(self):
        return len(self.bags)

    def _create_bag(self):
        bag_list = []
        label_list = []
        for bag in self.csv['slides'].unique():

            bag_df = self.csv[self.csv['slides']==bag]
            bag_list.append(torch.from_numpy(bag_df.drop(['slides', 'Meso_type'], axis=1).values).unsqueeze(1))
            # TODO: just get one label for the bag
            label_list.append(torch.from_numpy(bag_df['Meso_type'].values))
        return bag_list, label_list

    def __getitem__(self, idx):
        return self.bags[idx], self.labels[idx]
    
    def get_slide_bag(self, slide):
        bag_df = self.csv[self.csv['slides']==slide]
        bag = torch.from_numpy(bag_df.drop(['slides', 'Meso_type'], axis=1).values).unsqueeze(1)
        label = torch.from_numpy(bag_df['Meso_type'].values)
        return bag, label
    
def make_weighted_loder(data):
    from torch.utils.data import WeightedRandomSampler
    all_labels = torch.tensor([t[0].item() for t in CsvBags(data).labels], dtype=torch.long)
    class_counts = torch.bincount(all_labels)
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[all_labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    loader = torch.utils.data.DataLoader(CsvBags(data), batch_size=1, sampler=sampler)
    return loader



class GatedAttention(nn.Module):
    def __init__(self, input_dim, att_dim):
        super(GatedAttention, self).__init__()
        self.M = input_dim
        self.L = att_dim
        self.ATTENTION_BRANCHES = 1

        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh(),
            nn.Dropout(0.25)
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix U
            nn.Sigmoid(),
            nn.Dropout(0.25)
        )

        self.attention_w = nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)

        self.classifier = nn.Sequential(
            nn.Linear(self.M*self.ATTENTION_BRANCHES, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2),
            nn.Sigmoid()
        )

    def forward(self, x, return_features=False):
        A_V = self.attention_V(x)  # KxL
        A_U = self.attention_U(x)  # KxL
        A = self.attention_w(A_V * A_U) # element wise multiplication # KxATTENTION_BRANCHES
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K
        logits = A  # ATTENTION_BRANCHESxK

        Z = torch.mm(A, x)  # ATTENTION_BRANCHESxM

        Y_prob = self.classifier(Z)
        # Y_hat = torch.ge(Y_prob, 0.4).float()
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1]

        return logits, Y_prob, Y_hat, A, Z


class Attn_Net(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

class CLAM_SB(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = 0., k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024, hidden_dim_1 = 512, hidden_dim_2 = 256):
        super().__init__()
        self.size_dict = {"small": [embed_dim, hidden_dim_1, hidden_dim_2], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()
    
    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier): 
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets
    
    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):

        A, h = self.attention_net(h)  # NxK        
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)
                
        M = torch.mm(A, h) 
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict

class NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data = np.load(data_path, allow_pickle=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        instance = torch.from_numpy(self.data[idx][0])
        label = torch.tensor(self.data[idx][1], dtype=torch.float32)
        slide = self.data[idx][2]

        return instance, label, slide
    


if __name__ == '__main__':
    for fold in range(5):
        print('Loading data...')

        loaders = []
        for data_path in ['{}/data/subtype_train_fold_{}.npy'.format(mil_path, fold), 
                          '{}/data/subtype_test_fold_{}.npy'.format(mil_path, fold),
                          '{}/data/subtype_additional_fold_{}.npy'.format(mil_path, fold)]:
            print('Loading:', data_path)
            # number of positive and negative samples
            temp = NumpyDataset(data_path)
            print(temp.__len__())
            counts=np.array([len(np.where(temp.data[:,1]==i)[0]) for i in np.unique(temp.data[:,1])])
            weight = 1. / counts
            samples_weight = torch.from_numpy(np.array([weight[int(t)] for t in temp.data[:,1]]))
            samples_weigth = samples_weight.double()
            sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
            loader = torch.utils.data.DataLoader(temp, batch_size=1, sampler=sampler)
            loaders.append(loader)

        train_loader, test_loader, csv_loader_additional = loaders



        # Training Settings
        input_dim = 128  # Dimension of each instance's feature vector
        hidden_dim_1 = 64  # Dimension of the hidden layer
        hidden_dim_2 = 32  # Dimension of the output layer
        model_name = 'CLAMGatedAttention'
        k_sample = 10  # Number of instances to sample from each bag
        model = CLAM_SB(gate=True, size_arg='small', dropout=0.25, k_sample=k_sample, n_classes=2, 
                        instance_loss_fn=nn.CrossEntropyLoss(weight=torch.tensor([0.2, 0.8])), subtyping=True,
                        embed_dim=input_dim, hidden_dim_1=hidden_dim_1, hidden_dim_2=hidden_dim_2)

        # model = AttentionMIL(input_dim, hidden_dim)
        # model = GatedAttention(input_dim, hidden_dim_1)
        # Training settings
        learning_rate = 1e-4
        num_epochs = 50
        c1 = 0.9
        c2 = 0.3
        alpha = .7
        gamma = 1.5
        pos_weight = torch.tensor([5])
        scores = pd.DataFrame()
        scores_add = pd.DataFrame()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Device:', device)
        model = model.to(device)
        pos_weight = pos_weight.to(device)
        mil = 'clam'

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.
            train_error = 0.

            predictions, probs = [], []
            labels = []
            for instances, label, _ in train_loader:
                instances, label = instances.to(device), label.to(device)
                num_tiles = instances.shape[1]
                if num_tiles < k_sample:
                    continue
                instances = instances.squeeze(0).float()
                label_onehot = F.one_hot(label.long(), num_classes=2).float()
                optimizer.zero_grad()
                logits, y_prob, y_hat, A_raw, results_dict = model(instances, label=label.long(), return_features=True, instance_eval=True)

                if mil == 'clam':
                    
                    BCE_loss = F.binary_cross_entropy_with_logits(y_prob, label_onehot, reduction='none', pos_weight=pos_weight)
                    pt = torch.exp(-BCE_loss) # prevents nans when probability 0
                    F_loss = alpha * (1-pt)**gamma * BCE_loss
                    F_loss = F_loss.mean()

                    # bag_loss = F.binary_cross_entropy(y_prob, label_onehot)
                    bag_loss = F_loss
                    instance_loss = results_dict['instance_loss']
                    loss = c1*bag_loss + c2*instance_loss
                else:
                    loss = F.binary_cross_entropy(y_prob, label_onehot)
                train_loss += loss.item()

                predictions.append(int(y_hat))
                # labels.append(int(label))
                # probs.append(y_prob.squeeze(0)[:, 1].squeeze(0).item())

                loss.backward()
                optimizer.step()
                
            
            train_loss /= len(train_loader)
            print('Train Set, Epoch: {}, Loss: {:.5f}'.format(epoch, train_loss))
            # get_scores_simple(labels, probs)

            # Test loop
            with torch.no_grad():
                model.eval()
                predictions, probs = [], []
                labels = []
                for instances, label, _ in test_loader:
                    instances, label = instances.to(device), label.to(device)
                    instances = instances.squeeze(0).float()
                    label_onehot = F.one_hot(label.long(), num_classes=2).float()
                    logits, y_prob, y_hat, A_raw, results_dict = model(instances)

                    predictions.append(int(y_hat))
                    labels.append(int(label))
                    probs.append(y_prob[:, 1].item())

                test_acc = accuracy_score(labels, predictions)*100
                print('Test Set, Accuracy: {:.4f}%'.format(test_acc))
                # get_scores_simple(labels, probs)
                print('-----------------------------------------------------------------------------')

                scr_dict = pd.DataFrame([get_scores_simple(labels, probs)])
                scores = pd.concat([scores, scr_dict], ignore_index=True)


            # evaluate on the additional dataset
            torch.cuda.empty_cache()
            with torch.no_grad():
                predictions, probs = [], []
                labels = []
                for instances, label, slide in csv_loader_additional:
                    instances, label = instances.to(device), label.to(device)
                    instances = instances.squeeze(0).float()
                    label_onehot = F.one_hot(label.long(), num_classes=2).float()
                    logits, y_prob, y_hat, A_raw, results_dict = model(instances)
                    predictions.append(int(y_hat))
                    labels.append(int(label))
                    probs.append(y_prob[:, 1].item())

                val_acc = accuracy_score(labels, predictions)*100
                print('Additional Set, Accuracy: {:.4f}%'.format(val_acc))
                val_simple_scores = get_scores_simple(labels, probs)
                print(val_simple_scores)
                print('-----------------------------------------------------------------------------')

                scr_dict_add = pd.DataFrame([get_scores_simple(labels, probs)])
                scores_add = pd.concat([scores_add, scr_dict_add], ignore_index=True)

            scheduler.step(val_simple_scores['ROC AUC'])


        folder_path = '{}/results/CLAMGatedAttention/{}'.format(mil_path, fold)
        if not os.path.exists(folder_path):os.makedirs(folder_path, exist_ok=True)
        model_path = '{}/model_{}_fold_{}.pth'.format(folder_path, model_name, fold)
        model.cpu()
        torch.save(model.state_dict(), model_path)



        print(scores)
        df_mean = scores.mean().apply(lambda x: np.round(x,2))
        df_std = scores.std().apply(lambda x: np.round(x,2))
        temp = (df_mean.astype(str) + ' ± ' + df_std.astype(str)).reset_index()
        temp = pd.concat([scores, temp])
        temp.to_csv('{}/scores_fold_{}.csv'.format(folder_path, fold), index=False)

        print(scores_add)
        df_mean_add = scores_add.mean().apply(lambda x: np.round(x,2))
        df_std_add = scores_add.std().apply(lambda x: np.round(x,2))
        temp_add = (df_mean_add.astype(str) + ' ± ' + df_std_add.astype(str)).reset_index()
        temp_add = pd.concat([scores_add, temp_add])
        temp_add.to_csv('{}/scores_add_fold_{}.csv'.format(folder_path, fold), index=False)
