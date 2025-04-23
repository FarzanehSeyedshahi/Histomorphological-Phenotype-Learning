
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
import warnings
warnings.filterwarnings("ignore")

# Images
main_path = '/mnt/cephfs/sharedscratch/users/fshahi/Projects/Histomorphological-Phenotype-Learning'
import sys
import os
sys.path.append(main_path)
from data_manipulation.data import Data
from models.clustering.correlations import *
from models.clustering.data_processing import *


# Attention MIL
def get_scores_simple(labels, y_prob):
    scores = dict()
    labels = np.array(labels)
    y_prob = np.array(y_prob)
    scores['ROC AUC'] = roc_auc_score(y_true=list(labels), y_score=list(y_prob))
    scores['Avg. Precision'] = average_precision_score(y_true=list(labels), y_score=list(y_prob))
    tn, fp, fn, tp = confusion_matrix(labels, (y_prob>0.5)*1.0).ravel()
    scores['Sensitivity'] = tp / (tp+fn)
    scores['Specificity'] = tn / (tn+fp)
    scores['Accuracy'] = accuracy_score(labels, (y_prob>0.5)*1.0)
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
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)
        x = x.squeeze(1)

        A_V = self.attention_V(x)  # KxL
        A_U = self.attention_U(x)  # KxL
        A = self.attention_w(A_V * A_U) # element wise multiplication # KxATTENTION_BRANCHES
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K

        Z = torch.mm(A, x)  # ATTENTION_BRANCHESxM

        Y_prob = self.classifier(Z)
        Y_hat = torch.ge(Y_prob, 0.4).float()

        return Y_prob, Y_hat, A, Z







dataset = 'Meso'
h5_complete_path   = '{}/results/BarlowTwins_3/{}/h224_w224_n3_zdim128/hdf5_{}_he_complete_filtered_metadata.h5'.format(main_path, dataset, dataset)
frame, dims, rest = representations_to_frame(h5_complete_path, meta_field='Meso_type', rep_key='z_latent')


additional_dataset = 'TCGA_MESO'
h5_additional_path = '{}/results/BarlowTwins_3/{}/h224_w224_n3_zdim128/hdf5_{}_he_complete_filtered_metadata.h5'.format(main_path, additional_dataset, additional_dataset)
frame_additional, dims_additional, rest_additional = representations_to_frame(h5_additional_path, meta_field='Meso_type', rep_key='z_latent')



# Data packing for MIL (Just one fold)
csv_data = frame.iloc[:,0:128]
csv_data['Meso_type'] = frame['Meso_type'].astype(float)
csv_data['slides'] = frame['slides']
ratio = 0.8
train_df_slides = np.random.choice(csv_data['slides'].unique(), size=int(len(csv_data['slides'].unique())*ratio), replace=False)
test_df_slides = list(set(csv_data['slides'].unique()) - set(train_df_slides))
train_df = csv_data[csv_data['slides'].isin(train_df_slides)]
test_df = csv_data[csv_data['slides'].isin(test_df_slides)]
print(f'SIZE: Train slides: {len(train_df_slides)} - Test slides: {len(test_df_slides)}', f'| Train samples: {len(train_df)} - Test samples: {len(test_df)}')


csv_data_additional = frame_additional.iloc[:,0:128]
csv_data_additional['Meso_type'] = frame_additional['type']
csv_data_additional['Meso_type'] = csv_data_additional['Meso_type'].replace({'Epithelioid': 0, 'Sarcomatoid': 1, 'Biphasic': 1}).astype(float)
csv_data_additional['slides'] = frame_additional['slides']




# takes time :( ->for Meso Data: 14 minutes
# train_loader = torch.utils.data.DataLoader(CsvBags(train_df), batch_size=1, shuffle=True)
# test_loader = torch.utils.data.DataLoader(CsvBags(test_df), batch_size=1, shuffle=True)
train_loader = make_weighted_loder(train_df)
test_loader = make_weighted_loder(test_df)
csv_loader_additional = make_weighted_loder(csv_data_additional)


# Training Settings
input_dim = 128  # Dimension of each instance's feature vector
hidden_dim = 64  # Dimension of the hidden layer
# model = AttentionMIL(input_dim, hidden_dim)
model = GatedAttention(input_dim, hidden_dim)
# Training settings
learning_rate = 1e-4
num_epochs = 30
scores = pd.DataFrame()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)



# Training loop
# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.
    train_error = 0.

    predictions, probs = [], []
    labels = []
    for instances, label in train_loader:
        num_tiles = instances.shape[1]
        instances = instances.float()
        # TODO: just get one label for the bag
        label = label.squeeze(0)[0].unsqueeze(0).float()
        optimizer.zero_grad()
        y_prob, y_hat, attention_score, weighted_instances = model(instances)

        loss = F.binary_cross_entropy(y_prob.squeeze(0), label)
        train_loss += loss.item()

        predictions.append(int(y_hat))
        labels.append(int(label))
        probs.append(y_prob.item())

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
        for instances, label in test_loader:
            instances = instances.float()
            # TODO: just get one label for the bag
            label = label.squeeze(0)[0].unsqueeze(0).float()
            y_prob, y_hat, attention_score, weighted_instances = model(instances)

            predictions.append(int(y_hat))
            labels.append(int(label))
            probs.append(y_prob.item())

        test_acc = accuracy_score(labels, predictions)*100
        print('Test Set, Accuracy: {:.4f}%'.format(test_acc))
        get_scores_simple(labels, probs)
        print('-----------------------------------------------------------------------------')

        scr_dict = pd.DataFrame([get_scores_simple(labels, probs)])
        scores = pd.concat([scores, scr_dict], ignore_index=True)


    # evaluate on the additional dataset
    with torch.no_grad():
        model.eval()
        predictions, probs = [], []
        labels = []
        for instances, label in csv_loader_additional:
            instances = instances.float()
            label = label.squeeze(0)[0].unsqueeze(0).float()
            y_prob, y_hat, attention_score, weighted_instances = model(instances)
            predictions.append(int(y_hat))
            labels.append(int(label))
            probs.append(y_prob.item())
        val_acc = accuracy_score(labels, predictions)*100
        print('Additional Set, Accuracy: {:.4f}%'.format(val_acc))
        get_scores_simple(labels, probs)
        print('-----------------------------------------------------------------------------')
        # scr_dict_additional = pd.DataFrame([get_scores_simple(labels, probs)])
        # display(scr_dict_additional)
        # scores_additional = pd.concat([scores, scr_dict_additional], ignore_index=True)

    scheduler.step(val_acc)

# Save the model
from copy import deepcopy
model_trained = deepcopy(model)
# save the model
folder_path = '/nfs/home/users/fshahi/Projects/Histomorphological-Phenotype-Learning/workflow/mil_results/'
os.makedirs(folder_path, exist_ok=True)
model_path = os.path.join(folder_path, 'GatedAttention_{}_{}.pth'.format(dataset, additional_dataset))
torch.save(model_trained.state_dict(), model_path)
# Save the scores
scores_path = os.path.join(folder_path, 'GatedAttention_{}_{}_scores.csv'.format(dataset, additional_dataset))
scores.to_csv(scores_path, index=False)



print(scores)
# df_mean = scores.mean().apply(lambda x: np.round(x,2))
# df_std = scores.std().apply(lambda x: np.round(x,2))
# temp = (df_mean.astype(str) + ' ± ' + df_std.astype(str)).reset_index()
