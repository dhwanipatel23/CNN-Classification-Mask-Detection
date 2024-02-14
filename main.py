#!/usr/bin/env python
# coding: utf-8

# In[4]:


# importing necessary libraries

import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
import torch.nn.init as init
from torch import long, tensor
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torch.nn as nn
from torch import Tensor
from torch.nn import (Conv2d, CrossEntropyLoss, Linear, MaxPool2d, ReLU,Sequential, functional)
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from typing import Dict, List, Union
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from numpy import vstack
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import itertools
import matplotlib.pyplot as plt

# defining class to transform images to tensor and normalise them 

class MaskDetectionDataset(Dataset):
    def __init__(self, dataFrame):
        self.dataFrame = dataFrame
        self.transformations = Compose([Resize((32, 32)),ToTensor(),Normalize((0.5667, 0.5198, 0.4955),(0.3082, 0.2988, 0.3053))])

    def __getitem__(self, key):
        if isinstance(key, slice):
            raise NotImplementedError('Slicing is supported')

        row = self.dataFrame.iloc[key]
        image = Image.open(row['image']).convert('RGB')
        return {
            'image': self.transformations(image),
            'mask': tensor([row['mask']], dtype=long),
            'path': row['image']
        }

    def __len__(self):
        return len(self.dataFrame.index)

    # defining the CNN model with layers

class FaceMaskDetectorCNN(nn.Module):
    def __init__(self):
        super(FaceMaskDetectorCNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(8 * 8 * 64, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 4)
        )
    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

def ques():
        # Loading dataset and assigning seperate variable to each type

        datasetPath = Path('data/dataset')
        noMaskPath = datasetPath/'no'
        clothMaskPath = datasetPath/'cloth'
        FFP2MaskPath = datasetPath/'FFP2'
        surgicalMaskPath = datasetPath/'surgical'
        maskDF = pd.DataFrame()

        # using the tqdm to check progress in loading of datasets
        for imgPath in tqdm(list(noMaskPath.iterdir()), desc='no'):
            maskDF = maskDF.append({'image': str(imgPath),'mask': 0}, ignore_index=True)

        for imgPath in tqdm(list(clothMaskPath.iterdir()), desc='cloth'):
            maskDF = maskDF.append({'image': str(imgPath),'mask': 1}, ignore_index=True)

        for imgPath in tqdm(list(FFP2MaskPath.iterdir()), desc='FFP2'):
            maskDF = maskDF.append({'image': str(imgPath),'mask': 2}, ignore_index=True)

        for imgPath in tqdm(list(surgicalMaskPath.iterdir()), desc='surgical'):
            maskDF = maskDF.append({'image': str(imgPath),'mask': 3}, ignore_index=True)



        face_mask_detector_cnn = FaceMaskDetectorCNN()
        np.save("model.npy", face_mask_detector_cnn)

        # establishing and plotting confusion matrix

        def plot_cm(cm, classes, normalize=False, title='Visualization of the confusion matrix', cmap=plt.cm.Reds):
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)

            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            np.save("confusion matrix", cm)

            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

            plt.tight_layout()
            plt.ylabel('Actual True label')
            plt.xlabel('Predicted label')
            plt.savefig('matrix.png')

        # implementing stratified kfold

        def prepare_data(mask_df_path) -> None:
                mask_df = pd.read_pickle(mask_df_path)
                skf = StratifiedKFold(n_splits=10, shuffle=True)
                train_folds = []
                validate_folds = []
                for train_index, validate_index in skf.split(mask_df, mask_df['mask']):
                    train_folds.append(MaskDetectionDataset(mask_df.iloc[train_index]))
                    validate_folds.append(MaskDetectionDataset(mask_df.iloc[validate_index]))

                return [train_folds, validate_folds, CrossEntropyLoss()]

        def train_dataloader(train_df) -> DataLoader:
            return DataLoader(train_df, batch_size=32, shuffle=True, num_workers=0)

        def val_dataloader(validate_df) -> DataLoader:
            return DataLoader(validate_df, batch_size=32, num_workers=0)   

        train_dfs, validate_dfs, cross_entropy_loss = prepare_data("dataset/dataset.pickle")

        # training steps

        epochs = 10
        learning_rate = 0.001 
        retrain = False

        import warnings
        warnings.filterwarnings('ignore')

        def train_model(train_fold):
            acc_list = []
            loss_list = []
            optimizer = Adam(face_mask_detector_cnn.parameters(), lr=learning_rate)
            for epoch in range(epochs):
                total=0
                correct=0
                loss_train = 0.0
                for i, data in enumerate(train_dataloader(train_fold), 0):
                    inputs, labels = data['image'], data['mask']
                    labels = labels.flatten()
                    outputs = face_mask_detector_cnn(inputs)
                    loss = cross_entropy_loss(outputs, labels)
                    loss_list.append(loss.item())
                    optimizer.zero_grad() 
                    loss.backward()
                    optimizer.step()

                    #training accuracy
                    total += labels.size(0)
                    _, predicted = torch.max(outputs.data, 1) 
                    correct += (predicted == labels).sum().item() 
                    loss_train += loss

        # returning values for confusion matrix

        def evaluate_model(validate_fold):
            predictions, actuals = torch.tensor([]), torch.tensor([])
            for i, data in enumerate(val_dataloader(validate_fold)):
                inputs, targets = data['image'], data['mask']
                targets = targets.flatten()
                output = face_mask_detector_cnn(inputs)
                output = torch.argmax(output,axis=1)
                predictions = torch.cat((predictions, output.flatten()), dim=0)
                actuals = torch.cat((actuals, targets), dim=0)

            return (confusion_matrix(actuals.numpy(), predictions.numpy()),accuracy_score(actuals, predictions),*precision_recall_fscore_support(actuals.numpy(), predictions.numpy(),average='macro'))

        # running folds and genereating metrics

        fold_results = []
        fold_confusion_matrix = np.zeros((4,4))
        classes = ['without_mask', 'cloth_mask', 'ffp2_mask', 'surgical_mask']

        for fold_index in range(len(train_dfs)):
            train_model(train_dfs[fold_index])
            fold_result = evaluate_model(validate_dfs[fold_index])
            #conf_mat, acc, pre, recall, f-score
            fold_results.append(fold_result[1:-1])
            fold_confusion_matrix = np.add(fold_confusion_matrix,fold_result[0])
            if fold_index != len(train_dfs)-1:
                face_mask_detector_cnn = FaceMaskDetectorCNN()

        # saving metrics in dataframe    
        metrics_df = pd.DataFrame(fold_results, columns=['accuracy', 'precision', 'recall', 'f-score'])

        # calling the function to plot
        plot_cm(fold_confusion_matrix, classes)

if __name__ == "__main__":
    ques()


# In[ ]:




