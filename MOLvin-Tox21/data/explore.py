import deepchem as dc 
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
import seaborn as sns
import numpy as np
import pandas as pd
import os
import random
##
from collections import OrderedDict
import deepchem.models
from deepchem.models import BasicMolGANModel as MolGAN
from deepchem.models.optimizers import ExponentialDecay
import tensorflow as tf
from tensorflow import one_hot
from rdkit.Chem.Draw import IPythonConsole

from deepchem.feat.molecule_featurizers.molgan_featurizer import GraphMatrix
##

# from sklearn.preprocessing import StandardScaler


# Reproducibility
np.random.seed(0), random.seed(0)



# Load the Tox21 dataset
tox21_tasks, tox21_dataset, transformers = dc.molnet.load_tox21()
train_dataset, valid_dataset, test_dataset = tox21_dataset

# Print out datasets
print(f"Training dataset: {train_dataset.X.shape}")
print(f"Validation dataset: {valid_dataset.X.shape}")
print(f"Test dataset: {test_dataset.X.shape}")

# Print out first 5 SMILE strings and labels from the training set
print("\nFirst 5 SMILE strings and labels from the training set:")
for i in range(5):
    print(f"SMILE string: {train_dataset.ids[i]}, Label: {train_dataset.y[i]}")

# Visualization of the label distribution in the training set
plt.figure(figsize=(10, 6))
sns.countplot(x=train_dataset.y.flatten())
plt.title('Distribution of Labels in the Tox21 Training Dataset')
plt.xlabel('Toxicity Label')
plt.ylabel('Count')
plt.show()

smiles_train = [Chem.MolFromSmiles(smile) for smile in train_dataset.ids]
mols_train = [mol for mol in smiles_train if mol is not None]  # Remove None values if any corrupt SMILES

# 1. Molecular Descriptors Calculation
mol_descriptors = {'MolWt': [], 'LogP': [], 'NumHDonors': [], 'NumHAcceptors': []}
for mol in mols_train:
    mol_descriptors['MolWt'].append(Descriptors.MolWt(mol))
    mol_descriptors['LogP'].append(Descriptors.MolLogP(mol))
    mol_descriptors['NumHDonors'].append(Descriptors.NumHDonors(mol))
    mol_descriptors['NumHAcceptors'].append(Descriptors.NumHAcceptors(mol))
    
df_descriptors = pd.DataFrame(mol_descriptors)

# 2. Statistical Summary
print(df_descriptors.describe())

# Histograms for each descriptor
descriptor_names = ['MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors']
for descriptor in descriptor_names:
    plt.figure(figsize=(6, 4))
    sns.histplot(df_descriptors[descriptor], kde=True, color='blue')
    plt.title(f'Distribution of {descriptor}')
    plt.xlabel(descriptor)
    plt.ylabel('Frequency')
    plt.show()

# Correlation matrix for the descriptors
correlation_matrix = df_descriptors.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Molecular Descriptors')
plt.show()


# Pairwise scatter plots
sns.pairplot(df_descriptors)
plt.suptitle('Pairwise Scatter Plots of Descriptors', y=1.02)
plt.show()


# Combining the descriptors with the toxicity labels for the training set
toxicity_labels = np.where(train_dataset.y.flatten() > 0, 'Toxic', 'Non-Toxic')
df_descriptors['Toxicity'] = toxicity_labels[:len(df_descriptors)]

# Box plots for each descriptor by toxicity category
for descriptor in descriptor_names:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='Toxicity', y=descriptor, data=df_descriptors)
    plt.title(f'{descriptor} by Toxicity Category')
    plt.show()
    
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(df_descriptors)
# X_valid_scaled = scaler.transform(valid_dataset_descriptors)
# X_test_scaled = scaler.transform(test_dataset_descriptors)

