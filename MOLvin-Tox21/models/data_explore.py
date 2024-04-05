# data_exploration.py

import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
import pandas as pd

# Function to calculate molecular descriptors
def calculate_descriptors(smiles_list):
    mols = [Chem.MolFromSmiles(smile) for smile in smiles_list]
    mols = [mol for mol in mols if mol is not None]
    mol_descriptors = {
        'MolWt': [Descriptors.MolWt(mol) for mol in mols],
        'LogP': [Descriptors.MolLogP(mol) for mol in mols],
        'NumHDonors': [Descriptors.NumHDonors(mol) for mol in mols],
        'NumHAcceptors': [Descriptors.NumHAcceptors(mol) for mol in mols],
    }
    return pd.DataFrame(mol_descriptors)

# Function to create histograms for descriptors
def plot_histograms(df_descriptors):
    descriptor_names = ['MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors']
    for descriptor in descriptor_names:
        plt.figure(figsize=(6, 4))
        sns.histplot(df_descriptors[descriptor], kde=True, color='blue')
        plt.title(f'Distribution of {descriptor}')
        plt.xlabel(descriptor)
        plt.ylabel('Frequency')
        plt.show()

# Function to plot correlation matrix
def plot_correlation_matrix(df_descriptors):
    correlation_matrix = df_descriptors.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix of Molecular Descriptors')
    plt.show()

# ... [Add additional visualization functions here] ...
