# Import necessary libraries
import deepchem as dc
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
import pandas as pd
import os
import random
import matplotlib.pyplot as plt

base_output_dir = './data/run_data'
# Set up reproducibility
def set_reproducibility(seed=0):
    np.random.seed(seed)
    random.seed(seed)

# Load the dataset
def load_data():
    tox21_tasks, tox21_dataset, transformers = dc.molnet.load_tox21()
    train_dataset, valid_dataset, test_dataset = tox21_dataset
    return train_dataset, valid_dataset, test_dataset

# Print dataset summaries
def print_dataset_summary(train_dataset, valid_dataset, test_dataset):
    print(f"Training dataset: {train_dataset.X.shape}")
    print(f"Validation dataset: {valid_dataset.X.shape}")
    print(f"Test dataset: {test_dataset.X.shape}")

# Molecular Descriptors Calculation
def calculate_descriptors(mols_train):
    mol_descriptors = {'MolWt': [], 'LogP': [], 'NumHDonors': [], 'NumHAcceptors': []}
    for mol in mols_train:
        mol_descriptors['MolWt'].append(Descriptors.MolWt(mol))
        mol_descriptors['LogP'].append(Descriptors.MolLogP(mol))
        mol_descriptors['NumHDonors'].append(Descriptors.NumHDonors(mol))
        mol_descriptors['NumHAcceptors'].append(Descriptors.NumHAcceptors(mol))
    return pd.DataFrame(mol_descriptors)

# Print Statistical Summary
def print_statistical_summary(df_descriptors):
    print(df_descriptors.describe())

# Visualization Functions
def plot_histograms(df_descriptors, descriptor_names):
    for descriptor in descriptor_names:
        plt.figure(figsize=(6, 4))
        sns.histplot(df_descriptors[descriptor], kde=True, color='blue')
        plt.title(f'Distribution of {descriptor}')
        plt.xlabel(descriptor)
        plt.ylabel('Frequency')
        filename = os.path.join(base_output_dir, f'{descriptor}_histogram.png')
        plt.savefig(filename)  # Save the figure to a file
        plt.close()

def plot_correlation_matrix(df_descriptors):
    correlation_matrix = df_descriptors.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix of Molecular Descriptors')
    filename = os.path.join(base_output_dir, 'correlation_matrix.png')
    plt.savefig(filename)  # Save the figure to a file
    plt.close()

def plot_scatter_plots(df_descriptors):
    sns.pairplot(df_descriptors)
    plt.suptitle('Pairwise Scatter Plots of Descriptors', y=1.02)
    filename = os.path.join(base_output_dir, 'scatter_plots.png')
    plt.savefig(filename)  # Save the figure to a file
    plt.close()

def plot_box_plots(df_descriptors, descriptor_names):
    for descriptor in descriptor_names:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x='Toxicity', y=descriptor, data=df_descriptors)
        plt.title(f'{descriptor} by Toxicity Category')
        filename = os.path.join(base_output_dir, f'{descriptor}_box_plot.png')
        plt.savefig(filename)  # Save the figure to a file
        plt.close()

# Combining descriptors with toxicity labels
def combine_descriptors_labels(train_dataset, df_descriptors):
    toxicity_labels = np.where(train_dataset.y.flatten() > 0, 'Toxic', 'Non-Toxic')
    df_descriptors['Toxicity'] = toxicity_labels[:len(df_descriptors)]
    return df_descriptors

def dataframe_setup(datasets):
    df = pd.DataFrame(datasets.ids, columns=['ID'])
    return df
 













def main():
    
    ## TODO: set as data prep and exploration function
    set_reproducibility()
    base_output_dir = './data/run_data'
    os.makedirs(base_output_dir, exist_ok=True)
    
    train_dataset, valid_dataset, test_dataset = load_data()
    print_dataset_summary(train_dataset, valid_dataset, test_dataset)
    smiles_train = [Chem.MolFromSmiles(smile) for smile in train_dataset.ids]
    mols_train = [mol for mol in smiles_train if mol is not None]  # Filter out None values
    # ----EXPORE ---- #
    # base_output_dir = './data/run_data'
    
    df_descriptors = calculate_descriptors(mols_train)
    print_statistical_summary(df_descriptors)
    plot_histograms(df_descriptors, ['MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors'])
    plot_correlation_matrix(df_descriptors)
    plot_scatter_plots(df_descriptors)
    df_descriptors = combine_descriptors_labels(train_dataset, df_descriptors)
    plot_box_plots(df_descriptors, ['MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors'])
    
    max_atoms = 12 # Maximum number of atoms in a molecule to consider. Increase for larger molecules
    df = dataframe_setup(train_dataset)
    
    print (f"Dataframe: {df.head()}")
    

if __name__ == "__main__":
    main()
