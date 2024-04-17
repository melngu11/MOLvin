# Import necessary libraries
import deepchem as dc
# import deepchem.models
from deepchem.models import GAN
from deepchem.feat.molecule_featurizers.molgan_featurizer import GraphMatrix
#from deepchem.models import BasicMolGANModel as MolGAN
from deepchem.models.molgan import BasicMolGANModel as MolGAN
from deepchem.data.datasets import NumpyDataset as DS
from deepchem.models.optimizers import ExponentialDecay

import tensorflow as tf
from tensorflow import one_hot

import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors
# from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw

import numpy as np
import pandas as pd
import os
import random
import matplotlib.pyplot as plt
from collections import OrderedDict



base_output_dir = './data/run_data'
# Set up reproducibility
def set_reproducibility(seed=0):
    '''Set the random seed for reproducibility.
    '''
    np.random.seed(seed)
    random.seed(seed)

# Load the dataset
def load_data():
    '''
    Load the Tox21 dataset from DeepChem.
    
    Returns:
    - train_dataset (deepchem.data.datasets.DiskDataset): Training dataset
    - valid_dataset (deepchem.data.datasets.DiskDataset): Validation dataset
    - test_dataset (deepchem.data.datasets.DiskDataset): Test dataset
    
    '''
    tox21_tasks, tox21_dataset, transformers = dc.molnet.load_tox21()
    train_dataset, valid_dataset, test_dataset = tox21_dataset
    return train_dataset, valid_dataset, test_dataset

# Print dataset summaries
def print_dataset_summary(train_dataset, valid_dataset, test_dataset):
    '''
    Print the number of samples in each dataset.
    '''
    print(f"Training dataset: {train_dataset.X.shape}")
    print(f"Validation dataset: {valid_dataset.X.shape}")
    print(f"Test dataset: {test_dataset.X.shape}")

# Molecular Descriptors Calculation
def calculate_descriptors(mols_train):
    '''
    Calculate molecular descriptors for the training set.
    
    Parameters:
    - mols_train (list): List of RDKit molecule objects
    
    Returns:
    - df_descriptors (pd.DataFrame): DataFrame containing the calculated descriptors
    
    '''
    mol_descriptors = {'MolWt': [], 'LogP': [], 'NumHDonors': [], 'NumHAcceptors': []}
    for mol in mols_train:
        mol_descriptors['MolWt'].append(Descriptors.MolWt(mol))
        mol_descriptors['LogP'].append(Descriptors.MolLogP(mol))
        mol_descriptors['NumHDonors'].append(Descriptors.NumHDonors(mol))
        mol_descriptors['NumHAcceptors'].append(Descriptors.NumHAcceptors(mol))
    return pd.DataFrame(mol_descriptors)

# Print Statistical Summary
def print_statistical_summary(df_descriptors):
    '''
    Print the statistical summary of the calculated molecular descriptors.
    
    Parameters:
    - df_descriptors (pd.DataFrame): DataFrame containing the calculated descriptors
    
    Returns:
    - None
    
    '''
    print(df_descriptors.describe())

# Visualization Functions
def plot_histograms(df_descriptors, descriptor_names):
    ''' 
    Plot histograms of the molecular descriptors.
    Parameters:
    - df_descriptors (pd.DataFrame): DataFrame containing the calculated descriptors
    - descriptor_names (list): List of descriptor names to plot
    
    Return:
    - None
    '''
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
    '''
    Plot the correlation matrix of the molecular descriptors.
    
    Parameters:
    - df_descriptors (pd.DataFrame): DataFrame containing the calculated descriptors
    
    Returns:
    - None
    
    '''
    correlation_matrix = df_descriptors.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix of Molecular Descriptors')
    filename = os.path.join(base_output_dir, 'correlation_matrix.png')
    plt.savefig(filename)  # Save the figure to a file
    plt.close()

def plot_scatter_plots(df_descriptors):
    '''
    Plot pairwise scatter plots of the molecular descriptors.
    
    Parameters:
    - df_descriptors (pd.DataFrame): DataFrame containing the calculated descriptors
    
    Returns:
    - None
        
    '''
    sns.pairplot(df_descriptors)
    plt.suptitle('Pairwise Scatter Plots of Descriptors', y=1.02)
    filename = os.path.join(base_output_dir, 'scatter_plots.png')
    plt.savefig(filename)  # Save the figure to a file
    plt.close()

def plot_box_plots(df_descriptors, descriptor_names):
    '''
    Plot box plots of the molecular descriptors by toxicity category.
    
    Parameters:
    - df_descriptors (pd.DataFrame): DataFrame containing the calculated descriptors
    - descriptor_names (list): List of descriptor names to plot
    
    Returns:
    - None
    
    
    '''
    for descriptor in descriptor_names:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x='Toxicity', y=descriptor, data=df_descriptors)
        plt.title(f'{descriptor} by Toxicity Category')
        filename = os.path.join(base_output_dir, f'{descriptor}_box_plot.png')
        plt.savefig(filename)  # Save the figure to a file
        plt.close()

# Combining descriptors with toxicity labels
def combine_descriptors_labels(train_dataset, df_descriptors):
    
    '''
    Combine the calculated molecular descriptors with the toxicity labels.
    
    Parameters:
    - train_dataset (deepchem.data.datasets.DiskDataset): Training dataset
    - df_descriptors (pd.DataFrame): DataFrame containing the calculated descriptors
    
    Returns:
    - df_descriptors (pd.DataFrame): Updated DataFrame with the toxicity labels
    
    '''
    toxicity_labels = np.where(train_dataset.y.flatten() > 0, 'Toxic', 'Non-Toxic')
    df_descriptors['Toxicity'] = toxicity_labels[:len(df_descriptors)]
    return df_descriptors

def dataframe_setup(dataset):
    '''
    Create a DataFrame from a DeepChem DiskDataset object.
    
    Parameters:
    - dataset (deepchem.data.datasets.DiskDataset): DeepChem DiskDataset object
    
    Returns:
    - df (pd.DataFrame): DataFrame containing the SMILES strings
    
    '''
    
    # Ensure that 'dataset' is a DeepChem DiskDataset object
    if isinstance(dataset, dc.data.DiskDataset):
        # Create a DataFrame using the 'ids' attribute of the DiskDataset
        df = pd.DataFrame(data = {'smiles': dataset.ids})
        return df
    else:
        raise ValueError("Provided dataset is not a DeepChem DiskDataset object.")
 
def create_feature_and_filter(data , num_atoms, atom_labels):
    """
    Creates a MolGAN featurizer and filters SMILES based on the number of atoms.

    Parameters:
    - data (pd.DataFrame): DataFrame containing the SMILES strings under a 'smiles' column.
    - num_atoms (int): Maximum number of atoms allowed per molecule.
    - atom_labels (list of int): List of atomic numbers to include in the featurizer.

    Returns:
    - valid_features (list): List of valid GraphMatrix features for the MolGAN model.

    """
    # Initialize the MolGanFeaturizer
    feat = dc.feat.MolGanFeaturizer(max_atom_count=num_atoms, atom_labels=atom_labels)

    # Extract the SMILES from the DataFrame
    smiles = data['smiles'].values

    # Filter out the molecules with too many atoms
    filtered_smiles = [smile for smile in smiles if Chem.MolFromSmiles(smile).GetNumAtoms() < num_atoms]
    features = feat.featurize(filtered_smiles)
    indices = [i for i, feature in enumerate(features) if type(feature) is GraphMatrix]
    valid_features = [features[i] for i in indices]

    return valid_features, feat

def create_model(data, features, num_atoms):
    gan = MolGAN(learning_rate=ExponentialDecay(0.001, 0.9, 5000), vertices=num_atoms)
    dataset = dc.data.NumpyDataset([x.adjacency_matrix for x in features], [x.node_features for x in features])
    return gan, dataset

def iterbatches(epochs, gan, dataset):
    for i in range(epochs):
        for batch in dataset.iterbatches(batch_size=gan.batch_size, pad_batches=True):
            adjacency_tensor = one_hot(batch[0], gan.edges)
            node_tensor = one_hot(batch[1], gan.nodes)
            yield {gan.data_inputs[0]: adjacency_tensor, gan.data_inputs[1]: node_tensor}
def train_model(gan, dataset, epochs):
    gan.fit_gan(iterbatches(epochs, gan, dataset), generator_steps=0.2, checkpoint_interval=5000)
    generate_molecules = gan.predict_gan_generator(1000) 
    
    return generate_molecules
            






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
    
    filtered_valid_smiles, feat = create_feature_and_filter(df, max_atoms, [0, 5, 6, 7, 8, 9, 11, 12, 13, 14])  #15, 16, 17, 19, 20, 24, 29, 35, 53, 80])
    print(f"Number of valid SMILES strings: {len(filtered_valid_smiles)}")
    
    molGAN_model = create_model(df, filtered_valid_smiles, max_atoms)
    
    generated_mols = train_model(molGAN_model, filtered_valid_smiles, iterbatches(25))
    nmols = feat.defeaturize(generated_mols)
    print(f"{len(nmols)} unique valid molecules generated")
    # Remove invalid molecules from the list
    nmols = list(filter(lambda x: x is not None, nmols))    
    print ("{} valid molecules".format(len(nmols)))
    
    nmols_smiles = [Chem.MolToSmiles(m) for m in nmols]
    nmols_smiles_unique = list(OrderedDict.fromkeys(nmols_smiles))
    nmols_viz = [Chem.MolFromSmiles(x) for x in nmols_smiles_unique]
    print ("{} unique valid molecules".format(len(nmols_viz)))
    
    # Draw the molecules
    img = Draw.MolsToGridImage(nmols_viz[0:100], molsPerRow=5, subImgSize=(250, 250), maxMols=100, legends=None, returnPNG=False)
    img.show()
    
    
    
    
    
    

if __name__ == "__main__":
    main()
