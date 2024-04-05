import deepchem as dc
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
import numpy as np
import pandas as pd
import os
import random
from collections import OrderedDict
from deepchem.models.optimizers import ExponentialDecay
from tensorflow import one_hot
from rdkit.Chem.Draw import IPythonConsole

# Reproducibility
np.random.seed(0), random.seed(0)

# Load the Tox21 dataset
tox21_tasks, tox21_dataset, transformers = dc.molnet.load_tox21()
train_dataset, valid_dataset, test_dataset = tox21_dataset

# ... [Your initial data exploration and visualization code] ...

# Integrate MolGAN setup and training here
# Set up the MolGAN model
from deepchem.models import BasicMolGANModel as MolGAN  # Import statement added here
from deepchem.feat.molecule_featurizers.molgan_featurizer import GraphMatrix

# Specify the number of atoms for the featurizer
num_atoms = 12  # You may adjust this based on the dataset or desired molecule size

# Convert SMILES strings to MolGAN-friendly format
smiles_list = [Chem.MolToSmiles(mol) for mol in mols_train]
featurizer = dc.feat.MolGanFeaturizer(max_atom_count=num_atoms, atom_labels=[0, 5, 6, 7, 8, 9, 11, 12, 13, 14])
features = featurizer.featurize(smiles_list)

# Filter out invalid features
valid_features = [feature for feature in features if feature is not None]
indices = [i for i, feature in enumerate(valid_features) if type(feature) is GraphMatrix]
valid_features = [valid_features[i] for i in indices]

# Prepare dataset for MolGAN
gan_data = dc.data.NumpyDataset([x.adjacency_matrix for x in valid_features],
                                [x.node_features for x in valid_features])

# Define the MolGAN model
gan = MolGAN(learning_rate=ExponentialDecay(0.001, 0.9, 5000), vertices=num_atoms)

# Train the MolGAN model
def iterbatches(epochs):
    for i in range(epochs):
        for batch in gan_data.iterbatches(batch_size=gan.batch_size, pad_batches=True):
            adjacency_tensor = one_hot(batch[0], gan.edges)
            node_tensor = one_hot(batch[1], gan.nodes)
            yield {gan.data_inputs[0]: adjacency_tensor, gan.data_inputs[1]:node_tensor}

# Fit the MolGAN model with the prepared batches
gan.fit_gan(iterbatches(25), generator_steps=0.2, checkpoint_interval=5000)

# Generate new molecules with the trained model
generated_data = gan.predict_gan_generator(1000)

# Defeaturize the generated data back into RDKit molecule objects
nmols = featurizer.defeaturize(generated_data)

# Filter out None values and convert to SMILES
nmols = [mol for mol in nmols if mol is not None]
nmols_smiles = [Chem.MolToSmiles(mol) for mol in nmols]
nmols_smiles_unique = list(OrderedDict.fromkeys(nmols_smiles))
print(f"{len(nmols_smiles_unique)} unique valid molecules generated")

# Visualization of generated molecules
nmols_viz = [Chem.MolFromSmiles(smile) for smile in nmols_smiles_unique if smile is not None]
img = Draw.MolsToGridImage(nmols_viz, molsPerRow=5, subImgSize=(250, 250), maxMols=100)
plt.show(img)
