# Graph_pKa: 
Graph_pKa are graph-based models trained on a custom dataset generated from high-throughput molecular dynamics simulations using the advanced polarizable AMOEBA force field to predict pKa values of four ionizable protein residues: *Asp*, *Lys*, *Glu*, and *His*.  

This repository contains the complete implementation of our paper:  

**Graph-Based Deep Learning Models for Predicting pKa Values of Protein-Ionizable Residues via Physically Inspired Feature Engineering**, https://pubs.acs.org/doi/10.1021/acs.jcim.5c01681

<img width="500" height="321" alt="image" src="https://github.com/user-attachments/assets/c4f7d9b6-1e94-43c8-8bd9-02c51d7e5d3b" />

## **Environment:**
### To install the reuquired environment (python=3.12.2) using conda (recomended):

```bash
conda env create -f environment.yml
conda activate pKa
```
or

```bash
pip install -r requirements.txt
```
## **Data:**
The processed data for each protein residue generated from simulations are provided in `/Data/`, including data from [PKAD-2](http://compbio.clemson.edu/PKAD-2/) (for wild-type proteins, `../WT/`) and [PKAD-R](http://compbio.clemson.edu/PKAD-R/) (for mutant proteins, `../Mutant/`).
## **Model Training:**
All models for the three architectures (**GCN**, **GIN**, and **GAT**) obtained during the hyperparameter grid search and trained on five datasets (with different radii) are provided in this repository.

You can also run the grid search training from scratch.
### Step 1: Generate the training datasets:

```bash
python create_data.py
```
### Step 2: Train the model (GAT as an example):

```bash
python GAT.py
```

## **Make Predictions:**
**Note:** Although Conda provides a Tinker **8.11.3** package installation via:
```bash
conda install bioconda::tinker
```

the packaged version contains known source-code issues that result in invalid `.uind` files (induced dipole moment files), which are required files for model inference.

In addition, Tinker **8.11.3** is no longer available. Therefore, a newer Tinker release (`Tinker_Output_Processing.py` is updated for Tinker **25.5.3**) must be downloaded and compiled from [TinkerTools](https://github.com/TinkerTools). 
Detailed compilation instructions are available [here](https://dasher.wustl.edu/tinker/).

---

### Step 1: HTP Tinker Simulations:
```bash
python Tinker_EM.py
```
### Step 2: Feature Extraction from Simulations Files and Data Generation:
```bash
python Tinker_Output_Processing.py
```
### Step 3: Model Inference:
```bash
python Predict.py
```
## **Benchmarking Results:**
Predictions from three benchmarking models (**PROPKA**, **DeepKa**, and **PKAI+** on both **WT** and **Mutant** proteins, as reported in the paper from papaer for benchmarking are provided in the `/Benchmarking/` dir.

## **Contact:**
For any questions regarding the code or the papaer, please feel free to contact: **zsong01@villanova.edu**
