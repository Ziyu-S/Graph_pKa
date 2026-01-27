# Graph_pKa: 
Graph_pKa are graph-based models trained on a custom dataset generated from high-throughput molecular dynamics simulations using the advanced polarizable AMOEBA force field to predict pKa values of four ionizable protein residues: *Asp*, *Lys*, *Glu*, and *His*.  

This repository contains the complete implementation of our paper:  

**Graph-Based Deep Learning Models for Predicting pKa Values of Protein-Ionizable Residues via Physically Inspired Feature Engineering**, https://pubs.acs.org/doi/10.1021/acs.jcim.5c01681

<img width="500" height="321" alt="image" src="https://github.com/user-attachments/assets/c4f7d9b6-1e94-43c8-8bd9-02c51d7e5d3b" />

## **Environment:**
### To install the reuquired environment (python=3.11.3) using conda (recomended):

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
## **Models:**
