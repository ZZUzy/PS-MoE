# PS-MoE

**A reliability-aware pairwise learning framework for drug–target interaction prediction with interaction-driven graph representations**

PS-MoE explicitly models atom–residue interaction pathways and adaptively integrates sequence- and graph-based experts through reliability supervision. The framework improves DTI prediction and generalization, particularly for unseen drug–target pairs and scarce-data settings.

![PS-MoE framework](https://github.com/user-attachments/assets/97a80fb1-d9a5-42c2-aefe-2f8fc5e17e36)

## Contents

- [Installation](#installation)
- [Demo data](#demo-data)
- [Reproducibility](#reproducibility)
- [Contact](#contact)

## Installation

PS-MoE is implemented in Python and PyTorch. A CUDA-enabled GPU is recommended for training and inference.

### Environment

The experiments are conducted using the following main environment:

- Python 3.9
- PyTorch 2.1.1
- NumPy 1.26.3
- Pandas 2.3.3
- Scikit-learn 1.6.1
- RDKit 2025.9.1
- PyTorch Geometric
- Transformers

### Setup

```bash
# Clone the repository
git clone https://github.com/ZZUzy/PS-MoE.git
cd PS-MoE

# Create the environment
conda create -n psmoe python=3.9
conda activate psmoe

# Install PyTorch according to the local CUDA version
pip install torch==2.1.1

# Install the remaining dependencies
conda install -c conda-forge rdkit
pip install torch-geometric transformers numpy pandas scikit-learn tqdm scipy
```

### Required files

Before running PS-MoE, download or prepare the following resources and update their paths in `Data_process.py`:

```text
Prot_Bert_bfd/       # ProtBERT-BFD model
PubChem10M/          # molecular language model
ComplexGraph_6A/     # preprocessed interaction-driven graphs
```

The default paths used in `Data_process.py` are:

```python
prot_bert_bfd_path = "./Prot_Bert_bfd"
PubChem10M_path = "./PubChem10M"
data_dir = "./ComplexGraph_6A/"
```

The graph preprocessing scripts are provided in `PS-MoE/Complex`, including protein and ligand embedding generation, graph construction, and dataset construction.

## Demo data

The repository currently provides raw interaction data for Davis and DrugBank:

```text
PS-MoE/data/davis/Davis.txt
PS-MoE/data/DrugBank/DrugBank.txt
```

Each line follows the format:

```text
Drug_ID Protein_ID SMILES Protein_sequence Interaction_label
```

where `Interaction_label` is `1` for an interacting pair and `0` for a non-interacting pair.

The included files can be used as example inputs after the corresponding pretrained representations and interaction-driven graph files have been prepared. If the data are retained under the current nested directory, update the dataset path in `Data_process.py` accordingly.

## Reproducibility

The main training script is `Train_cpi.py`. The current implementation uses script-level configuration rather than command-line arguments.

### Dataset selection

Set the dataset in `Train_cpi.py`:

```python
CPIdatasets = ['Davis']
# or
CPIdatasets = ['DrugBank']
```

### Evaluation settings

Modify the `type` argument in the following line:

```python
train_data, dev_data, test_data = create_CPI_dataset(
    type="warmup",
    dataset=dataset
)
```

Supported settings are:

```text
warmup      warm-start split
drugcold    drug cold-start split
targetcold  target cold-start split
unseen      drug–target pair cold-start split
Scarce      scarce-data split
```

For scarce-data experiments, specify the training ratio:

```python
train_data, dev_data, test_data = create_CPI_dataset(
    type="Scarce",
    dataset=dataset,
    train_ratio=0.05
)
```

The ratios used in the manuscript are `0.05`, `0.10`, `0.20`, and `0.30`.

### Training configuration

To match the experimental settings reported in the manuscript, use:

```python
SEED = 3407
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
LR = 5e-5
NUM_EPOCHS = 100

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)
```

Run training and evaluation with:

```bash
python Train_cpi.py
```

The trained model and prediction results are saved to:

```text
Pretrain_Models/
Pretrain_results/
```

For different datasets, it is recommended to save checkpoints using the dataset name:

```python
model_file_name = f"Pretrain_Models/{dataset}.model"
```

The current public code provides the main model and training workflow. Reproducing the complete reported results additionally requires the corresponding fold-specific data splits, pretrained model files, and preprocessed interaction-driven graphs.

## Contact

For questions regarding the code or data, please open an issue in this repository or contact:

**Zhen Tian**  
Email: `ieztian@zzu.edu.cn`
