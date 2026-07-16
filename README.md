# PS-MoE

**A reliability-aware pairwise learning framework for drug–target interaction prediction with interaction-driven graph representations**

PS-MoE explicitly models atom–residue interaction pathways and adaptively integrates sequence- and graph-based experts through reliability supervision. The framework improves DTI prediction and generalization, particularly for unseen drug–target pairs and scarce-data settings.

![PS-MoE framework](https://github.com/user-attachments/assets/97a80fb1-d9a5-42c2-aefe-2f8fc5e17e36)

## Contents

- [Installation](#installation)
- [Repository structure](#repository-structure)
- [Demo data](#demo-data)
- [Complex construction](#complex-construction)
- [Reproducibility](#reproducibility)
- [Contact](#contact)

## Installation

PS-MoE is implemented in Python and PyTorch. A CUDA-enabled GPU is recommended for feature extraction, training, and inference.

### Environment

The main environment used in the experiments includes:

- Python 3.9
- PyTorch 2.1.1
- PyTorch Geometric
- RDKit 2025.9.1
- Transformers
- NumPy 1.26.3
- Pandas 2.3.3
- Scikit-learn 1.6.1
- SciPy
- Biopython
- tqdm
- ANKH and ESM packages for complex preprocessing

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

# Install the main dependencies
conda install -c conda-forge rdkit
pip install torch-geometric transformers numpy pandas scikit-learn scipy biopython tqdm ankh fair-esm
```

### Pre-trained models

Before running the main training code, download the pre-trained sequence models and place them in the repository root, or update their paths in `Data_process.py`:

```text
Prot_Bert_bfd/       # protein language model
PubChem10M/          # molecular language model
ComplexGraph_6A/     # preprocessed interaction-driven graph files
```

The default paths in `Data_process.py` are:

```python
prot_bert_bfd_path = "./Prot_Bert_bfd"
PubChem10M_path = "./PubChem10M"
graph_path = "./ComplexGraph_6A/"
```

## Repository structure

```text
PS-MoE/
├── Data_process.py                 # dataset splitting and feature loading
├── Model_EMOE.py                   # PS-MoE model
├── Train_cpi.py                    # training and evaluation
├── metric.py                       # evaluation metrics
├── utils.py                        # utility functions
├── Complex_dataprep_workflow.py    # complex preprocessing workflow
├── PS-MoE/
│   ├── Complex/                    # complex feature and graph construction scripts
│   │   ├── ankh_features.py
│   │   ├── esm_features.py
│   │   ├── chemberta_features.py
│   │   ├── graph_construction.py
│   │   └── construct_dataset.py
│   └── data/
│       ├── davis/
│       └── DrugBank/
├── LICENSE
└── README.md
```

## Demo data

The repository provides Davis and DrugBank interaction data:

```text
PS-MoE/data/davis/Davis.txt
PS-MoE/data/DrugBank/DrugBank.txt
```

Each line follows the format:

```text
Drug_ID Protein_ID SMILES Protein_sequence Interaction_label
```

where `Interaction_label` is `1` for an interacting pair and `0` for a non-interacting pair.

A small set of drug–target complex data is also provided in the Davis directory for testing the complex-processing and graph-loading workflow. These demo files are intended for a quick functional test rather than reproduction of the complete Davis results.

Two types of demo files may be used:

1. **Preprocessed graph files (`*_graph.pth`)**: place them in `ComplexGraph_6A/` and keep their names consistent with the pair identifiers in `Davis.txt`.
2. **Raw complex files (`.pdb` and `.sdf`)**: process them using the scripts in `PS-MoE/Complex/` before loading them with the main model.

For a drug–target pair with drug ID `D` and protein ID `P`, `Data_process.py` expects the graph name:

```text
D+P_graph.pth
```

The public demo data can therefore be used to verify that graph files are successfully loaded and passed through PS-MoE before running the complete dataset.

## Complex construction

The `PS-MoE/Complex/` directory contains the scripts used to convert protein–ligand complexes into interaction-driven graph representations.

### Input format

Each complex is represented by a protein structure in PDB format and a ligand structure in SDF format. The PDB and SDF files must have the same base name:

```text
complex_id.pdb
complex_id.sdf
```

For PS-MoE datasets, the recommended base name is the corresponding drug–target pair identifier:

```text
Drug_ID+Protein_ID.pdb
Drug_ID+Protein_ID.sdf
```

The SDF file must contain three-dimensional ligand coordinates. A single SDF file may also contain multiple ligands; in this case, the graph-construction script assigns ligand-specific suffixes to the generated files.

### Processing scripts

- `ankh_features.py` extracts residue-level ANKH embeddings from protein PDB files.
- `esm_features.py` extracts residue-level ESM embeddings from protein PDB files.
- `chemberta_features.py` extracts ligand embeddings from molecules stored in SDF files.
- `graph_construction.py` parses matched PDB/SDF files and constructs PyTorch Geometric graph objects.
- `construct_dataset.py` optionally combines individual graph objects and labels into a serialized dataset.
- `Complex_dataprep_workflow.py` provides a wrapper for executing the above stages sequentially.

### Interaction-driven graph generation

The graph-construction code performs the following operations:

1. Parses protein residues and ligand heavy atoms from the matched PDB and SDF files.
2. Identifies protein residues containing atoms within 6 Å of each ligand atom.
3. Represents ligand atoms and nearby protein residues as graph nodes.
4. Constructs atom–residue interaction edges and records distances to residue backbone/reference atoms as edge features.
5. Incorporates ANKH, ESM, and ChemBERTa representations when the corresponding embedding files are available.
6. Saves each graph as:

```text
complex_id_graph.pth
```

A graph-generation log is written to the `graph_generation_logs/` directory, including the number of successful, skipped, and failed complexes.

### Running the workflow

The intended workflow can be launched with:

```bash
python PS-MoE/Complex_dataprep_workflow.py \
  --data_dir <directory_containing_matched_pdb_and_sdf_files> \
  --save_dir <output_name_or_directory> \
  --y_data <label_file.csv_or_json>
```

Before running the workflow, update the local paths of the ANKH, ESM, and ChemBERTa checkpoints in the corresponding scripts. The preprocessing scripts retain several local default paths, so these paths must be replaced with locations available on the current machine.

The main PS-MoE training code directly loads the individual `*_graph.pth` files from `ComplexGraph_6A/`. Therefore, running `construct_dataset.py` is optional when the goal is to train PS-MoE with `Data_process.py`.

## Reproducibility

The main training entry is `Train_cpi.py`. The current implementation uses configuration variables inside the Python scripts rather than command-line arguments.

### 1. Configure dataset paths

The repository stores the Davis file under:

```text
PS-MoE/data/davis/Davis.txt
```

However, the current default path in `Data_process.py` is:

```python
data_file = os.path.join('./data/Davis', dataset + '.txt')
```

Either move the dataset to `./data/Davis/` or change the path to the uploaded repository structure, for example:

```python
data_file = os.path.join('./PS-MoE/data/davis', dataset + '.txt')
save_dir = './PS-MoE/data/davis'
```

Apply the corresponding path change for DrugBank when required.

### 2. Select the dataset

Edit `Train_cpi.py`:

```python
CPIdatasets = ['Davis']
# or
CPIdatasets = ['DrugBank']
```

### 3. Select the evaluation setting

Modify the `type` argument in:

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

The ratios evaluated in the manuscript are:

```text
0.05, 0.10, 0.20, and 0.30
```

### 4. Training configuration

The manuscript reports the following main settings:

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

The current `Train_cpi.py` uses `LR = 1e-4` without weight decay. Update these values when reproducing the settings reported in the manuscript.

### 5. Run training

```bash
python Train_cpi.py
```

The model checkpoints and results are saved to:

```text
Pretrain_Models/
Pretrain_results/
```

To avoid overwriting checkpoints across datasets, use:

```python
model_file_name = f"Pretrain_Models/{dataset}.model"
```

The public training script performs one seeded split. To reproduce the mean and standard deviation reported over five runs, repeat the experiment using the corresponding split files or seeds and aggregate the resulting metrics.

## Contact

For questions regarding the code or data, please open an issue in this repository or contact:

**Zhen Tian**  
Email: `ieztian@zzu.edu.cn`

## License

This project is released under the Apache License 2.0.
