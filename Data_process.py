import pandas as pd
import sys
import random
from rdkit import Chem
import torch
from utils import *

sys.path.append('/')


# Convert the sets to DataFrame and save as CSV (without column names)
def save_to_csv(entries, filename):
    # The entries contain [Drug_SMILES, Protein_Seq, Interaction_Value]
    df = pd.DataFrame(entries)
    df.to_csv(filename, index=False, header=False)

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}

def label_sequence(line, smi_ch_ind, MAX_SEQ_LEN=1000):
    X = np.zeros(MAX_SEQ_LEN, np.int64())
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]
    return X

def label_smiles(line, smi_ch_ind, MAX_SMI_LEN=100):
    X = np.zeros(MAX_SMI_LEN, dtype=np.int64())
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        X[i] = smi_ch_ind[ch]
    return X

def parse_sdf_file(file_path):
    """
    Parses an SDF file and returns a list of molecules.

    Args:
        file_path (str): The path to the SDF file.

    Returns:
        list: A list of molecules parsed from the SDF file.
    """
    suppl = Chem.SDMolSupplier(file_path, sanitize=True, removeHs=True, strictParsing=True)
    molecules = []
    for mol in suppl:
        if mol is not None:
            molecules.append(mol)
    return molecules

def create_fold_setting_cold(df, fold_seed, frac, entities):
    """create cold-split where given one or multiple columns, it first splits based on
    entities in the columns and then maps all associated data points to the partition

    Args:
            df (pd.DataFrame): dataset dataframe
            fold_seed (int): the random seed
            frac (list): a list of train/valid/test fractions
            entities (Union[str, List[str]]): either a single "cold" entity or a list of
                    "cold" entities on which the split is done

    Returns:
            dict: a dictionary of splitted dataframes, where keys are train/valid/test and values correspond to each dataframe
    """
    if isinstance(entities, str):
        entities = [entities]

    train_frac, val_frac, test_frac = frac

    # For each entity, sample the instances belonging to the test datasets
    test_entity_instances = [
        df[e]
        .drop_duplicates()
        .sample(frac=test_frac, replace=False, random_state=fold_seed)
        .values
        for e in entities
    ]

    # Select samples where all entities are in the test set
    test = df.copy()
    for entity, instances in zip(entities, test_entity_instances):
        test = test[test[entity].isin(instances)]

    if len(test) == 0:
        raise ValueError(
            "No test samples found. Try another seed, increasing the test frac or a "
            "less stringent splitting strategy."
        )

    # Proceed with validation data
    train_val = df.copy()
    for i, e in enumerate(entities):
        train_val = train_val[~train_val[e].isin(test_entity_instances[i])]

    val_entity_instances = [
        train_val[e]
        .drop_duplicates()
        .sample(frac=val_frac / (1 - test_frac), replace=False, random_state=fold_seed)
        .values
        for e in entities
    ]
    val = train_val.copy()
    for entity, instances in zip(entities, val_entity_instances):
        val = val[val[entity].isin(instances)]

    if len(val) == 0:
        raise ValueError(
            "No validation samples found. Try another seed, increasing the test frac "
            "or a less stringent splitting strategy."
        )

    train = train_val.copy()
    for i, e in enumerate(entities):
        train = train[~train[e].isin(val_entity_instances[i])]

    return {
        "train": train.reset_index(drop=True),
        "valid": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    print('The code uses CPU!!!')

from transformers import BertModel, BertTokenizer
from transformers import AutoModel, AutoTokenizer

prot_bert_bfd_path = "./Prot_Bert_bfd"
prot_tokenizer = BertTokenizer.from_pretrained(prot_bert_bfd_path, do_lower_case=False)
prot_model = BertModel.from_pretrained(prot_bert_bfd_path).to(device)

PubChem10M_path = "./PubChem10M"
chem_tokenizer = AutoTokenizer.from_pretrained(PubChem10M_path, do_lower_case=False)
chem_model = AutoModel.from_pretrained(PubChem10M_path).to(device)

def create_CPI_dataset(type, dataset='Davis', save_dir='./data/Davis', train_ratio=None):
    # load dataset
    data_file = os.path.join('./data/Davis', dataset + '.txt')

    all_entries = []
    all_p_entries = []
    all_n_entries = []
    drug_smiles = []
    pro_seqs = []
    pro_keys = []
    drug_keys = []
    proteins = {}
    drugs = {}
    complex = {}
    ligand_graph = {}
    protein_graph = {}

    for line in open(data_file, 'r').readlines():
        if line.strip() == '':
            continue
        arrs = line.split(' ')
        entry = [str(arrs[0]), arrs[1], arrs[2], arrs[3], float(arrs[4])]

        if arrs[0] not in drug_keys:
            drug_keys.append(str(arrs[0]))
        if arrs[1] not in pro_keys:
            pro_keys.append(arrs[1])
        if arrs[2] not in drug_smiles:
            drug_smiles.append(arrs[2])
        if arrs[3] not in pro_seqs:
            pro_seqs.append(arrs[3])

        proteins[arrs[1]] = arrs[3]
        drugs[str(arrs[0])] = arrs[2]
        complex[str(arrs[0]),arrs[1]] = str(arrs[0]) + '+' + str(arrs[1]) + '_graph.pth'
        ligand_graph[str(arrs[0])] = str(arrs[0]) + '.sdf'
        protein_graph[arrs[1]] = arrs[1] + '.pdb'

        all_entries.append(entry)
        if float(arrs[4]) == 1:
            all_p_entries.append(entry)
        if float(arrs[4]) == 0:
            all_n_entries.append(entry)

    drug_keys = list(set(drug_keys))
    pro_keys = list(set(pro_keys))
    drug_smiles = list(set(drug_smiles))
    pro_seqs = list(set(pro_seqs))

    print('drug number:', len(drug_smiles), len(drug_keys))
    print('protein number:', len(pro_seqs), len(pro_keys))
    print('number of entries:', len(all_entries))
    print('number of positive entries:', len(all_p_entries))
    print('number of negative entries:', len(all_n_entries))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if type=="warmup":
        train_file = os.path.join(save_dir, 'train_set.csv')
        dev_file = os.path.join(save_dir, 'dev_set.csv')
        test_file = os.path.join(save_dir, 'test_set.csv')
        print("warmup")
    if type=="unseen":
        train_file = os.path.join(save_dir, 'unseen_train_set.csv')
        dev_file = os.path.join(save_dir, 'unseen_dev_set.csv')
        test_file = os.path.join(save_dir, 'unseen_test_set.csv')
        print("unseen")
    if type=="drugcold":
        train_file = os.path.join(save_dir, 'drug_cold_train_set.csv')
        dev_file = os.path.join(save_dir, 'drug_cold_dev_set.csv')
        test_file = os.path.join(save_dir, 'drug_cold_test_set.csv')
        print("drugcold")

    if type=="targetcold":
        train_file = os.path.join(save_dir, 'target_cold_train_set.csv')
        dev_file = os.path.join(save_dir, 'target_cold_dev_set.csv')
        test_file = os.path.join(save_dir, 'target_cold_test_set.csv')
        print("targetcold")

    if type == "Scarce":
        # 根据 train_ratio 生成带比例的文件名
        if train_ratio is None:
            raise ValueError("For type='Scarce', you must provide train_ratio (e.g., 0.05, 0.10, 0.20, 0.30).")
        ratio_percent = int(train_ratio * 100)  # 例如 5, 10, 20, 30
        train_file = os.path.join(save_dir, f'train_{ratio_percent}%.csv')
        dev_file = os.path.join(save_dir, f'dev_{ratio_percent}%.csv')
        test_file = os.path.join(save_dir, f'test_{ratio_percent}%.csv')
        print(f"Scarce dataset with training ratio {train_ratio * 100}%")

    if os.path.exists(train_file) and os.path.exists(dev_file) and os.path.exists(test_file):
        print("已经存在划分好的数据集，正在加载...")
        # 直接加载已存在的划分好的数据集
        train_set = pd.read_csv(train_file, header=None)
        dev_set = pd.read_csv(dev_file, header=None)
        test_set = pd.read_csv(test_file, header=None)

    else:

        if type=="warmup":
            np.random.seed(10)
            # shuffle
            random.shuffle(all_entries)
            random.shuffle(all_p_entries)
            random.shuffle(all_n_entries)

            # to used all data
            used_entries = all_entries

            # split training, validation and test sets
            used_entries = np.array(used_entries)
            ratio = 0.8
            n = int(ratio * len(used_entries))
            train_set, dataset_ = used_entries[:n], used_entries[n:]
            ratio = 0.5
            n = int(ratio * len(dataset_))
            dev_set, test_set = dataset_[:n], dataset_[n:]

            # Save datasets as CSV files (no column headers)
            save_to_csv(train_set, train_file)
            save_to_csv(dev_set, dev_file)
            save_to_csv(test_set, test_file)
            print("Warm up !!! Training set, validation set, and test set have been saved")

        if type=="drugcold":
            # 冷启动数据集的划分
            print("Creating cold-start datasets...")
            # Create cold-start train/valid/test sets
            cold_train_set = create_fold_setting_cold(
                df=pd.DataFrame(all_entries, columns=["drug_ID", "Protein_ID", "drug", "protein", "interaction"]),
                fold_seed=10,
                frac=[0.8, 0.1, 0.1],
                entities=["drug"]  # 仅按药物进行冷启动划分
            )
            cold_train_set["train"].to_csv(os.path.join(save_dir, 'drug_cold_train_set.csv'), index=False)
            cold_train_set["valid"].to_csv(os.path.join(save_dir, 'drug_cold_dev_set.csv'), index=False)
            cold_train_set["test"].to_csv(os.path.join(save_dir, 'drug_cold_test_set.csv'), index=False)
            # 加载
            train_set = pd.read_csv(train_file, header=None)
            dev_set = pd.read_csv(dev_file, header=None)
            test_set = pd.read_csv(test_file, header=None)
            print("Cold-start datasets have been saved.")

        if type == "targetcold":
            # Create target cold-start dataset (e.g., target cold-start split)
            cold_target_set = create_fold_setting_cold(
                df=pd.DataFrame(all_entries, columns=["drug_ID", "Protein_ID", "drug", "protein", "interaction"]),
                fold_seed=10,
                frac=[0.8, 0.1, 0.1],
                entities=["protein"]  # 仅按靶标进行冷启动划分
            )
            cold_target_set["train"].to_csv(os.path.join(save_dir, 'target_cold_train_set.csv'), index=False)
            cold_target_set["valid"].to_csv(os.path.join(save_dir, 'target_cold_dev_set.csv'), index=False)
            cold_target_set["test"].to_csv(os.path.join(save_dir, 'target_cold_test_set.csv'), index=False)
            # 加载
            train_set = pd.read_csv(train_file, header=None)
            dev_set = pd.read_csv(dev_file, header=None)
            test_set = pd.read_csv(test_file, header=None)
            print("Cold-start datasets have been saved.")

        if type == "unseen":
            # Create Unseen cold-start dataset (e.g., neither drug nor protein has been seen)
            cold_unseen_set = create_fold_setting_cold(
                df=pd.DataFrame(all_entries, columns=["drug_ID", "Protein_ID", "drug", "protein", "interaction"]),
                fold_seed=10,
                frac=[0.8, 0.1, 0.1],
                entities=["drug", "protein"]  # 既按药物也按靶标进行冷启动划分
            )
            cold_unseen_set["train"].to_csv(os.path.join(save_dir, 'unseen_train_set.csv'), index=False)
            cold_unseen_set["valid"].to_csv(os.path.join(save_dir, 'unseen_dev_set.csv'), index=False)
            cold_unseen_set["test"].to_csv(os.path.join(save_dir, 'unseen_test_set.csv'), index=False)
            # 加载
            train_set = pd.read_csv(train_file, header=None)
            dev_set = pd.read_csv(dev_file, header=None)
            test_set = pd.read_csv(test_file, header=None)
            print("Cold-start datasets have been saved.")

        if type == "Scarce":

            if train_ratio is None:
                raise ValueError(
                    "For type='Scarce', the 'train_ratio' parameter must be provided (e.g., 0.05, 0.10, 0.20, 0.30).")
            np.random.seed(10)
            random.shuffle(all_entries)
            used_entries = np.array(all_entries)

            n_train = int(train_ratio * len(used_entries))
            train_set = used_entries[:n_train]
            rest_set = used_entries[n_train:]

            # 剩余部分按 1:9 分为验证集和测试集（验证集占剩余 10%）
            n_val = int(0.1 * len(rest_set))
            dev_set = rest_set[:n_val]
            test_set = rest_set[n_val:]

            # 保存为 CSV 文件（无列头）
            save_to_csv(train_set, train_file)
            save_to_csv(dev_set, dev_file)
            save_to_csv(test_set, test_file)
            print(
                f"Scarce dataset created with train_ratio={train_ratio} (train: {len(train_set)}, dev: {len(dev_set)}, test: {len(test_set)})")

    # drug to fcfp
    d_LM = {}
    d_l = 100
    for i in tqdm(range(len(drug_keys))):
        key = drug_keys[i]
        smilestr = drugs[str(key)]
        chem_input = chem_tokenizer.batch_encode_plus([smilestr], add_special_tokens=True, padding=True)
        c_IDS = torch.tensor(chem_input["input_ids"]).to(device)
        c_a_m = torch.tensor(chem_input["attention_mask"]).to(device)
        with torch.no_grad():
            chem_outputs = chem_model(input_ids=c_IDS, attention_mask=c_a_m)
        chem_feature = chem_outputs.last_hidden_state.squeeze(0).to('cpu').data.numpy()  # .mean(dim=1)
        C_L = chem_feature.shape[0]
        molecule_LM = torch.zeros((d_l, 768), device=device)
        if C_L >= 100:
            molecule_LM[:, :] = torch.tensor(chem_feature[0:100, :])
        else:
            molecule_LM[:C_L, :] = torch.tensor(chem_feature)
        d_LM[key] = molecule_LM

    # protein to esm
    p_LM = {}
    p_l = 1200
    for i in tqdm(range(len(pro_keys))):
        key = pro_keys[i]
        seq = proteins[key]
        protein_input = prot_tokenizer.batch_encode_plus([" ".join(seq)], add_special_tokens=True,
                                                         padding=True)  # "longest", max_length=1200, truncation=True, return_tensors='pt')
        p_IDS = torch.tensor(protein_input["input_ids"]).to(device)
        p_a_m = torch.tensor(protein_input["attention_mask"]).to(device)
        with torch.no_grad():
            prot_outputs = prot_model(input_ids=p_IDS, attention_mask=p_a_m)
        prot_feature = prot_outputs.last_hidden_state.squeeze(0).to('cpu').data.numpy()
        P_L = prot_feature.shape[0]
        protein_LM = torch.zeros((p_l, 1024), device=device)
        if P_L >= 1200:
            protein_LM[:, :] = torch.tensor(prot_feature[0:1200, :])
        else:
            protein_LM[:P_L, :] = torch.tensor(prot_feature)
        p_LM[key] = protein_LM

# create target sequence
    target_sequence = {}
    target_lenth = {}
    protein_max = 1000
    for i in tqdm(range(len(pro_keys))):
        key = pro_keys[i]
        proteinstr = proteins[key]
        proteinint = torch.from_numpy(label_sequence(proteinstr, CHARPROTSET, protein_max))
        target_sequence[key] = proteinint
        target_lenth[key] = len(proteinstr)
# create smiles
    smile_sequence = {}
    smile_lenth = {}
    smile_max = 100
    for i in tqdm(range(len(drug_keys))):
        key = drug_keys[i]
        smilestr = drugs[str(key)]
        smileint = torch.from_numpy(label_smiles(smilestr, CHARISOSMISET, smile_max))
        smile_sequence[str(key)] = smileint
        smile_lenth[str(key)] = len(smilestr)

    graph_complex = {}
    protein_embeddings = ['ankh_base', 'esm2_t6']
    ligand_embeddings = ['ChemBERTa_77M']
    for i in tqdm(range(len(all_entries))):
        data_dir = "./ComplexGraph_6A/" + complex[all_entries[i][0], all_entries[i][1]]
        graph = torch.load(data_dir)
        id = graph.id
        pos = graph.pos
        x = graph.x

        for emb in protein_embeddings:
            emb_tensor = graph[emb]
            if emb_tensor is not None:
                x = torch.concatenate((x, emb_tensor), axis=1)

        ligand_embedding = None
        for emb in ligand_embeddings:
            emb_vector = graph[emb]
            if emb_vector is None:
                print(f"Embedding {emb} not found for {id}")
            else:
                if ligand_embedding is None:
                    ligand_embedding = emb_vector
                else:
                    ligand_embedding = torch.concatenate((ligand_embedding, emb_vector), axis=1)
                    ligand_embedding = ligand_embedding.float()

        edge_index = graph.edge_index
        edge_index_lig = graph.edge_index_lig
        edge_index_prot = graph.edge_index_prot

        edge_attr = graph.edge_attr
        edge_attr_lig = graph.edge_attr_lig
        edge_attr_prot = graph.edge_attr_prot

        graph_complex[all_entries[i][0], all_entries[i][1]] = {
            'x': x,
            'edge_index': edge_index,
            'edge_index_lig': edge_index_lig,
            'edge_index_prot': edge_index_prot,
            'edge_attr_lig': edge_attr_lig,
            'edge_attr_prot': edge_attr_prot,
            'edge_attr': edge_attr,
            'ligand_embedding': ligand_embedding,
            'pos': pos,
            'id': id
        }

    # 'data/davis_fold_0_train.csv' or data/kiba_fold_0__train.csv'
    # train dataset construct
    train_drug_keys, train_pro_keys, train_Y = np.asarray(train_set)[:, 0], np.asarray(train_set)[:, 1], np.asarray(train_set)[:, 4]
    train_dataset = DTADataset(root='data', dataset=dataset + '_' + 'train', drug_key=train_drug_keys, target_key=train_pro_keys,
                               y=train_Y.astype(float), target_seq=target_sequence, smile_sequence=smile_sequence,
                               complex_graph=graph_complex, p_LM=p_LM, d_LM=d_LM, target_lenth=target_lenth, smile_lenth=smile_lenth)
    # valid dataset construct
    dev_drug_keys, dev_pro_keys, dev_Y = np.asarray(dev_set)[:, 0], np.asarray(dev_set)[:, 1], np.asarray(
        dev_set)[:, 4]
    dev_dataset = DTADataset(root='data', dataset=dataset + '_' + 'dev', drug_key=dev_drug_keys,
                             target_key=dev_pro_keys, y=dev_Y.astype(float), target_seq=target_sequence,
                             smile_sequence=smile_sequence, complex_graph=graph_complex, p_LM=p_LM, d_LM=d_LM, target_lenth=target_lenth, smile_lenth=smile_lenth)
    # test dataset construct
    test_drug_keys, test_pro_keys, test_Y = np.asarray(test_set)[:, 0], np.asarray(test_set)[:, 1], np.asarray(
        test_set)[:, 4]
    test_dataset = DTADataset(root='data', dataset=dataset + '_' + 'test', drug_key=test_drug_keys,
                              target_key=test_pro_keys, y=test_Y.astype(float), target_seq=target_sequence,
                              smile_sequence=smile_sequence, complex_graph=graph_complex, p_LM=p_LM, d_LM=d_LM, target_lenth=target_lenth, smile_lenth=smile_lenth)
    # temp_y = test_Y.astype(float) # for test
    # print(type(temp_y))
    return train_dataset, dev_dataset, test_dataset


if __name__ == '__main__':
    create_CPI_dataset('human', 1)

