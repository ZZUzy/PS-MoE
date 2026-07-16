import os
from torch_geometric.data import InMemoryDataset, DataLoader, Batch
from torch_geometric import data as DATA
import torch
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import csv


def convert_csv_to_json(input_file, output_file):
    """
    Convert a CSV file to a JSON file with a specific structure.

    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to the output JSON file

    Returns:
        str: Path to the generated JSON file
    """
    # Read CSV data and convert to the desired JSON structure
    data_dict = {}
    try:
        with open(input_file, "r") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=";")
            for row in csv_reader:
                if len(row) == 2:  # Ensure there are exactly two items: key and value
                    key = row[0].strip()  # The outer key
                    value = row[1].strip().lstrip("_")  # Remove leading underscores
                    try:
                        # Build the required nested structure
                        data_dict[key] = {
                            "log_kd_ki": float(value),
                            "dataset": ["general"],  # Default dataset list
                        }
                    except ValueError:
                        # Skip non-numeric rows
                        print(f"Skipping row: {row}")

        # Write dictionary to JSON file
        with open(output_file, "w") as json_file:
            json.dump(data_dict, json_file, indent=4)

        print(f"Data successfully written to {output_file}")
        return output_file

    except Exception as e:
        print(f"Error converting CSV to JSON: {e}")
        sys.exit(1)


def process_tensor(tensor):
    heads_mean = tensor.mean(dim=1).squeeze(0)

    drug_mean = heads_mean.mean(dim=1, keepdim=True).squeeze(1)

    drug_mean_shape = drug_mean.shape[0]+1
    print(drug_mean_shape)
    target_mean = heads_mean.mean(dim=0, keepdim=True).squeeze(0)
    target_mean_shape = target_mean.shape[0]+1
    print(target_mean_shape)
    def normalize(x):
        return (x - x.min()) / (x.max() - x.min())
    print('***************************************************')
    drug_normalized = normalize(drug_mean[1:drug_mean_shape])
    print(drug_normalized.shape)
    target_normalized = normalize(target_mean[1:target_mean_shape])
    print(target_normalized.shape)

    return drug_normalized, target_normalized

# initialize the dataset
class DTADataset(InMemoryDataset):
    def __init__(self, root='D:\\DZY\\WGNN-DTA-main\\WGNN-DTA-main\\data\\', dataset='davis',
                  transform=None, pre_transform=None, drug_key=None, target_key=None, y=None, target_seq=None, smile_sequence=None, complex_graph=None, p_LM=None, d_LM=None, target_lenth=None, smile_lenth=None):
        super(DTADataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.target_key = target_key
        self.drug_key = drug_key
        self.y = y
        self.target_seq = target_seq
        self.smile_sequence = smile_sequence
        self.complex_graph = complex_graph
        self.p_LM = p_LM
        self.d_LM = d_LM
        self.target_lenth = target_lenth
        self.smile_lenth = smile_lenth
        self.process(target_key, drug_key, y, target_seq, smile_sequence, complex_graph, p_LM, d_LM, target_lenth, smile_lenth)

    @property
    def raw_file_names(self):
        pass

    # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '_data_mol.pt', self.dataset + '_data_pro.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, target_key, drug_key, y, target_seq, smile_sequence, complex_graph, p_LM, d_LM, target_lenth, smile_lenth):
        assert (len(target_key) == len(drug_key) and len(drug_key) == len(y)), 'The three lists must have the same length!'
        data_list_graph = []
        data_list_target_seq_LM = []
        data_list_smile_sequence_LM = []
        data_list_target_length = []
        data_list_smile_length = []

        print('loading tensors ...')
        for i in tqdm(range(len(drug_key))):
            drug_keys = drug_key[i]
            tar_keys = target_key[i]
            labels = y[i]
            graph = complex_graph[str(drug_keys), tar_keys]
            p_LMs = p_LM[tar_keys]
            d_LMs = d_LM[str(drug_keys)]
            target_lenths = target_lenth[tar_keys]
            smile_lenths = smile_lenth[str(drug_keys)]

            x = graph['x']
            # edge_index = graph['edge_index']
            # edge_index_lig = graph['edge_index_lig']
            edge_index_prot = graph['edge_index_prot']
            # edge_attr = graph['edge_attr']
            # edge_attr_lig = graph['edge_attr_lig']
            edge_attr_prot = graph['edge_attr_prot']
            ligand_embedding = graph['ligand_embedding']
            pos = graph['pos']
            id = graph['id']

            # print(i,target_size,target_features.shape, target_edge_index.shape,target_edge_weight.shape,y[i])
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            Complex_Graph = DATA.Data(x=x.float(),
                               # edge_index=edge_index.long(),
                               # edge_attr=edge_attr.float(),
                               # edge_index_lig=edge_index_lig.long(),
                               edge_index_prot=edge_index_prot.long(),
                               # edge_attr_lig=edge_attr_lig.float(),
                               edge_attr_prot=edge_attr_prot.float(),
                               y=torch.tensor(labels, dtype=torch.float),
                               lig_emb=ligand_embedding,
                               pos=pos,
                               id=id
                               )

            data_list_graph.append(Complex_Graph)
            data_list_target_seq_LM.append(p_LMs)
            data_list_smile_sequence_LM.append(d_LMs)
            data_list_target_length.append(target_lenths)
            data_list_smile_length.append(smile_lenths)


        self.data_list_graph = data_list_graph
        self.data_list_target_seq_LM = data_list_target_seq_LM
        self.data_list_smile_sequence_LM = data_list_smile_sequence_LM
        self.data_list_target_length = data_list_target_length
        self.data_list_smile_length = data_list_smile_length

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # return GNNData_mol, GNNData_pro
        return self.data_list_target_seq_LM[idx], self.data_list_smile_sequence_LM[idx], self.data_list_graph[idx], self.data_list_target_length[idx], self.data_list_smile_length[idx]

def eva_imp(y_pred, y_true):
    res = (y_pred - y_true) ** 2
    return res

def entropy_balance(probs):
    probs = torch.clamp(probs, min=1e-9)
    N = probs.size(1)
    entropy = N * torch.sum(probs * torch.log(probs), dim=1)
    return torch.mean(entropy)

def uni_distill(logits1, logits2):
    prob1 = torch.softmax(logits1, dim=-1)
    prob2 = torch.softmax(logits2, dim=-1)
    mse = torch.mean((prob1 - prob2) ** 2, dim=-1)
    return torch.mean(mse)

# training function at each epoch
def train(model, device, train_loader, optimizer, epoch, loss_fn, TRAIN_BATCH_SIZE=512, log_file='weight_dist_log_No_Balance_Sim.csv'):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    LOG_INTERVAL = 10

    # 如果是第一个 epoch，初始化日志文件（写表头）
    if epoch == 1 and not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'batch_idx', 'sample_idx',
                             'w_channel_0', 'w_channel_1',
                             'dist_0', 'dist_1'])

    for batch_idx, data in enumerate(train_loader):
        data_seq = [torch.tensor(data).to(device) for data in data[0]]
        data_smile = [torch.tensor(data).to(device) for data in data[1]]
        complex_graph = data[2].to(device)
        seq_len = [torch.tensor(data).to(device) for data in data[3]]
        smile_len = [torch.tensor(data).to(device) for data in data[4]]
        optimizer.zero_grad()
        out, _, _ = model(data_seq, data_smile, complex_graph, seq_len, smile_len)

        loss_task_t = torch.mean(loss_fn(out['logits_t'], complex_graph.y.long().to(device)))
        loss_task_g = torch.mean(loss_fn(out['logits_g'], complex_graph.y.long().to(device)))
        loss_task_c = torch.mean(loss_fn(out['logits_c'], complex_graph.y.long().to(device)))

        w = out['channel_weight']
        # t_dist = eva_imp(out['logits_t_one'], complex_graph.y.view(-1, 1).float().to(device))
        #         # g_dist = eva_imp(out['logits_g_one'], complex_graph.y.view(-1, 1).float().to(device))
        #         # dist = torch.zeros(t_dist.shape[0], 2).to(device)
        #         # for i, _ in enumerate(t_dist):
        #         #     s = 1 / (t_dist[i] + 0.1) + 1 / (g_dist[i] + 0.1)
        #         #     dist[i][0] = (1 / (t_dist[i] + 0.1)) / s
        #         #     dist[i][1] = (1 / (g_dist[i] + 0.1)) / s
        # loss_sim = torch.mean(torch.mean((dist.detach() - w) ** 2, dim=-1))

        y_long = complex_graph.y.long().to(device)
        t_error = F.cross_entropy(out['logits_t'], y_long, reduction='none')
        g_error = F.cross_entropy(out['logits_g'], y_long, reduction='none')
        errors = torch.stack([t_error, g_error], dim=1)
        gamma = 1.0
        r_target = torch.softmax(-errors / gamma, dim=1)
        loss_sim = F.kl_div(
            torch.log(w.clamp_min(1e-8)),
            r_target.detach(),
            reduction='batchmean'
        )

        loss_ety = entropy_balance(w)

        loss_ud = uni_distill(out['c_proj'], (out['t_proj'] * w[:, 0].view(-1, 1) + out['g_proj'] * w[:, 1].view(-1, 1)).detach())

        loss = loss_task_c + loss_task_g + loss_task_t + 0.1 * (loss_ety + 0.1 * loss_sim) + 0.1 * loss_ud
        # loss = loss_task_c + loss_task_g + loss_task_t + 0.1 * loss_sim + 0.1 * loss_ud
        #loss = loss_task_c + loss_task_g + loss_task_t + 0.1 * loss_ud

        loss.backward()
        optimizer.step()

        # # ---------- 存储 w 和 dist（不需要 seq/smiles）----------
        # w_cpu = w.detach().cpu().numpy()
        # dist_cpu = dist.detach().cpu().numpy()
        # with open(log_file, 'a', newline='') as f:
        #     writer = csv.writer(f)
        #     for i in range(w_cpu.shape[0]):
        #         writer.writerow([
        #             epoch,
        #             batch_idx,
        #             i,
        #             w_cpu[i, 0],
        #             w_cpu[i, 1],
        #             dist_cpu[i, 0],
        #             dist_cpu[i, 1]
        #         ])

        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tTotal_Loss: {:.6f}\tFusion_Loss: {:.6f}\tGraph_Loss: {:.6f}\tText_Loss: {:.6f}\tloss_ety: {:.6f}\tloss_sim: {:.6f}\tloss_ud: {:.6f}'.format(epoch,
                                                                           batch_idx * TRAIN_BATCH_SIZE,
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item(),
                                                                           loss_task_c.item(),
                                                                           loss_task_g.item(),
                                                                           loss_task_t.item(),
                                                                           loss_ety.item(),
                                                                           loss_sim.item(),
                                                                           loss_ud.item(),
                                                                           ))
# predict
def predicting(model, device, loader):
    Y, P, S, id = [], [], [], []
    model.eval()

    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():

        for data in loader:
            data_seq = [torch.tensor(data).to(device) for data in data[0]]
            data_smile = [torch.tensor(data).to(device) for data in data[1]]
            complex_graph = data[2].to(device)
            seq_len = [torch.tensor(data).to(device) for data in data[3]]
            smile_len = [torch.tensor(data).to(device) for data in data[4]]
            out, attn, attn_prot = model(data_seq, data_smile, complex_graph, seq_len, smile_len)

            total_scores = F.softmax(out['logits_c'], 1).to('cpu').numpy()
            total_predictions = np.argmax(total_scores, axis=1)
            total_scores = total_scores[:, 1]
            id.extend(complex_graph.id)
            #T_New = torch.cat((T, New.cpu()), 0)
            Y.extend(complex_graph.y.long().to('cpu'))
            P.extend(total_predictions)
            S.extend(total_scores)
    return Y,P,S,attn,attn_prot


def predicting_record(model, device, loader, epoch):
    Y, P, S, id = [], [], [], []
    model.eval()

    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():

        for data in loader:
            data_seq = [torch.tensor(data).to(device) for data in data[0]]
            data_smile = [torch.tensor(data).to(device) for data in data[1]]
            complex_graph = data[2].to(device)
            seq_len = [torch.tensor(data).to(device) for data in data[3]]
            smile_len = [torch.tensor(data).to(device) for data in data[4]]
            out, attn, attn_prot = model(data_seq, data_smile, complex_graph, seq_len, smile_len)

            # 新增这一行：记录图5定量实验需要的值
            record_reliability(
                out=out,
                y=complex_graph.y.to(device),
                sample_ids=complex_graph.id,
                batch_idx=epoch,
                log_file='Davis_Alignment_reliability_test.csv'
            )

            total_scores = F.softmax(out['logits_c'], 1).to('cpu').numpy()
            total_predictions = np.argmax(total_scores, axis=1)
            total_scores = total_scores[:, 1]
            id.extend(complex_graph.id)
            #T_New = torch.cat((T, New.cpu()), 0)
            Y.extend(complex_graph.y.long().to('cpu'))
            P.extend(total_predictions)
            S.extend(total_scores)
    return Y,P,S,attn,attn_prot

# prepare the protein and drug pairs
def collate(data_list):

    seq = []
    smile = []
    tar_len = []
    smile_len = []

    for data in data_list:
        # 假设你要提取的内容是 `x` 或其他数据字段，而不是直接用索引
        seq.append(data[0])
        smile.append(data[1])
        tar_len.append(data[3])
        smile_len.append(data[4])

    batchGraph = Batch.from_data_list([data[2] for data in data_list])


    return seq, smile, batchGraph, tar_len, smile_len

def record_reliability(out, y, sample_ids, batch_idx, log_file='reliability_test.csv', eps=0.1):
    """
    记录图5定量实验需要的值：
    w: router权重
    dist: 根据单模态误差得到的可靠性目标
    delta_w / delta_dist: 用于后续相关性分析
    """

    # 第一次写入时加表头
    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'batch_idx', 'sample_idx', 'sample_id', 'label',
                'w_text', 'w_graph',
                'dist_text', 'dist_graph',
                'text_error', 'graph_error',
                'delta_w', 'delta_dist',
                'router_select', 'oracle_select'
            ])

    y = y.view(-1, 1).float()

    # router 输出权重
    w = out['channel_weight']   # [B, 2]

    # 单模态误差，和你训练时保持一致
    t_dist = eva_imp(out['logits_t_one'], y).view(-1)
    g_dist = eva_imp(out['logits_g_one'], y).view(-1)

    # error -> reliability target
    inv_t = 1.0 / (t_dist + eps)
    inv_g = 1.0 / (g_dist + eps)
    inv_sum = inv_t + inv_g

    dist_t = inv_t / inv_sum
    dist_g = inv_g / inv_sum

    # 差值，后面用于相关性分析
    delta_w = w[:, 1] - w[:, 0]
    delta_dist = dist_g - dist_t

    # router 选择哪个模态：0=text, 1=graph
    router_select = torch.argmax(w, dim=1)

    # oracle 选择哪个模态：误差小的那个
    oracle_select = torch.argmin(torch.stack([t_dist, g_dist], dim=1), dim=1)

    # 转 CPU
    w = w.detach().cpu()
    t_dist = t_dist.detach().cpu()
    g_dist = g_dist.detach().cpu()
    dist_t = dist_t.detach().cpu()
    dist_g = dist_g.detach().cpu()
    delta_w = delta_w.detach().cpu()
    delta_dist = delta_dist.detach().cpu()
    router_select = router_select.detach().cpu()
    oracle_select = oracle_select.detach().cpu()
    y_cpu = y.detach().cpu().view(-1).long()

    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)

        for i in range(w.shape[0]):
            writer.writerow([
                batch_idx,
                i,
                sample_ids[i] if sample_ids is not None else i,
                y_cpu[i].item(),

                w[i, 0].item(),
                w[i, 1].item(),

                dist_t[i].item(),
                dist_g[i].item(),

                t_dist[i].item(),
                g_dist[i].item(),

                delta_w[i].item(),
                delta_dist[i].item(),

                router_select[i].item(),
                oracle_select[i].item()
            ])