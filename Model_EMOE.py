import torch.nn as nn
import torch_geometric.nn as geom_nn
from torch_geometric.nn import GATv2Conv, global_add_pool
from torch.nn import BatchNorm1d
from Data_process import *
from torch.nn.utils import parameters_to_vector

cuda_name = 'cuda:0'
USE_CUDA = torch.cuda.is_available()
device = torch.device(cuda_name if USE_CUDA else 'cpu')

def update_params(loss, model,update_lr):
    grads = torch.autograd.grad(loss, model.parameters())
    return parameters_to_vector(model.parameters()) - parameters_to_vector(grads) * update_lr

class Encoder(nn.Module):
    def __init__(self, inc, outc, pad1=15):
        super(Encoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv_in = nn.Conv1d(in_channels=inc, out_channels=outc, kernel_size=(pad1 * 2 + 1), stride=1, padding=pad1,
                                 bias=False)

        self.conv_out = nn.Conv1d(in_channels=outc, out_channels=outc, kernel_size=(pad1 * 2 + 1),
                                  stride=1,
                                  padding=pad1, bias=False)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.relu(x)
        x = self.conv_out(x)
        x = self.relu(x)
        return x

class MaskAttention(nn.Module):
    def __init__(self, input_dim, n_heads):
        super(MaskAttention, self).__init__()
        self.query = nn.Linear(input_dim, n_heads)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, masks):
        query = self.query(x).transpose(1, 2) # (B,heads,seq_len)
        value = x # (B,seq_len,hidden_dim)

        minus_inf = -9e15 * torch.ones_like(query) # (B,heads,seq_len)
        e = torch.where(masks > 0.5, query, minus_inf)  # (B,heads,seq_len)
        a = self.softmax(e) # (B,heads,seq_len)

        out = torch.matmul(a, value) # (B,heads,seq_len) * (B,seq_len,hidden_dim) = (B,heads,hidden_dim)
        out = torch.mean(out, dim=1).squeeze(1) # (B,hidden_dim)
        return out, a

# GAT based model
class PS_MoE(torch.nn.Module):
    def __init__(self, n_output=2, dropout=0.2):
        super(PS_MoE, self).__init__()

        print('PS_MoE loading ...')
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.embeding_dim = 256
        self.is_bidirectional = True
        self.bilstm_layers = 2
        self.lstm_dim = 64
        self.n_heads = 4
        self.n_output = n_output

        self.encoder_protein_LM = Encoder(1024, 128, pad1=5)
        self.encoder_drug = Encoder(768, 128, pad1=5)

        self.out_attentions3 = MaskAttention(128, 4)
        self.out_attentions2 = MaskAttention(128, 4)
        self.out_attentions = MaskAttention(128, 4)

        self.proj1_t = nn.Linear(384, 384)
        self.proj2_t = nn.Linear(384, 384)
        self.out_layer_t = nn.Linear(384, 2)
        self.out_layer_t_one = nn.Linear(384, 1)
        self.proj1_g = nn.Linear(384, 384)
        self.proj2_g = nn.Linear(384, 384)
        self.out_layer_g = nn.Linear(384, 2)
        self.out_layer_g_one = nn.Linear(384, 1)
        self.proj1_c = nn.Linear(384, 384)
        self.proj2_c = nn.Linear(384, 384)
        self.out_layer_c = nn.Linear(384, 2)
        self.out_layer_c_one = nn.Linear(384, 1)

        self.layer2 = self.build_layer( node_f=64, node_f_hidden=64, node_f_out=64,
                                        edge_f=64, edge_f_hidden=64, edge_f_out=64,
                                        glob_f=384, glob_f_hidden=384, glob_f_out=384,
                                        residuals=False, dropout=0
                                        )

        self.NodeTransform = FeatureTransformMLP(1148, 256, 64, dropout=0.5)

        self.layer1 = self.build_layer(node_f=64, node_f_hidden=64, node_f_out=64,
                                       edge_f=20, edge_f_hidden=64, edge_f_out=64,
                                       glob_f=384, glob_f_hidden=384, glob_f_out=384,
                                       residuals=False, dropout=0
                                       )

        self.node_bn1 = BatchNorm1d(64)
        self.edge_bn1 = BatchNorm1d(64)
        self.u_bn1 = BatchNorm1d(384)
        self.dropout_layer = nn.Dropout(0.5)

        self.T_fc1 = nn.Linear(384, 128)
        self.T_fc2 = nn.Linear(128, 2)

        self.G_fc1 = nn.Linear(384, 128)
        self.G_fc2 = nn.Linear(128, 2)

        self.G2_fc1 = nn.Linear(128, 128)
        self.G2_fc2 = nn.Linear(128, 2)

        self.F_fc1 = nn.Linear(768, 128)
        self.F_fc2 = nn.Linear(128, 2)

        self.fc1 = nn.Linear(384, 128)
        self.fc2 = nn.Linear(128, 2)

        self.Router = router(768, 2, 0.1)

    def build_layer(self,
                    node_f, node_f_hidden, node_f_out,
                    edge_f, edge_f_hidden, edge_f_out,
                    glob_f, glob_f_hidden, glob_f_out,
                    residuals, dropout):
        return geom_nn.MetaLayer(
            edge_model=EdgeModel(node_f, edge_f, edge_f_hidden, edge_f_out, residuals=residuals, dropout=dropout),
            node_model=NodeModel(node_f, edge_f_out, node_f_hidden, node_f_out, residuals=residuals, dropout=dropout),
            global_model=GlobalModel(node_f_out, glob_f, glob_f_hidden, glob_f_out, dropout=dropout)
        )

    def generate_mask(self, adj, adj_sizes, n_heads):
        out = torch.ones(adj.shape[0], adj.shape[1])
        max_size = adj.shape[1]
        if isinstance(adj_sizes, int):
            out[0, adj_sizes:max_size] = 0
        else:
            for e_id, drug_len in enumerate(adj_sizes):
                out[e_id, drug_len: max_size] = 0
        out = out.unsqueeze(1).expand(-1, n_heads, -1)
        return out.cuda(device=adj.device)

    def forward(self, seq, smile, graph, seq_len, smile_len):

        # 文本
        molecule_LM = torch.stack(smile, dim=0)
        protein_LM = torch.stack(seq, dim=0)

        proteins_acids_LM = self.encoder_protein_LM(protein_LM.permute(0, 2, 1)).permute(0, 2, 1)  # .mean(dim=1) 16,1200,256
        molecule_smiles_LM = self.encoder_drug(molecule_LM.permute(0, 2, 1)).permute(0, 2, 1)  # .mean(dim=1) 16,100,256

        smiles_mask = self.generate_mask(molecule_smiles_LM, smile_len, self.n_heads)  # B * head* seq len
        protein_mask = self.generate_mask(proteins_acids_LM, seq_len, self.n_heads)  # B * head * tar_len
        smiles_out, smile_attn = self.out_attentions3(molecule_smiles_LM, smiles_mask)  # B * lstm_dim*2
        protein_out, prot_attn = self.out_attentions2(proteins_acids_LM, protein_mask)  # B * (lstm_dim *2)

        out_cat = torch.cat((molecule_smiles_LM, proteins_acids_LM), dim=1)  # B * head * lstm_dim *2
        out_masks = torch.cat((smiles_mask, protein_mask), dim=2)  # B * tar_len+seq_len * (lstm_dim *2)
        out_cat, out_attn = self.out_attentions(out_cat, out_masks)
        out = torch.cat([smiles_out, protein_out, out_cat], dim=-1)  # B * (rnn*2 *3)

        # 图
        edge_index = graph.edge_index_prot
        x = self.NodeTransform(graph.x)
        x, edge_attr, u = self.layer1(x, edge_index, graph.edge_attr_prot, u=graph.lig_emb, batch=graph.batch)
        x = self.node_bn1(x)
        edge_attr = self.edge_bn1(edge_attr)
        u = self.u_bn1(u)

        x, _, u = self.layer2(x, edge_index, edge_attr, u=u, batch=graph.batch)
        u = self.dropout_layer(u)

        # EMOE
        m_i = torch.cat((out, u), 1)
        m_w = self.Router(m_i)

        out_proj = self.proj2_t(
            F.dropout(F.relu(self.proj1_t(out), inplace=True), p=0.5,
                      training=self.training))
        out_proj += out
        logits_out = self.out_layer_t(out_proj)
        logits_out_one = self.out_layer_t_one(out_proj)

        u_proj = self.proj2_g(
            F.dropout(F.relu(self.proj1_g(u), inplace=True), p=0.5,
                      training=self.training))
        u_proj += u
        logits_u = self.out_layer_g(u_proj)
        logits_u_one = self.out_layer_g_one(u_proj)

        for i in range(m_w.shape[0]):
            c_f = out[i] * m_w[i][0] + u[i] * m_w[i][1]
            if i == 0:
                c_fusion = c_f.unsqueeze(0)
            else:
                c_fusion = torch.cat([c_fusion, c_f.unsqueeze(0)], dim=0)

        c_proj = self.proj2_c(
            F.dropout(F.relu(self.proj1_c(c_fusion), inplace=True), p=0.5,
                      training=self.training))
        c_proj += c_fusion
        logits_c = self.out_layer_c(c_proj)
        logits_c_one = self.out_layer_c_one(c_proj)

        res = {
            'logits_c': logits_c,
            'logits_t': logits_out,
            'logits_g': logits_u,
            'channel_weight': m_w,
            'c_proj': c_proj,
            't_proj': out_proj,
            'g_proj': u_proj,
            'c_fea': c_fusion,
            'logits_c_one': logits_c_one,
            'logits_t_one': logits_out_one,
            'logits_g_one': logits_u_one,
        }

        return res, torch.mean(smile_attn, dim=1).squeeze(), torch.mean(prot_attn, dim=1).squeeze()

class FeatureTransformMLP(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, out_dim, dropout):
        super(FeatureTransformMLP, self).__init__()
        self.dropout_layer = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim))

    def forward(self, node_features):
        x = self.mlp(node_features)
        return self.dropout_layer(x)


class EdgeModel(torch.nn.Module):
    def __init__(self, n_node_f, n_edge_f, hidden_dim, out_dim, residuals, dropout):
        super().__init__()
        self.residuals = residuals
        self.dropout_layer = nn.Dropout(dropout)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * n_node_f + n_edge_f, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, src, dest, edge_attr, u, batch):
        out = torch.cat([src, dest, edge_attr], 1)
        out = self.dropout_layer(out)
        out = self.edge_mlp(out)
        if self.residuals:
            out = out + edge_attr
        return out


class NodeModel(torch.nn.Module):
    def __init__(self, n_node_f, n_edge_f, hidden_dim, out_dim, residuals, dropout):
        super(NodeModel, self).__init__()
        self.residuals = residuals
        self.heads = 4

        self.conv = GATv2Conv(n_node_f, int(out_dim/self.heads), edge_dim=n_edge_f, heads=self.heads, dropout=dropout)

    def forward(self, x, edge_index, edge_attr, u, batch):
        out = F.relu(self.conv(x, edge_index, edge_attr))
        if self.residuals:
            out = out + x
        return out


class GlobalModel(torch.nn.Module):
    def __init__(self, n_node_f, glob_f_in, glob_f_hidden, glob_f_out, dropout):
        super().__init__()
        self.dropout_layer = nn.Dropout(dropout)
        self.global_mlp = nn.Sequential(
            nn.Linear(n_node_f + glob_f_in, glob_f_hidden),
            nn.ReLU(),
            nn.Linear(glob_f_hidden, glob_f_out))

    def forward(self, x, edge_index, edge_attr, u, batch):
        out = torch.cat([u, global_add_pool(x, batch=batch)], dim=1)
        out = self.dropout_layer(out)
        return self.global_mlp(out)


class router(nn.Module):
    def __init__(self, dim, channel_num, t):
        super().__init__()
        self.l1 = nn.Linear(dim, int(dim / 8))
        self.l2 = nn.Linear(int(dim / 8), channel_num)
        self.t = t

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.l2(F.relu(F.normalize(self.l1(x), p=2, dim=1))) / self.t
        output = torch.softmax(x, dim=1)
        return output



