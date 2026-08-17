import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from common.abstract_recommender import GeneralRecommender

class GRAMREC(GeneralRecommender):
    def __init__(self, config, dataset):
        super(GRAMREC, self).__init__(config, dataset)
        self.setup_parameters(config, dataset)

    def setup_parameters(self, config, dataset):
        # Initialize parameters and embeddings
        self.n_nodes = self.n_users + self.n_items
        self.dim_latent = config['embedding_size']
        self.n_layers = config['n_mm_layers']
        self.knn_k = config['knn_k']
        self.mm_image_weight = config['mm_image_weight']
        self.dropout = config['dropout']
        self.n_ui_layers = config['n_ui_layers']
        self.dataset = dataset
        self.v_rep, self.t_rep, self.id_rep = None, None, None
        self.v_preference, self.t_preference, self.id_preference = None, None, None
        self.mm_adj = None
        self.gamma = config['gamma']
        self.lambda_volume = config['lambda_volume']
        self.tau_intra = config['tau_intra']
        self.tau_inter = config['tau_inter']

        # Load user graph dictionary
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])

        # Create user and item embeddings
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)

        self.theta = nn.Parameter(nn.init.xavier_normal_(
            torch.tensor(np.random.randn(self.n_users, 3, 1), dtype=torch.float32, requires_grad=True)))

        # Load or generate mm_adj
        mm_adj_file = os.path.join(dataset_path, f'mm_adj_{self.knn_k}_{int(10 * self.mm_image_weight)}.pt')
        if os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file)
        else:
            if self.v_feat is not None:
                indices, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
                self.mm_adj = image_adj
            if self.t_feat is not None:
                indices, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
                self.mm_adj = text_adj
            if self.v_feat is not None and self.t_feat is not None:
                self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
                del text_adj
                del image_adj
            torch.save(self.mm_adj, mm_adj_file)
        self.mm_adj = self.mm_adj.to(self.device)

        # Construct interaction edge
        self.train_interactions = dataset.inter_matrix(form='coo').astype(np.float32) # ([19445, 7050])
        self.edge_indices, self.edge_values = self.get_edge_info() # ([2, 118551]) ([118551])
        self.edge_indices, self.edge_values = self.edge_indices.to(self.device), self.edge_values.to(self.device)

        # Create normalized adjacency matrices
        self.norm_adj = self.get_norm_adj_mat().coalesce().to(self.device) # ([26495, 26495])
        self.masked_adj = self.norm_adj # ([26495, 26495])

        # Create GCN layers for different modalities
        self.create_gcn_layers()

        # Create user graph and result embeddings
        # self.user_graph = User_Graph_sample(self.n_users, self.dim_latent)
        self.result_embed = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(self.n_nodes, self.dim_latent)))
        ).to(self.device)

    def create_gcn_layers(self):
        if self.v_feat is not None:
            self.v_gcn = GCNLayer(self.n_users, self.n_items, n_ui_layers=self.n_ui_layers, dim_latent=64,
                                  device=self.device, features=self.v_feat)
        if self.t_feat is not None:
            self.t_gcn = GCNLayer(self.n_users, self.n_items, n_ui_layers=self.n_ui_layers, dim_latent=64,
                                  device=self.device, features=self.t_feat)

        self.id_feat = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(self.n_items, self.dim_latent), dtype=torch.float32,
                                                requires_grad=True), gain=1).to(self.device))
        self.id_gcn = GCNLayer(self.n_users, self.n_items, n_ui_layers=self.n_ui_layers, dim_latent=64,
                               device=self.device, features=self.id_feat)

    def get_edge_info(self):
        rows = torch.from_numpy(self.train_interactions.row)
        cols = torch.from_numpy(self.train_interactions.col)
        edges = torch.stack([rows, cols]).type(torch.LongTensor)
        values = self._normalize_adj_m(edges, torch.Size((self.n_users, self.n_items)))
        return edges, values

    def _normalize_adj_m(self, indices, adj_size):
        adj = torch.sparse_coo_tensor(indices, torch.ones_like(indices[0], dtype=torch.float32), adj_size).coalesce()
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return values

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.train_interactions
        inter_M_t = self.train_interactions.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        # A._update(data_dict)
        for (row, col), value in data_dict.items():
            A[row, col] = value
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse_coo_tensor(i, data, torch.Size((self.n_nodes, self.n_nodes))).coalesce()

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def pre_epoch_processing(self):
        # Keep epoch-wise random user-neighbor sampling by default for stable upper-bound performance.
        if self.dropout <= .0:
            self.masked_adj = self.norm_adj
            return
        degree_len = int(self.edge_values.size(0) * (1. - self.dropout))
        degree_idx = torch.multinomial(self.edge_values, degree_len)
        # random sample
        keep_indices = self.edge_indices[:, degree_idx]
        # norm values
        keep_values = self._normalize_adj_m(keep_indices, torch.Size((self.n_users, self.n_items)))
        all_values = torch.cat((keep_values, keep_values))
        # update keep_indices to users/items+self.n_users
        keep_indices[1] += self.n_users
        all_indices = torch.cat((keep_indices, torch.flip(keep_indices, [0])), 1)
        self.masked_adj = torch.sparse_coo_tensor(all_indices, all_values, self.norm_adj.shape).coalesce().to(self.device)

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        return np.column_stack((rows, cols))

    def forward(self): # interaction ([3, 2048])
        id_rep, id_preference = self.id_gcn(self.id_feat, self.masked_adj) # ([26495, 64])
        id_rep_data = id_rep # ([26495, 64])

        representation = id_rep_data # ([26495, 64])

        if self.v_feat is not None:
            self.v_rep, self.v_preference = self.v_gcn(self.v_feat, self.masked_adj) #self.id_feat ([7050, 64]) self.v_feat ([7050, 4096])
            # Eq.6
            representation = torch.cat((id_rep_data, self.v_rep), dim=1) # ([26495, 128])

        if self.t_feat is not None:
            self.t_rep, self.t_preference = self.t_gcn(self.t_feat, self.masked_adj) # self.t_feat ([7050, 384])
            representation = torch.cat((id_rep_data, self.t_rep) if representation is None
                                       else (id_rep_data, self.v_rep, self.t_rep), dim=1) # ([26495, 192])

        item_rep = representation[self.n_users:] # ([7050, 192])
        # Eq.13
        h_i = item_rep # A_i^{L_i} ([7050, 192])
        for i in range(self.n_layers):
            # Eq.12 加权聚合， mm_adj 是 item-item KNN 图
            h_i = torch.sparse.mm(self.mm_adj, h_i) # ([7050, 192])
        item_rep = item_rep + h_i # E_f(I) + A_i^{L_i} ([7050, 192])
        
        user_rep = representation[:self.n_users]
        id_rep = user_rep[:, :self.dim_latent] # ([19445, 64])
        v_rep = user_rep[:, self.dim_latent: 2 * self.dim_latent] # ([19445, 64])
        t_rep = user_rep[:, 2 * self.dim_latent:] # ([19445, 64])
        _att2 = F.softmax(self.theta, dim=1) # ([19445, 3, 1])
        user_rep = torch.cat((_att2[:, 0, :] * id_rep, _att2[:, 1, :] * v_rep, _att2[:, 2, :] * t_rep), dim=1) # ([19445, 192])
        # Eq.14
        self.result_embed = torch.cat((user_rep, item_rep), dim=0) # ([26495, 192])

        # return user_rep, item_rep # 换成bpr的时候记得换
        return F.normalize(user_rep, dim=-1), F.normalize(item_rep, dim=-1)

    ##### LOSS
    @staticmethod
    def alignment(x, y, alpha=2):
        return (x - y).norm(p=2, dim=1).pow(alpha).mean()
    
    @staticmethod
    def uniformity(x, t=2):
        x = F.normalize(x, dim=-1)
        sim = x @ x.t()
        dist2 = 2 - 2 * sim

        # 去掉对角线，避免 self-distance
        mask = ~torch.eye(x.size(0), dtype=torch.bool, device=x.device)
        dist2 = dist2[mask]

        return torch.exp(-t * dist2).mean().log()

    def gram_loss(self, x, y ,z): 
        loss_intra, loss_inter = pmrl_gram_loss(x, y, z, tau_intra=self.tau_intra, tau_inter=self.tau_inter)
        return (loss_intra + loss_inter) / 2
    
    def bpr_loss(self, users, pos_items, neg_items):
        # users=h_u  pos_items=h_i或h_i^m  neg_items=h_j或h_j^m
        # h_u^T * h_i就是预测评分呐，和正样本点积的结果肯定高，和负样本点积的结果肯定低呐，二者相减越大越好呐，因为前面有个负号（看return上面那句）
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        return mf_loss

    def calculate_loss(self, interaction):
        users = interaction[0] # ([2048])
        pos_items = interaction[1] # ([2048])
        neg_items = interaction[2] # ([2048])
        
        users_emb, items_emb = self.forward() # ([19445, 192]) ([7050, 192])
        batch_users_emb = users_emb[users] # ([2048, 192])
        batch_pos_emb = items_emb[pos_items] # ([2048, 192])
        batch_neg_emb = items_emb[neg_items] # ([2048, 192])

        id_i_emb = batch_neg_emb[:, :self.dim_latent]
        v_i_emb = batch_neg_emb[:, self.dim_latent: 2 * self.dim_latent]
        t_i_emb = batch_neg_emb[:, 2 * self.dim_latent:]

        # mf_loss = self.bpr_loss(batch_users_emb, batch_pos_emb, batch_neg_emb)

        # ---------- 原始 ID BPR ----------
        align = self.alignment(batch_users_emb, batch_pos_emb)
        
        # uniform = self.gamma * (self.uniformity(batch_users_emb) + self.uniformity(batch_neg_emb)) / 2 # 全部都搞一块的
        uniform = self.gamma * ( # nb 说明用户的向量别拆开（拆开就降），物品的向量得拆开（拆开就涨！各模态分别均匀性）（论文可写）
            self.uniformity(batch_users_emb) + 
            # self.uniformity(batch_neg_emb)) / 2
            self.uniformity(v_i_emb) + self.uniformity(t_i_emb) + self.uniformity(id_i_emb)) / 2 
        
        volume_align = self.lambda_volume * self.gram_loss(id_i_emb, v_i_emb, t_i_emb)
    
        mf_loss = align + uniform + volume_align
        # mf_loss = mf_loss + volume_align
        
        # Eq.16
        return mf_loss

    def full_sort_predict(self, interaction):
        user_tensor = self.result_embed[:self.n_users]
        item_tensor = self.result_embed[self.n_users:]

        temp_user_tensor = user_tensor[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        return score_matrix

class GCNLayer(torch.nn.Module):
    def __init__(self, num_user, num_item, n_ui_layers, dim_latent=None, device=None, features=None):
        super(GCNLayer, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.dim_feat = features.size(1)
        self.dim_latent = dim_latent
        self.n_ui_layers = n_ui_layers
        self.device = device
        self.preference = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(num_user, self.dim_latent), dtype=torch.float32,
                                                requires_grad=True), gain=1).to(self.device))
        self.MLP = nn.Linear(self.dim_feat, 4 * self.dim_latent)
        self.MLP_1 = nn.Linear(4 * self.dim_latent, self.dim_latent)

    def forward(self, features, adj):
        # Eq.2 
        temp_features = self.MLP_1(F.leaky_relu(self.MLP(features))) if self.dim_latent else features # ([7050, 64])
        x = torch.cat((self.preference, temp_features), dim=0) # ([26495. 64])
        x = F.normalize(x, dim=1) # ([26495. 64])
        ego_embeddings = x # ([26495. 64])
        all_embeddings = [ego_embeddings] # ([26495. 64])

        # Eq.4
        for layer_idx in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings) # Ã E^(l−1)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        # Eq.5
        representation = torch.stack(all_embeddings, dim=0).sum(dim=0)

        return representation, self.preference


def pmrl_gram_loss(x, y, z, tau_intra, tau_inter):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    z = F.normalize(z, dim=-1)
    stacked = torch.stack([x, y, z], dim=-1) # [2048, 64, 3]
    gram = torch.matmul(stacked.transpose(1, 2), stacked) # [2048, 3, 3]
    eigvals, eigvecs = torch.linalg.eigh(gram) # [2048, 3]  [2048, 3, 3]
    eigvals = torch.clamp(eigvals, min=1e-12) # [2048, 3]
    singular_values = torch.sqrt(eigvals).flip(dims=[1]) # sort to descending # [2048, 3]

    logits_intra = singular_values / tau_intra # [2048, 3]
    target_intra = torch.zeros(logits_intra.size(0), dtype=torch.long, device=logits_intra.device) # ([2048])
    loss_intra = F.cross_entropy(logits_intra, target_intra)

    principal_right = eigvecs[:, :, -1] # top singular vector in modal space # [2048, 3]
    principal = torch.matmul(stacked, principal_right.unsqueeze(-1)).squeeze(-1) # [2048, 64]
    principal = F.normalize(principal, dim=1) # [2048, 64]
    if principal.size(0) <= 1:
        loss_inter = torch.zeros((), device=principal.device, dtype=principal.dtype)
    else:
        logits_inter = torch.matmul(principal, principal.t()) / tau_inter # [2048, 2048]
        target_inter = torch.arange(principal.size(0), device=principal.device) # [2048]
        loss_inter = F.cross_entropy(logits_inter, target_inter)

    return loss_intra, loss_inter