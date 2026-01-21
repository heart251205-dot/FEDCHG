import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Tanh
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool, global_add_pool


# =============================================================================
# 1. Graph Autoencoder (Section 4.2) - Phase 1
# =============================================================================

class FedCHG_GAE(nn.Module):

    def __init__(self, in_dim=7, hid_dim=64, dropout=0.5):
        super(FedCHG_GAE, self).__init__()
        self.dropout_ratio = dropout



        self.enc_lin = Linear(in_dim, hid_dim)


        self.enc_conv1 = GCNConv(hid_dim, hid_dim)
        self.enc_conv2 = GCNConv(hid_dim, hid_dim)
        self.enc_conv3 = GCNConv(hid_dim, hid_dim)


        self.enc_post = Linear(hid_dim, hid_dim)


        self.dec_conv1 = GCNConv(hid_dim, hid_dim)
        self.dec_conv2 = GCNConv(hid_dim, hid_dim)

        self.dec_conv3 = GCNConv(hid_dim, in_dim)


        self.dec_post = Linear(in_dim, in_dim)

    def encode(self, x, edge_index):


        x = torch.tanh(self.enc_lin(x))


        x = torch.tanh(self.enc_conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)

        x = torch.tanh(self.enc_conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)

        h_L = torch.tanh(self.enc_conv3(x, edge_index))  # H^(L)


        x_stru = self.enc_post(h_L)
        x_stru = F.dropout(x_stru, p=self.dropout_ratio, training=self.training)

        return x_stru, h_L

    def decode(self, z, edge_index):


        x = torch.tanh(self.dec_conv1(z, edge_index))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)

        x = torch.tanh(self.dec_conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)

        h_prime_L = self.dec_conv3(x, edge_index)  # No non-linearity on final output usually for regression


        se_prime = self.dec_post(h_prime_L)

        return se_prime, h_prime_L  # Return h_prime_L for in-domain constraint loss

    def forward(self, data):

        z, h_L = self.encode(data.se_stru_0, data.edge_index)
        recon, h_prime_L = self.decode(z, data.edge_index)
        return recon, z, h_L, h_prime_L


# =============================================================================
# 2. Decoupled Models (Section 4.3) - Phase 2
# =============================================================================

class FeatureModel(nn.Module):


    def __init__(self, input_dim, hidden_dim=64, dropout=0.5):
        super(FeatureModel, self).__init__()
        self.dropout_ratio = dropout


        self.mlp1 = Sequential(
            Linear(input_dim, hidden_dim),
            BatchNorm1d(hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            BatchNorm1d(hidden_dim),
            ReLU()
        )
        self.conv1 = GINConv(self.mlp1, train_eps=True)  # train_eps=True enables learning epsilon in Eq. 34


        self.mlp2 = Sequential(
            Linear(hidden_dim, hidden_dim),
            BatchNorm1d(hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            BatchNorm1d(hidden_dim),
            ReLU()
        )
        self.conv2 = GINConv(self.mlp2, train_eps=True)


        self.mlp3 = Sequential(
            Linear(hidden_dim, hidden_dim),
            BatchNorm1d(hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            BatchNorm1d(hidden_dim),
            ReLU()
        )
        self.conv3 = GINConv(self.mlp3, train_eps=True)

    def forward(self, x, edge_index, return_layers=False):

        xs = []

        x = self.conv1(x, edge_index)
        xs.append(x)
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)

        x = self.conv2(x, edge_index)
        xs.append(x)
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)

        x = self.conv3(x, edge_index)
        xs.append(x)
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)

        if return_layers:
            return xs
        return x


class StructuralModel(nn.Module):


    def __init__(self, input_dim=7, hidden_dim=64, dropout=0.5):
        super(StructuralModel, self).__init__()
        self.dropout_ratio = dropout


        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, return_layers=False):

        xs = []

        x = F.relu(self.conv1(x, edge_index))
        xs.append(x)
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)

        x = F.relu(self.conv2(x, edge_index))
        xs.append(x)
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)

        x = F.relu(self.conv3(x, edge_index))
        xs.append(x)
        # Note: No dropout on final layer usually, but consistency implies similar structure

        if return_layers:
            return xs
        return x


# =============================================================================
# 3. Joint Classifier (Section 4.3.2 & Eq. 36)
# =============================================================================

class FedCHG_Classifier(nn.Module):


    def __init__(self, feat_model, stru_model, hidden_dim=64, num_classes=2, fusion_type='late'):
        super(FedCHG_Classifier, self).__init__()
        self.feat_model = feat_model
        self.stru_model = stru_model
        self.fusion_type = fusion_type


        self.lin1 = Linear(hidden_dim * 2, hidden_dim)
        self.lin2 = Linear(hidden_dim, num_classes)

    def forward(self, data):

        x_feat = data.x


        if hasattr(data, 'se_stru_prime'):
            x_stru = data.se_stru_prime
        elif hasattr(data, 'se_stru_0'):
            x_stru = data.se_stru_0
        else:
            raise ValueError("No structural features found in data object")

        edge_index = data.edge_index
        batch = data.batch




        h_feat = self.feat_model(x_feat, edge_index)
        h_stru = self.stru_model(x_stru, edge_index)


        g_feat = global_mean_pool(h_feat, batch)
        g_stru = global_mean_pool(h_stru, batch)


        g_final = torch.cat([g_feat, g_stru], dim=1)


        x = F.relu(self.lin1(g_final))
        x = F.dropout(x, p=0.5, training=self.training)
        out = self.lin2(x)

        return F.log_softmax(out, dim=1)  # Eq. 36 uses log(p)