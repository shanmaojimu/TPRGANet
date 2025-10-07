import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from utils.tools import normalize
from abc import abstractmethod
from math import sqrt
from utils.init import glorot_weight_zero_bias
EOS = 1e-10


class BaseModel(nn.Module):
    """
    Base class for all models
    """

    @abstractmethod
    def forward(self, inputs):
        """
        Forward pass logic
        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class MBDSCA(nn.Module):
    """
    Spatial KGAT with Multi-branch Dynamic Strong-Connection Attention (MDSCA).
    - Multiple attention branches per layer (num_branches).
    - Each branch has its own temperature (learnable) for diversity.
    - Branch outputs are fused via learnable softmax weights per layer.
    - Annealed Top-K sparsification is preserved per branch.
    """
    def __init__(self,
                 n_nodes=22,
                 adj=None,
                 in_channels=64,
                 hidden_channels=64,
                 out_channels=64,
                 n_layers=2,
                 dropout=0.1,
                 device=0,
                 topk_start=10,
                 topk_end=3,
                 num_branches=3):
        super(MBDSCA, self).__init__()
        self.n_nodes = n_nodes
        self.n_layers = n_layers
        self.device = device
        self.dropout = dropout
        self.topk_start = topk_start
        self.topk_end = topk_end
        self.num_branches = num_branches

        # base adjacency (learnable)
        if adj is not None:
            adj_tensor = adj.clone().detach()
            adj_tensor = torch.clamp(adj_tensor, 0, 1)
            adj_tensor = adj_tensor * 0.01 + torch.eye(n_nodes) * 0.05
            if device != 'cpu' and torch.cuda.is_available():
                adj_tensor = adj_tensor.cuda(device)
            self.adj = nn.Parameter(adj_tensor, requires_grad=True)
        else:
            eye_tensor = torch.eye(n_nodes) * 0.05
            if device != 'cpu' and torch.cuda.is_available():
                eye_tensor = eye_tensor.cuda(device)
            self.adj = nn.Parameter(eye_tensor, requires_grad=True)

        # per-branch temperatures (learnable), one set shared across layers
        # shape: [num_branches]
        self.branch_temps = nn.Parameter(torch.ones(num_branches), requires_grad=True)

        # per-layer fusion logits over branches -> softmax to weights
        # shape: [n_layers, num_branches]
        self.fusion_logits = nn.Parameter(torch.zeros(n_layers, num_branches), requires_grad=True)

    def forward(self, x):
        batch_size, time_windows, n_nodes, channels = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, n_nodes, -1)

        adj_clamped = torch.clamp(self.adj, 0, 1)
        adj_normalized = self.normalize_adj(adj_clamped.to(x.device))

        outputs = []
        for b in range(batch_size):
            x_b = x[b]  # [n_nodes, features]

            for layer in range(self.n_layers):
                # annealed Top-K for this layer
                current_topk = int(
                    self.topk_start - (self.topk_start - self.topk_end) *
                    (layer / max(1, self.n_layers - 1))
                )

                # fusion weights for this layer: softmax over num_branches
                alpha = F.softmax(self.fusion_logits[layer], dim=-1)  # [num_branches]

                # accumulate per-branch outputs
                y_fused = 0.0
                for k in range(self.num_branches):
                    temp_k = torch.clamp(self.branch_temps[k], 0.1, 10.0)
                    att_k = self._compute_branch_attention(
                        x_b, adj_normalized.to(x_b.device), current_topk, temp_k
                    )  # [n_nodes, n_nodes]
                    y_k = torch.matmul(att_k, x_b)  # [n_nodes, features]
                    y_fused = y_fused + alpha[k] * y_k

                x_b = y_fused

                # residual + nonlinearity + dropout
                if layer > 0:
                    x_b = x_b + x[b]
                if layer < self.n_layers - 1:
                    x_b = F.relu(x_b)
                    x_b = F.dropout(x_b, p=self.dropout, training=self.training)

            outputs.append(x_b)

        x = torch.stack(outputs, dim=0)  # [B, n_nodes, features]
        return x, adj_normalized.to(x.device)

    @torch.no_grad()
    def _check_nan_inf(self, tensor, name="tensor"):
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            print(f"Warning: NaN/Inf detected in {name}")

    def _compute_branch_attention(self, x, adj, topk, temperature):
        """
        Per-branch spatial attention with annealed Top-K.
        Args:
            x: [n_nodes, features]
            adj: [n_nodes, n_nodes]
            topk: int
            temperature: scalar tensor
        Returns:
            attention: [n_nodes, n_nodes]
        """
        try:
            if torch.isnan(x).any() or torch.isinf(x).any():
                print("Warning: NaN or Inf in input features")
                return torch.eye(x.size(0), device=x.device)

            x_norm = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-6)
            sim = torch.mm(x_norm, x_norm.T)           # cosine similarity
            sim = sim * adj                            # mask by adjacency
            att = sim / temperature                    # temperature scaling
            att = att + torch.eye(att.size(0), device=att.device) * 0.1  # diagonal stabilization

            # annealed Top-K per row
            K = min(topk, att.size(1))
            if K <= 0:
                return F.softmax(att, dim=-1)

            topk_mask = torch.zeros_like(att)
            _, idx = torch.topk(att, K, dim=-1)
            topk_mask.scatter_(dim=-1, index=idx, value=1.0)
            att = att * topk_mask

            att = F.softmax(att, dim=-1)

            if torch.isnan(att).any() or torch.isinf(att).any():
                print("Warning: NaN or Inf detected in spatial attention, using identity")
                return torch.eye(x.size(0), device=x.device)

            return att
        except Exception as e:
            print(f"Error in spatial attention computation: {e}")
            return torch.eye(x.size(0), device=x.device)

    def normalize_adj(self, adj):
        try:
            device = adj.device
            eye = torch.eye(adj.size(0), device=device)
            adj = adj + eye
            adj = torch.clamp(adj, 1e-8, float('inf'))
            row_sum = adj.sum(1)
            row_sum = torch.clamp(row_sum, 1e-8, float('inf'))
            d_inv_sqrt = torch.pow(row_sum, -0.5)
            d_inv_sqrt = torch.clamp(d_inv_sqrt, 0, 100)
            d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
            result = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

            if torch.isnan(result).any() or torch.isinf(result).any():
                print("Warning: NaN or Inf in normalized adjacency matrix")
                return eye
            return result
        except Exception as e:
            print(f"Error in adjacency normalization: {e}")
            return torch.eye(adj.size(0), device=adj.device)



class SSTG(nn.Module):
    def __init__(self,
                 window_size,
                 in_channels,
                 hidden_channels=64,
                 out_channels=64,
                 n_layers=2,
                 dropout=0.1):
        super(SSTG, self).__init__()
        self.window_size = window_size
        self.n_layers = n_layers
        self.temperature = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        time_adj_tensor = 0.01 * torch.ones(window_size, window_size) + 0.05 * torch.eye(window_size)
        self.time_adj = nn.Parameter(time_adj_tensor, requires_grad=True)
        self.dropout = dropout

    def forward(self, x):
        batch_size, n_nodes, window_size, channels = x.shape
        x = x.permute(0, 2, 1, 3)
        x = x.contiguous().view(batch_size, window_size, -1)
        time_adj_clamped = torch.clamp(self.time_adj, 0, 1)
        time_adj_norm = self.normalize_adj(time_adj_clamped.to(x.device))
        x_orig = x.clone()

        for i in range(self.n_layers):
            attention = self.compute_time_attention_stable(x, time_adj_norm.to(x.device))
            x = torch.bmm(attention, x)
            if i > 0:
                x = x + x_orig
            if i < self.n_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        x = x.view(batch_size, window_size, n_nodes, channels)
        x = x.permute(0, 2, 1, 3)
        return x, self.time_adj.to(x.device)

    def compute_time_attention_stable(self, x, adj):
        try:
            batch_size, window_size, features = x.shape
            if torch.isnan(x).any() or torch.isinf(x).any():
                print("Warning: NaN or Inf in time input features")
                eye = torch.eye(window_size, device=x.device)
                return eye.unsqueeze(0).expand(batch_size, -1, -1)

            x_norm = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-6)
            attention = torch.bmm(x_norm, x_norm.transpose(-2, -1))
            adj_mask = adj.unsqueeze(0).expand(batch_size, -1, -1)
            attention = attention * adj_mask
            temperature = torch.clamp(self.temperature, 0.1, 10.0)
            attention = attention / temperature
            eye = torch.eye(window_size, device=attention.device)
            attention = attention + eye.unsqueeze(0).expand(batch_size, -1, -1) * 0.1
            attention = F.softmax(attention, dim=-1)

            if torch.isnan(attention).any() or torch.isinf(attention).any():
                print("Warning: NaN or Inf detected in time attention, using identity")
                eye = torch.eye(window_size, device=x.device)
                return eye.unsqueeze(0).expand(batch_size, -1, -1)

            return attention

        except Exception as e:
            print(f"Error in time attention computation: {e}")
            eye = torch.eye(window_size, device=x.device)
            return eye.unsqueeze(0).expand(batch_size, -1, -1)

    def normalize_adj(self, adj):
        try:
            device = adj.device
            adj = (adj + adj.T) / 2
            adj = F.relu(adj)
            eye = torch.eye(adj.size(0), device=device)
            adj = adj + eye
            adj = torch.clamp(adj, 1e-8, float('inf'))
            row_sum = adj.sum(1)
            row_sum = torch.clamp(row_sum, 1e-8, float('inf'))
            d_inv_sqrt = torch.pow(row_sum, -0.5)
            d_inv_sqrt = torch.clamp(d_inv_sqrt, 0, 100)
            d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
            result = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

            if torch.isnan(result).any() or torch.isinf(result).any():
                print("Warning: NaN or Inf in time normalized adjacency matrix")
                return eye

            return result
        except Exception as e:
            print(f"Error in time adjacency normalization: {e}")
            return torch.eye(adj.size(0), device=adj.device)


class Conv(nn.Module):
    def __init__(self, conv, activation=None, bn=None):
        nn.Module.__init__(self)
        self.conv = conv
        self.activation = activation
        if bn:
            self.conv.bias = None
        self.bn = bn

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class Residual1DConv(nn.Module):
    def __init__(self, channels, kernel_size=31):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU()
        )

    def forward(self, x):
        return x + self.conv(x)

class TemporalConvBlock(nn.Module):
    """Single TCN residual block with dilated convolution."""
    def __init__(self, in_channels, out_channels, kernel_size=31, dilation=1, dropout=0.1):
        super(TemporalConvBlock, self).__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        out = self.dropout(out)
        return out + self.residual(x)


class TCN(nn.Module):
    """Temporal Convolutional Network consisting of multiple dilated residual blocks."""
    def __init__(self, num_inputs, num_channels, kernel_size=31, dropout=0.1):
        super(TCN, self).__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation = 2 ** i
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(TemporalConvBlock(in_ch, out_ch, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TPRGANet(BaseModel):
    def __init__(self,
                 Adj,
                 in_chans,
                 n_classes,
                 time_window_num,
                 k_spatial,
                 k_time,
                 dropout,
                 input_time_length=125,
                 out_chans=64,
                 kernel_size=31,
                 slide_window=8,
                 sampling_rate=250,
                 device=0,
                 use_kgat=True,
                 pool_size=50,
                 pool_stride=15):
        super(TPRGANet, self).__init__()
        self.__dict__.update(locals())
        del self.self
        self.device = device
        self.time_window_num = time_window_num

        self.temporal_proj = Conv(
            nn.Conv1d(in_chans, out_chans, kernel_size=1, bias=False, groups=1),
            bn=nn.BatchNorm1d(out_chans), activation=None
        )
        self.tcn = TCN(
            num_inputs=out_chans,
            num_channels=[out_chans, out_chans],
            kernel_size=kernel_size,
            dropout=dropout
        )

        self.ge = MBDSCA(
            n_nodes=in_chans,
            adj=Adj,
            in_channels=in_chans,
            hidden_channels=out_chans,
            out_channels=out_chans,
            n_layers=k_spatial,
            dropout=dropout,
            device=device
        )
        self.spatial_proj = Conv(
            nn.Conv1d(in_chans, out_chans, kernel_size=1, bias=False, groups=1),
            bn=nn.BatchNorm1d(out_chans), activation=None
        )

        self.fusion = nn.Sequential(
            nn.Conv1d(2 * out_chans, out_chans, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_chans),
            nn.GELU()
        )

        self.ge_space = SSTG(
            window_size=slide_window,
            in_channels=out_chans,
            hidden_channels=out_chans,
            out_channels=out_chans,
            n_layers=k_time,
            dropout=dropout
        )

        self.downSampling = nn.AvgPool1d(int(sampling_rate // 2), int(sampling_rate // 2))
        self.dp = nn.Dropout(dropout)
        self.fc = nn.Linear(out_chans * (input_time_length * slide_window // (sampling_rate // 2)), n_classes)

    def forward(self, x, perturbed_adj=None):
        # reshape to [B, C, S, Tw]
        B, C, T = x.shape
        S = self.time_window_num
        Tw = T // S
        x_reshaped = x.view(B, C, S, Tw)

        x_tem = self.temporal_proj(x)     # [B, F, S*Tw]
        x_tem = self.tcn(x_tem)           # [B, F, S*Tw]

        x_sp_in = x_reshaped.permute(0, 2, 1, 3)  # [B, S, C, Tw]
        original_adj = None
        if perturbed_adj is not None and hasattr(self.ge, 'adj'):
            original_adj = self.ge.adj.data.clone()
            self.ge.adj.data = perturbed_adj.to(self.ge.adj.device)
        x_sp, node_weights = self.ge(x_sp_in)
        if original_adj is not None:
            self.ge.adj.data = original_adj
        x_sp = self.spatial_proj(x_sp)    # [B, F, S*Tw]

        x_fused = torch.cat([x_tem, x_sp], dim=1)   # [B, 2F, S*Tw]
        x_fused = self.fusion(x_fused)              # [B, F, S*Tw]

        x_fused = x_fused.view(B, x_fused.size(1), S, Tw)
        x_sstg, space_node_weights = self.ge_space(x_fused)
        x_sstg = x_sstg.contiguous().view(B, x_sstg.size(1), -1)

        x_out = self.downSampling(x_sstg)
        x_out = self.dp(x_out)
        features_before_fc = x_out.view(B, -1)

        if self.device != 'cpu' and torch.cuda.is_available():
            features_before_fc = features_before_fc.cuda(self.device)

        logits = self.fc(features_before_fc)
        return logits, features_before_fc, node_weights, space_node_weights

