import torch
import numpy as np
import random
from typing import Tuple, Optional


class AdjacencyPerturbation:
    """
    邻接矩阵扰动数据增强类
    支持多种扰动策略：随机添加/删除边、调整边权、对称扰动等
    """
    
    def __init__(self, 
                 probability: float = 0.5, # 扰动概率
                 perturbation_type: str = "mixed",  # "add_delete", "weight_adjust", "symmetric", "mixed"
                 edge_change_ratio: float = 0.1,    # 边变化比例
                 weight_noise_std: float = 0.1,     # 权重噪声标准差
                 min_edge_weight: float = 0.01,     # 最小边权重
                 max_edge_weight: float = 1.0,      # 最大边权重
                 preserve_self_loops: bool = True,  # 保持自环
                 ensure_connectivity: bool = True,  # 确保连通性
                 device: str = 'cuda'):
        
        self.probability = probability
        self.perturbation_type = perturbation_type
        self.edge_change_ratio = edge_change_ratio
        self.weight_noise_std = weight_noise_std
        self.min_edge_weight = min_edge_weight
        self.max_edge_weight = max_edge_weight
        self.preserve_self_loops = preserve_self_loops
        self.ensure_connectivity = ensure_connectivity
        self.device = device
    
    def __call__(self, adj_matrix: torch.Tensor, apply_perturbation: bool = True) -> torch.Tensor:
        """
        对邻接矩阵应用扰动
        
        Args:
            adj_matrix: 原始邻接矩阵 [n_nodes, n_nodes]
            apply_perturbation: 是否应用扰动（用于控制训练时的随机性）
            
        Returns:
            扰动后的邻接矩阵
        """
        if not apply_perturbation or random.random() > self.probability:
            return adj_matrix.clone()
        
        
        perturbed_adj = adj_matrix.clone()
        n_nodes = perturbed_adj.size(0)
        
        
        if self.perturbation_type == "add_delete":
            perturbed_adj = self._add_delete_edges(perturbed_adj)
        elif self.perturbation_type == "weight_adjust":
            perturbed_adj = self._adjust_edge_weights(perturbed_adj)
        elif self.perturbation_type == "symmetric":
            perturbed_adj = self._symmetric_perturbation(perturbed_adj)
        elif self.perturbation_type == "mixed":
            
            strategies = ["add_delete", "weight_adjust", "symmetric"]
            chosen_strategy = random.choice(strategies)
            if chosen_strategy == "add_delete":
                perturbed_adj = self._add_delete_edges(perturbed_adj)
            elif chosen_strategy == "weight_adjust":
                perturbed_adj = self._adjust_edge_weights(perturbed_adj)
            else:
                perturbed_adj = self._symmetric_perturbation(perturbed_adj)
        
        
        perturbed_adj = self._ensure_symmetry(perturbed_adj) # 确保对称性
        
        
        if self.ensure_connectivity: #检查扰动后图是否仍然是连通图，避免模型在非连通图上性能坍塌
            perturbed_adj = self._ensure_connectivity_preservation(perturbed_adj, adj_matrix)
        
        
        perturbed_adj = torch.clamp(perturbed_adj, self.min_edge_weight, self.max_edge_weight) #裁剪边权范围，防止扰动后边权越界
        
        return perturbed_adj
    
    def _add_delete_edges(self, adj_matrix: torch.Tensor) -> torch.Tensor:
        """随机添加或删除边"""
        n_nodes = adj_matrix.size(0)
        
        # 构造扰动掩码
        mask = ~torch.eye(n_nodes, dtype=torch.bool, device=adj_matrix.device)
        if not self.preserve_self_loops: # 不扰动自环
            mask = torch.ones(n_nodes, n_nodes, dtype=torch.bool, device=adj_matrix.device)
        
        
        total_possible_edges = mask.sum().item() # 计算可扰动边数
        n_changes = int(total_possible_edges * self.edge_change_ratio) # 计算实际扰动边数
        
        # 从掩码中获取可扰动边索引
        flat_indices = torch.where(mask.flatten())[0]
        if len(flat_indices) > n_changes: # 随机选择具体扰动边
            selected_indices = torch.randperm(len(flat_indices))[:n_changes]
            selected_flat_indices = flat_indices[selected_indices]
        else:
            selected_flat_indices = flat_indices
        
        # 计算选中边的二维索引，将扁平索引转换回二维索引(i,j)
        selected_i = torch.div(selected_flat_indices, n_nodes, rounding_mode='floor')
        selected_j = selected_flat_indices % n_nodes
        
        # 随机增删边
        for i, j in zip(selected_i, selected_j):
            if random.random() < 0.5:  # 50%概率添加边
                if adj_matrix[i, j] < self.min_edge_weight:
                    adj_matrix[i, j] = random.uniform(self.min_edge_weight, self.max_edge_weight)
            else:  # 50%概率删除边
                if adj_matrix[i, j] > self.min_edge_weight:
                    adj_matrix[i, j] = self.min_edge_weight
        
        return adj_matrix
    
    def _adjust_edge_weights(self, adj_matrix: torch.Tensor) -> torch.Tensor:
        """通过高斯扰动，调整现有边的权重"""
        # 生成与邻接矩阵相同形状的正态分布噪声
        noise = torch.randn_like(adj_matrix) * self.weight_noise_std
        
        
        n_nodes = adj_matrix.size(0)
        edge_mask = adj_matrix > self.min_edge_weight # 构建边掩码：选出邻接矩阵中权重大于阈值的边，视为“存在边”
        if self.preserve_self_loops: # 自环不参与扰动
            
            diagonal_mask = ~torch.eye(n_nodes, dtype=torch.bool, device=adj_matrix.device)
            edge_mask = edge_mask & diagonal_mask
        
        adj_matrix[edge_mask] += noise[edge_mask]
        
        return adj_matrix
    
    def _symmetric_perturbation(self, adj_matrix: torch.Tensor) -> torch.Tensor:
        """对称扰动：保持图的连通性但打破确定性结构，随机对称加噪音或对称加边"""
        n_nodes = adj_matrix.size(0)
        
        # 生成对称噪声矩阵
        random_matrix = torch.randn(n_nodes, n_nodes, device=adj_matrix.device)
        symmetric_noise = (random_matrix + random_matrix.T) / 2 * self.weight_noise_std
        
        # 为已有边添加对称噪声
        edge_mask = adj_matrix > self.min_edge_weight
        adj_matrix[edge_mask] += symmetric_noise[edge_mask]
        
        # 新增对称边
        n_new_edges = int(n_nodes * self.edge_change_ratio)
        for _ in range(n_new_edges):
            i, j = random.randint(0, n_nodes-1), random.randint(0, n_nodes-1)
            if i != j and adj_matrix[i, j] <= self.min_edge_weight:
                weight = random.uniform(self.min_edge_weight, self.max_edge_weight * 0.3)
                adj_matrix[i, j] = weight
                adj_matrix[j, i] = weight  
        
        return adj_matrix
    
    def _ensure_symmetry(self, adj_matrix: torch.Tensor) -> torch.Tensor:
        """确保邻接矩阵的对称性"""
        return (adj_matrix + adj_matrix.T) / 2
    
    def _ensure_connectivity_preservation(self, perturbed_adj: torch.Tensor, 
                                        original_adj: torch.Tensor) -> torch.Tensor:
        """确保扰动后的图保持基本连通性"""
        n_nodes = perturbed_adj.size(0)
        
        
        for i in range(n_nodes):
            if self.preserve_self_loops: # 保留自环
                
                connections = perturbed_adj[i, :].clone()
                connections[i] = 0  
                if connections.sum() < self.min_edge_weight: # 根据加权出度计算该节点有无连接，如果没有，则从原始图中选择一个连接
                    
                    original_connections = original_adj[i, :] > self.min_edge_weight # 根据边阈值，有效选择连接
                    if original_connections.sum() > 1:  # 确保多于自环连接数量1
                        
                        valid_indices = torch.where(original_connections)[0]
                        if len(valid_indices) > 0: # 从有效节点中进行连接，对称赋值
                            chosen_idx = valid_indices[random.randint(0, len(valid_indices)-1)]
                            perturbed_adj[i, chosen_idx] = original_adj[i, chosen_idx]
                            perturbed_adj[chosen_idx, i] = original_adj[chosen_idx, i]
            else:# 不保留自环
                
                if perturbed_adj[i, :].sum() < self.min_edge_weight:
                    
                    original_connections = original_adj[i, :] > self.min_edge_weight
                    if original_connections.sum() > 0:
                        valid_indices = torch.where(original_connections)[0]
                        if len(valid_indices) > 0:
                            chosen_idx = valid_indices[random.randint(0, len(valid_indices)-1)]
                            perturbed_adj[i, chosen_idx] = original_adj[i, chosen_idx]
                            perturbed_adj[chosen_idx, i] = original_adj[chosen_idx, i]
        
        return perturbed_adj


class GraphAugmentationWrapper: # 图结构层面增强，不增加样本
    """
    图增强包装器，集成邻接矩阵扰动到现有的数据增强流程中
    """
    
    def __init__(self, 
                 base_transform=None,
                 adj_perturbation: Optional[AdjacencyPerturbation] = None,
                 apply_to_every_batch: bool = False):
        """
        Args:
            base_transform: 基础的数据变换（如CutMix、RandomCrop等）
            adj_perturbation: 邻接矩阵扰动器
            apply_to_every_batch: 是否对每个batch都应用邻接矩阵扰动
        """
        self.base_transform = base_transform
        self.adj_perturbation = adj_perturbation
        self.apply_to_every_batch = apply_to_every_batch
        
    def __call__(self, x, y=None, adj_matrix=None):
        """
        Args:
            x: 输入数据
            y: 标签
            adj_matrix: 邻接矩阵（如果需要扰动）
            
        Returns:
            如果adj_matrix为None，返回(x, y)；否则返回(x, y, perturbed_adj)
        """
        
        if self.base_transform is not None:
            if y is not None:
                x = self.base_transform(x, y)
            else:
                x = self.base_transform(x)
        
        
        if adj_matrix is not None and self.adj_perturbation is not None:
            perturbed_adj = self.adj_perturbation(adj_matrix, apply_perturbation=True)
            return x, y, perturbed_adj
        
        return x, y if y is not None else x 