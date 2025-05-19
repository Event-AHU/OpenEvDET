import cv2
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from scipy.spatial import KDTree
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch
import community.community_louvain as community_louvain

from src.core import register

__all__ = ["OLGraph"]

class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=1):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True)
        self.elu = nn.ELU()
        self.conv2 = GATConv(hidden_dim, output_dim, heads=heads, concat=False)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        device  = x.device
        self.conv1.to(device)
        self.conv2.to(device)
        
        x = self.conv1(x, edge_index)
        x = self.elu(x)
        x = self.conv2(x, edge_index)
        return x, batch

@register
class OLGraph(nn.Module):
    def __init__(self, patch_size, distance_thres, node_thres, k, input_dim, hidden_dim, output_dim) -> None:
        super(OLGraph, self).__init__()
        self.patch_size = patch_size
        self.distance_thres = distance_thres  # 用于初始图构建的距离阈值
        self.node_thres = node_thres          # 用于过滤小子图的节点数阈值
        self.k = k                            # k近邻图的参数
        
        # self.gnn = [GAT(input_dim, hidden_dim, dim) for dim in output_dim]


    def process_image(self, image):
        """
        将输入图像按照 patch_size 切分，返回每个图像块的中心坐标和图像块原始数据。
        """
        h, w = image.shape[-2:]
        centers = []
        patches = []
        for y in range(0, h, self.patch_size):
            for x in range(0, w, self.patch_size):
                center_x = x + self.patch_size // 2
                center_y = y + self.patch_size // 2
                centers.append((center_x, center_y))
                # 注意防止边缘越界
                patch = image[:, y:min(y+self.patch_size, h), x:min(x+self.patch_size, w)]
                patches.append(patch)
        if not centers:
            return None, None
        centers = torch.tensor(centers, dtype=float)
        return centers, patches

    def graph_construction(self, centers, patches):
        """
        根据每个图像块的中心构建初始图，并在节点中加入图像块的原始数据。
        使用 Louvain 算法进行图分割。
        接着对初始图进行抽象，然后构建 k 近邻图，最终转换为 torch_geometric 的 Data 对象。
        """
        device = patches[0].device
        # 构建初始图：利用 KDTree 查询距离小于 distance_thres 的块
        tree = KDTree(centers)
        edges = tree.query_pairs(self.distance_thres)
        G = nx.Graph()
        G.add_nodes_from(range(len(centers)))
        # 为每个节点赋予原始图像块数据作为特征
        for i, patch in enumerate(patches):
            G.nodes[i]['feature'] = patch
        G.add_edges_from(edges)

        x = torch.stack([G.nodes[i]['feature'] for i in G.nodes])
        node_num, _, _, _ = x.shape
        x = x.view(node_num, -1).to(device)
        edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1).to(device)  # 添加反向边
        data1 = Data(x=x, edge_index=edge_index)


        # 使用 Louvain 算法进行图分割
        partition = community_louvain.best_partition(G, resolution=5, random_state=0)

        G_community = G.copy()
        for u, v in list(G.edges()):
            if partition[u] != partition[v]:  # 如果两个节点不属于同一社区，则删除边
                G_community.remove_edge(u, v)

        edge_index_filtered = torch.tensor(list(G_community.edges), dtype=torch.long).t().contiguous()
        edge_index_filtered = torch.cat([edge_index_filtered, edge_index_filtered.flip(0)], dim=1).to(device)

        x_filtered = torch.stack([G_community.nodes[i]['feature'] for i in G_community.nodes])
        num_node, _, _, _ = x_filtered.shape
        x_filtered = x_filtered.view(num_node, -1).to(device)

        data2 = Data(x=x_filtered, edge_index=edge_index_filtered)


        # 将节点按社区分组
        community_dict = {}
        for node, comm in partition.items():
            if comm not in community_dict:
                community_dict[comm] = []
            community_dict[comm].append(node)
            
        new_nodes = []
        new_features = []
        for nodes_in_comm in community_dict.values():
            if len(nodes_in_comm) < self.node_thres:
                continue
            sg_centers = centers[nodes_in_comm]
            mean_center = sg_centers.mean(0)
            new_nodes.append(mean_center)

            # 聚合图像块特征
            patch_feats = torch.stack([G.nodes[i]['feature'] for i in nodes_in_comm])
            mean_patch = patch_feats.mean(dim=0)
            new_features.append(mean_patch)

        if not new_nodes:
            return None

        new_nodes = torch.stack(new_nodes)
        new_features = torch.stack(new_features)



        # 构建抽象图：以抽象节点的中心构建 k 近邻图
        tree = KDTree(new_nodes.cpu().numpy())
        new_edges = []
        for i in range(len(new_nodes)):
            distances, indices = tree.query(new_nodes[i].cpu().numpy().reshape(1, -1), k=self.k + 1)
            for idx in indices[0][1:]:  # 排除自身
                # new_edges.append((i, indices[0][idx]))
                new_edges.append((i, idx))

        new_G = nx.Graph()
        new_G.add_nodes_from(range(len(new_nodes)))
        for i in range(len(new_nodes)):
            if new_features[i] != None:
                new_G.nodes[i]['feature'] = new_features[i]
        new_G.add_edges_from(new_edges)

        # 将 new_G 转换为 torch_geometric 的 Data 对象
        x = []
        for i in range(len(new_G.nodes)):
            # if 'feature' in new_G.nodes[i]:
            feat = new_G.nodes[i]['feature']
            feat_vec = feat.flatten()
            x.append(feat_vec)
        x = torch.stack(x)
        # device = x.device
        # 构造 edge_index
        edge_index = torch.tensor(list(new_G.edges), dtype=torch.long).t().contiguous().to(device)
        data3 = Data(x=x, edge_index=edge_index)

        return [data1, data2, data3]
    


    def forward(self, img_batch):
        batch_data_s1 = []
        batch_data_s2 = []
        batch_data_s3 = []
        for img in img_batch:
            centers, patches = self.process_image(img)
            data1, data2, data3 = self.graph_construction(centers, patches)
            batch_data_s1.append(data1)
            batch_data_s2.append(data2)
            batch_data_s3.append(data3)
        batch_data_s1 = Batch.from_data_list(batch_data_s1)
        batch_data_s2 = Batch.from_data_list(batch_data_s2)
        batch_data_s3 = Batch.from_data_list(batch_data_s3)
        # batch_data_s1 = Batch.from_data_list(batch_data_s1)
        batch_data = [batch_data_s1, batch_data_s2, batch_data_s3, batch_data_s3]
        return batch_data
        # res = []
        # for i in range(len(self.gnn)):
        #     x, batch = self.gnn[i](batch_data[i])
        #     x, mask = to_dense_batch(x, batch) # x.shape=bs*n*dim
        #     res.append(x)
        # return res

        # batch_data = []
        # for img in img_batch:
        #     centers, patches = self.process_image(img)
        #     data = self.graph_construction(centers, patches)
        #     batch_data.append(data)
        # batch_data = Batch.from_data_list(batch_data)
        # res = []
        # for i in range(len(self.gnn)):
        #     x, batch = self.gnn[i](batch_data)
        #     x, mask = to_dense_batch(x, batch) # x.shape=bs*n*dim
        #     res.append(x)
        # return res
    

class OLGraph_MM(nn.Module):
    def __init__(self, patch_size, distance_thres, node_thres, k, input_dim, hidden_dim, output_dim) -> None:
        super(OLGraph_MM, self).__init__()
        self.patch_size = patch_size
        self.distance_thres = distance_thres  # 用于初始图构建的距离阈值
        self.node_thres = node_thres          # 用于过滤小子图的节点数阈值
        self.k = k                            # k近邻图的参数
        
        self.relu = nn.ReLU()
        self.gnn = [GAT(input_dim, hidden_dim, dim) for dim in output_dim]
        # self.tranform = nn.Sequential(nn.Conv2d(in_channels= resolution))

    def process_image(self, image):
        """
        将输入图像按照 patch_size 切分，返回每个图像块的中心坐标和图像块原始数据。
        """
        h, w = image.shape[-2:]
        centers = []
        patches = []
        for y in range(0, h, self.patch_size):
            for x in range(0, w, self.patch_size):
                center_x = x + self.patch_size // 2
                center_y = y + self.patch_size // 2
                centers.append((center_x, center_y))
                # 注意防止边缘越界
                patch = image[:, y:min(y+self.patch_size, h), x:min(x+self.patch_size, w)]
                patches.append(patch)
        if not centers:
            return None, None
        centers = torch.tensor(centers, dtype=float)
        return centers, patches

    def graph_construction(self, centers, patches):
        """
        根据每个图像块的中心构建初始图，并在节点中加入图像块的原始数据。
        使用 Louvain 算法进行图分割。
        接着对初始图进行抽象，然后构建 k 近邻图，最终转换为 torch_geometric 的 Data 对象。
        """
        # 构建初始图：利用 KDTree 查询距离小于 distance_thres 的块
        num_voxel, feat_dim = patches.shape
        tree = KDTree(centers)
        edges = tree.query_pairs(self.distance_thres)
        G = nx.Graph()
        G.add_nodes_from(range(len(centers)))
        # 为每个节点赋予原始图像块数据作为特征
        for i, patch in enumerate(patches):
            G.nodes[i]['feature'] = patch
        G.add_edges_from(edges)

        # 使用 Louvain 算法进行图分割
        partition = community_louvain.best_partition(G, resolution=10, random_state=0)

        # 将节点按社区分组
        community_dict = {}
        for node, comm in partition.items():
            if comm not in community_dict:
                community_dict[comm] = []
            community_dict[comm].append(node)

        # 对每个社区生成一个抽象节点
        new_nodes = []
        new_features = []
        for nodes_in_comm in community_dict.values():
            if len(nodes_in_comm) < self.node_thres:
                continue
            sg_centers = centers[nodes_in_comm]
            mean_center = sg_centers.mean(0)
            new_nodes.append(mean_center)

            # 聚合图像块特征
            patch_feats = torch.stack([G.nodes[i]['feature'] for i in nodes_in_comm])
            mean_patch = patch_feats.mean(dim=0)
            new_features.append(mean_patch)

        if not new_nodes:
            x = torch.zeros((1, feat_dim), dtype=torch.float32, device=patches.device)  # 用默认维度补1个点
            edge_index = torch.empty((2, 0), dtype=torch.long, device=patches.device)
            return Data(x=x, edge_index=edge_index)

        new_nodes = torch.stack(new_nodes)
        new_features = torch.stack(new_features)

        # 构建抽象图：以抽象节点的中心构建 k 近邻图
        tree = KDTree(new_nodes.cpu().numpy())
        new_edges = []
        for i in range(len(new_nodes)):
            distances, indices = tree.query(new_nodes[i].cpu().numpy().reshape(1, -1), k=self.k + 1)
            for idx in indices[0][1:]:  # 排除自身
                # new_edges.append((i, indices[0][idx]))
                new_edges.append((i, idx))

        new_G = nx.Graph()
        new_G.add_nodes_from(range(len(new_nodes)))
        for i in range(len(new_nodes)):
            if new_features[i] != None:
                new_G.nodes[i]['feature'] = new_features[i]
        new_G.add_edges_from(new_edges)

        # 将 new_G 转换为 torch_geometric 的 Data 对象
        x = []
        for i in range(len(new_G.nodes)):
            if 'feature' in new_G.nodes[i]:
                feat = new_G.nodes[i]['feature']
                feat_vec = feat.flatten()
                x.append(feat_vec)
        if len(x) == 0:
            x.append(torch.zeros((1, feat_dim), dtype=torch.float32, device=patches.device))
        x = torch.stack(x)
        device = x.device
        # 构造 edge_index
        edge_index = torch.tensor(list(new_G.edges), dtype=torch.long).t().contiguous().to(device)

        return Data(x=x, edge_index=edge_index)
    


    def forward(self, voxel_batch):
        """
        处理一个批次的图像并返回一个批次的结果。
        """

        batch_data = []
        for voxel in voxel_batch:
            centers = voxel['coor'].cpu()
            patches = voxel['features']
            data = self.graph_construction(centers, patches)
            batch_data.append(data)
        batch_data = Batch.from_data_list(batch_data)
        res = []
        for i in range(len(self.gnn)):
            x, batch = self.gnn[i](batch_data)
            x, mask = to_dense_batch(x, batch) # x.shape=bs*n*dim
            res.append(x)
        return res



# 使用示例
if __name__ == "__main__":
    device = 'cuda:6'
    image = torch.randn((2, 3, 640, 640), device = device)
    
    patch_size = 32  # 图像块大小
    d = 100          # 邻接距离阈值
    n = 5           # 最小子图节点数
    k = 3           # 近邻数量
    
    model = OLGraph(patch_size=patch_size, distance_thres=d, node_thres=n, k=k, input_dim=3072, hidden_dim=1024, output_dim=96)
    model = model.to(device) 

    res = model(image, 56)
    print(res.shape)
