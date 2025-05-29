import torch
import torch.nn as nn
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from skimage.util import img_as_float
from skimage.segmentation import slic
import time


class PixelAggregationNetwork(nn.Module):
    def __init__(self, in_channels, threshold_start=5, threshold_end=10, max_layers=100, texture_weight=0.5):
        super(PixelAggregationNetwork, self).__init__()
        self.in_channels = in_channels
        self.texture_weight = texture_weight
        self.threshold_start = threshold_start
        self.threshold_end = threshold_end
        self.max_layers = max_layers

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        segments = self.superpixel_segmentation(x)
        num_segments = np.max(segments) + 1
        segments = torch.from_numpy(segments).to(x.device)
        segments_flat = segments.view(batch_size, -1)
        pixel_features = x.view(batch_size, channels, -1).transpose(1, 2)  # 展平
        nodes = torch.stack([pixel_features[b][segments_flat[b] == i].mean(dim=0) for b in range(batch_size) for i in range(num_segments)], dim=0).view(batch_size, num_segments, -1)
        tree_structure = torch.zeros(batch_size, num_segments, 2, dtype=torch.long)
        tree_structure[:, :, 0] = torch.arange(num_segments).unsqueeze(0)
        current_layer = nodes
        total_loss = 0.0
        threshold = self.threshold_start
        layer = 0

        while layer < self.max_layers:
            start_time = time.time()
            texture_features = self.calculate_texture_features(x, segments, num_segments)
            weighted_features = self.combine_features(current_layer, texture_features)
            aggregated_nodes, parents = self.aggregate_nodes(weighted_features, threshold)
            tree_structure = self.update_tree_structure(tree_structure, parents, layer)
            segments = aggregated_nodes
            num_segments = aggregated_nodes.size(1)
            threshold = min(self.threshold_end, threshold * 1.1)
            if aggregated_nodes.size(1) == 1:
                break
            current_layer = aggregated_nodes
            layer += 1
        lca_matrix = self.compute_lca_matrix(tree_structure)
        layer_loss = self.hierarchical_contrastive_loss(tree_structure, lca_matrix)
        total_loss += layer_loss
        return tree_structure, total_loss

    def superpixel_segmentation(self, x):
        batch_size, channels, height, width = x.size()
        segments = []
        for b in range(batch_size):
            image = x[b].permute(1, 2, 0).detach().cpu().numpy()
            image = img_as_float(image)
            segment = slic(image, n_segments=10, compactness=10)
            segments.append(segment)
        return np.stack(segments)

    def calculate_texture_features(self, x, segments, num_segments):
        batch_size, channels, height, width = x.size()
        texture_features = []

        for b in range(batch_size):
            image = x[b].mean(dim=0)
            segments_b = segments[b]

            for i in range(num_segments):
                mask = (segments_b == i)
                if mask.sum() == 0:
                    texture_features.append(torch.zeros(1, device=x.device))
                    continue
                if mask.shape != (height, width):
                    new_mask = torch.zeros((height, width), dtype=torch.bool, device=mask.device)
                    mask_height, mask_width = mask.shape
                    new_mask[:mask_height, :mask_width] = mask
                    mask = new_mask
                region = image[mask]
                if region.numel() == 0:
                    texture_features.append(torch.zeros(1, device=x.device))
                    continue
                region_size = region.numel()
                region_height = int(torch.sqrt(torch.tensor(region_size, dtype=torch.float32)))
                region_width = region_height
                region_2d = torch.zeros((region_height, region_width), device=x.device)
                region_2d_size = region_2d.numel()
                if region_size > region_2d_size:
                    region_2d.view(-1)[:region_2d_size] = region.view(-1)[:region_2d_size]
                else:
                    region_2d.view(-1)[:region_size] = region
                fft = torch.fft.fft2(region_2d)
                fft_shift = torch.fft.fftshift(fft)
                magnitude_spectrum = torch.abs(fft_shift)
                energy = torch.sum(magnitude_spectrum ** 2) / magnitude_spectrum.numel()
                texture_features.append(energy.unsqueeze(0))
        texture_features = torch.stack(texture_features).to(x.device)
        texture_features = texture_features.view(batch_size, num_segments, -1)
        return texture_features

    def combine_features(self, nodes, texture_nodes):
        mean_value = torch.nanmean(nodes)
        nodes = torch.where(torch.isnan(nodes), mean_value, nodes)
        combined_features = nodes * (1 - self.texture_weight) + texture_nodes * self.texture_weight
        return combined_features

    def calculate_pairwise_distances(self, nodes, texture_nodes, threshold, layer):
        gray_features = nodes
        texture_features = texture_nodes
        weighted_features = gray_features * (1 - self.texture_weight) + texture_features * self.texture_weight
        dist_matrix = torch.cdist(weighted_features, weighted_features, p=2)
        return dist_matrix

    def aggregate_nodes(self, features, threshold):
        batch_size, num_nodes, _ = features.size()
        aggregated_nodes = []
        parents = []
        max_new_num_nodes = 0
        for b in range(batch_size):
            clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, linkage='average')
            cluster_labels = torch.from_numpy(clustering.fit_predict(features[b].detach().cpu().numpy())).to(features.device)
            unique_labels = torch.unique(cluster_labels)
            max_new_num_nodes = max(max_new_num_nodes, len(unique_labels))
        for b in range(batch_size):
            clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, linkage='average')
            cluster_labels = torch.from_numpy(clustering.fit_predict(features[b].detach().cpu().numpy())).to(features.device)
            unique_labels = torch.unique(cluster_labels)
            new_nodes = torch.stack([features[b][cluster_labels == label].mean(dim=0) for label in unique_labels])
            if new_nodes.size(0) < max_new_num_nodes:
                padding_size = max_new_num_nodes - new_nodes.size(0)
                padding = torch.zeros(padding_size, new_nodes.size(1), device=new_nodes.device)
                new_nodes = torch.cat([new_nodes, padding], dim=0)
            aggregated_nodes.append(new_nodes)
            parents.append(cluster_labels)
        aggregated_nodes = torch.stack(aggregated_nodes)
        parents = torch.stack(parents)
        return aggregated_nodes, parents

    def update_tree_structure(self, tree_structure, parents, layer):
        batch_size, num_nodes, _ = tree_structure.size()
        if tree_structure.size(2) < 100:
            tree_structure = torch.cat([tree_structure, torch.zeros_like(tree_structure).expand(-1, -1, tree_structure.size(2))], dim=2)
        for b in range(batch_size):
            for i in range(num_nodes):
                if i >= parents.size(1):
                    continue
                current_parent = parents[b, i]
                tree_structure[b, i, 0] = current_parent
                tree_structure[b, i, 1] = layer
                if layer > 0:
                    tree_structure[b, i, 2:layer + 2] = tree_structure[b, current_parent, 2:layer + 2]
        return tree_structure

    def hierarchical_contrastive_loss(self, features, lca_matrix, alpha=1.0):
        features = features.to('cuda').float()
        lca_matrix = lca_matrix.to('cuda').float()
        batch_size, num_nodes, feature_dim = features.size()
        distances = torch.cdist(features, features, p=2)
        max_depth = self.max_layers
        sim_matrix = lca_matrix / max_depth
        positive_loss = (1 - sim_matrix) * distances
        negative_loss = sim_matrix * torch.relu(alpha - distances)
        mask = torch.eye(num_nodes, device=features.device).bool().unsqueeze(0)
        total_loss = positive_loss.masked_fill(mask, 0).sum() + negative_loss.masked_fill(mask, 0).sum()
        return total_loss / (batch_size * num_nodes * (num_nodes - 1))


    def compute_lca_matrix(self, tree_structure):
        lca_finder = HTCIFinder(tree_structure)
        queries = {i: [] for i in range(tree_structure.size(1))}
        for i in range(tree_structure.size(1)):
            for j in range(tree_structure.size(1)):
                if i != j:
                    queries[i].append(j)
        return lca_finder.compute_lca(queries)


class HTCIFinder:
    def __init__(self, tree_structure):
        batch_size, num_nodes, _ = tree_structure.size()
        self.batch_size = batch_size
        self.num_nodes = num_nodes
        self.tree_structure = tree_structure
        self.ancestor = torch.arange(num_nodes, device=tree_structure.device).repeat(batch_size, 1)
        self.parent = torch.arange(num_nodes, device=tree_structure.device).repeat(batch_size, 1)
        self.rank = torch.zeros_like(self.parent, dtype=torch.int32)
        self.visited = torch.zeros((batch_size, num_nodes), dtype=torch.bool, device=tree_structure.device)

    def find(self, b, node):
        if self.parent[b, node] != node:
            self.parent[b, node] = self.find(b, self.parent[b, node])
        return self.parent[b, node]

    def union(self, b, node1, node2):
        root1 = self.find(b, node1)
        root2 = self.find(b, node2)
        if root1 != root2:
            if self.rank[b, root1] > self.rank[b, root2]:
                self.parent[b, root2] = root1
            elif self.rank[b, root1] < self.rank[b, root2]:
                self.parent[b, root1] = root2
            else:
                self.parent[b, root2] = root1
                self.rank[b, root1] += 1

    def dfs(self, b, node, adj_list, queries, lca_matrix):
        self.ancestor[b, node] = node
        self.visited[b, node] = True
        for child in adj_list[b][node]:
            if not self.visited[b, child]:
                self.dfs(b, child, adj_list, queries, lca_matrix)
                self.union(b, node, child)
                self.ancestor[b, self.find(b, node)] = node
        for other_node in queries[node]:
            if self.visited[b, other_node]:
                lca = self.ancestor[b, self.find(b, other_node)]
                lca_matrix[b, node, other_node] = lca
                lca_matrix[b, other_node, node] = lca

    def compute_lca(self, queries):
        adj_list = self.build_adjacency_list()
        lca_matrix = torch.zeros((self.batch_size, self.num_nodes, self.num_nodes),
                                 device=self.tree_structure.device, dtype=torch.int32)

        for b in range(self.batch_size):
            self.visited[b].fill_(False)
            self.parent[b] = torch.arange(self.num_nodes, device=self.tree_structure.device)
            self.rank[b].fill_(0)
            self.dfs(b, 0, adj_list, queries, lca_matrix)

        return lca_matrix

    def build_adjacency_list(self):
        adj_list = [[] for _ in range(self.batch_size)]
        for b in range(self.batch_size):
            adj_list[b] = [[] for _ in range(self.num_nodes)]
            for i in range(self.num_nodes):
                parent = self.tree_structure[b, i, 0].item()
                if parent != i:  # 排除根节点
                    adj_list[b][parent].append(i)
        return adj_list


if __name__ == '__main__':
    input_image = torch.randn(1, 1, 256, 256)  # 假设输入是一个256x256的3通道图像
    texture_nodes = torch.randn(1, 256 * 256, 1)  # 假设纹理特征与输入图像形状相同
    model = PixelAggregationNetwork(in_channels=1)

    tree_structure, aggregated_features = model(input_image, texture_nodes)
    print("Tree Structure Shape:", tree_structure.shape)
    print("Aggregated Features Shape:", aggregated_features.shape)

