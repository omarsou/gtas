import torch


class DrugDataset(torch.utils.data.Dataset):
    def __init__(self, graph_list, label_list):
        self.graph_list = graph_list
        self.label_list = label_list

    def __getitem__(self, idx):
        graph = self.graph_list[idx]
        y = self.label_list[idx]
        return graph, y

    def __len__(self):
        return len(self.graph_list)
