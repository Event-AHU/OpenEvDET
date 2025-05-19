import torch 
import torch.utils.data as data
from torch_geometric.data import Data, Batch

from src.core import register


__all__ = ['DataLoader']


@register
class DataLoader(data.DataLoader):
    __inject__ = ['dataset', 'collate_fn']

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for n in ['dataset', 'batch_size', 'num_workers', 'drop_last', 'collate_fn']:
            format_string += "\n"
            format_string += "    {0}: {1}".format(n, getattr(self, n))
        format_string += "\n)"
        return format_string



@register
def default_collate_fn(items):
    '''default collate_fn
    '''    
    return torch.cat([x[0][None] for x in items], dim=0), [x[1] for x in items]

@register
def eso_collate_fn(batch):
    imgs, density, targets = zip(*batch)  # 解压批次数据

    # 将imgs、density和targets转换为适当的tensor
    imgs = torch.stack(imgs,dim=0)  # 假设 imgs 是torch.Tensor
    density = torch.stack(density, dim=0)  # 假设 density 是torch.Tensor
    # targets = torch.stack(targets)  # 假设 targets 是torch.Tensor

    return imgs, density, targets

@register
def mm_collate_fn(items):

    return torch.cat([x[0][None] for x in items], dim=0), [x[1] for x in items], [x[2] for x in items]

@register
def graph_collate_fn(items):
    x0 = torch.cat([x[0][None] for x in items], dim=0)
    x2= [x[2] for x in items]
    x1=[]
    batch_data_s1 = []
    batch_data_s2 = []
    batch_data_s3 = []
    for item in items:
        data = item[1]
        batch_data_s1.append(Data(x=torch.tensor(data[0]['x']), edge_index=torch.tensor(data[0]['edge_index']), num_nodes=len(data[0]['x'])))
        batch_data_s2.append(Data(x=torch.tensor(data[1]['x']), edge_index=torch.tensor(data[1]['edge_index']), num_nodes=len(data[1]['x'])))
        batch_data_s3.append(Data(x=torch.tensor(data[2]['x']), edge_index=torch.tensor(data[2]['edge_index']), num_nodes=len(data[2]['x'])))
    batch_data_s1 = Batch.from_data_list(batch_data_s1)
    batch_data_s2 = Batch.from_data_list(batch_data_s2)
    batch_data_s3 = Batch.from_data_list(batch_data_s3)
    x1= [batch_data_s1, batch_data_s2, batch_data_s3, batch_data_s3]

    return x0, x1, x2