import torch 
import torch.utils.data as data

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