import time
import math
from functools import partial
from typing import Optional, Callable

from src.core import register
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from src.zoo.evheat.outline import OLGraph
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_dense_batch
# from outline import OLGraph

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

__all__ = ["CvHeat"]


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class to_channels_first(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2).contiguous()


class to_channels_last(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1).contiguous()


def build_norm_layer(dim, norm_layer, in_format="channels_last", out_format="channels_last", eps=1e-6):
    layers = []
    if norm_layer == "BN":
        if in_format == "channels_last":
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == "channels_last":
            layers.append(to_channels_last())
    elif norm_layer == "LN":
        if in_format == "channels_first":
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == "channels_first":
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(f"build_norm_layer does not support {norm_layer}")
    return nn.Sequential(*layers)


def build_act_layer(act_layer):
    if act_layer == "ReLU":
        return nn.ReLU(inplace=True)
    elif act_layer == "SiLU":
        return nn.SiLU(inplace=True)
    elif act_layer == "GELU":
        return nn.GELU()

    raise NotImplementedError(f"build_act_layer does not support {act_layer}")


class StemLayer(nn.Module):
    r"""Stem layer of InternImage
    Args:
        in_chans (int): number of input channels
        out_chans (int): number of output channels
        act_layer (str): activation layer
        norm_layer (str): normalization layer
    """

    def __init__(self, in_chans=3, out_chans=96, act_layer="GELU", norm_layer="BN"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans // 2, kernel_size=3, stride=2, padding=1)
        self.norm1 = build_norm_layer(out_chans // 2, norm_layer, "channels_first", "channels_first")
        self.act = build_act_layer(act_layer)
        self.conv2 = nn.Conv2d(out_chans // 2, out_chans, kernel_size=3, stride=2, padding=1)
        self.norm2 = build_norm_layer(out_chans, norm_layer, "channels_first", "channels_first")

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x

class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=1):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True)
        self.elu = nn.ELU()
        self.bn = nn.BatchNorm1d(hidden_dim * heads)
        self.conv2 = GATConv(hidden_dim, output_dim, heads=heads, concat=False)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.elu(x)
        x = self.conv2(x, edge_index)

        return x, batch

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0, channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Heat2D(nn.Module):
    """
    du/dt -k(d2u/dx2 + d2u/dy2) = 0;
    du/dx_{x=0, x=a} = 0
    du/dy_{y=0, y=b} = 0
    =>
    A_{n, m} = C(a, b, n==0, m==0) * sum_{0}^{a}{ sum_{0}^{b}{\phi(x, y)cos(n\pi/ax)cos(m\pi/by)dxdy }}
    core = cos(n\pi/ax)cos(m\pi/by)exp(-[(n\pi/a)^2 + (m\pi/b)^2]kt)
    u_{x, y, t} = sum_{0}^{\infinite}{ sum_{0}^{\infinite}{ core } }

    assume a = N, b = M; x in [0, N], y in [0, M]; n in [0, N], m in [0, M]; with some slight change
    =>
    (\phi(x, y) = linear(dwconv(input(x, y))))
    A(n, m) = DCT2D(\phi(x, y))
    u(x, y, t) = IDCT2D(A(n, m) * exp(-[(n\pi/a)^2 + (m\pi/b)^2])**kt)
    """

    def __init__(self, infer_mode=False, res=14, dim=96, hidden_dim=96, **kwargs):
        super().__init__()
        self.res = res
        self.dwconv = nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.infer_mode = infer_mode
        self.to_k = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
        )


        self.to_k2 = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim*2, bias=True),
            nn.ReLU(),
        )
        self.linear2 = nn.Linear(hidden_dim*2, hidden_dim*2, bias=True)
        self.linear3 = nn.Linear(hidden_dim*2, hidden_dim, bias=True)

    def infer_init_heat2d(self, freq):
        weight_exp = self.get_decay_map((self.res, self.res), device=freq.device)
        self.k_exp = nn.Parameter(torch.pow(weight_exp[:, :, None], self.to_k(freq)), requires_grad=False)
        del self.to_k

    @staticmethod
    def get_cos_map(N=224, device=torch.device("cpu"), dtype=torch.float):
        # cos((x + 0.5) / N * n * \pi) which is also the form of DCT and IDCT
        # DCT: F(n) = sum( (sqrt(2/N) if n > 0 else sqrt(1/N)) * cos((x + 0.5) / N * n * \pi) * f(x) )
        # IDCT: f(x) = sum( (sqrt(2/N) if n > 0 else sqrt(1/N)) * cos((x + 0.5) / N * n * \pi) * F(n) )
        # returns: (Res_n, Res_x)
        weight_x = (torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(1, -1) + 0.5) / N
        weight_n = torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(-1, 1)
        weight = torch.cos(weight_n * weight_x * torch.pi) * math.sqrt(2 / N)
        weight[0, :] = weight[0, :] / math.sqrt(2)
        return weight

    @staticmethod
    def get_decay_map(resolution=(224, 224), device=torch.device("cpu"), dtype=torch.float):
        # exp(-[(n\pi/a)^2 + (m\pi/b)^2])
        # returns: (Res_h, Res_w)
        resh, resw = resolution
        weight_n = torch.linspace(0, torch.pi, resh + 1, device=device, dtype=dtype)[:resh].view(-1, 1)
        weight_m = torch.linspace(0, torch.pi, resw + 1, device=device, dtype=dtype)[:resw].view(1, -1)
        weight = torch.pow(weight_n, 2) + torch.pow(weight_m, 2)
        weight = torch.exp(-weight)
        return weight

    def forward(self, x: torch.Tensor, freq_embed=None, outline_feat=None):
        B, C, H, W = x.shape
        x = self.dwconv(x)

        x = self.linear(x.permute(0, 2, 3, 1).contiguous())  # B, H, W, 2C
        x, z = x.chunk(chunks=2, dim=-1)  # B, H, W, C

        if ((H, W) == getattr(self, "__RES__", (0, 0))) and (getattr(self, "__WEIGHT_COSN__", None).device == x.device):
            weight_cosn = getattr(self, "__WEIGHT_COSN__", None)
            weight_cosm = getattr(self, "__WEIGHT_COSM__", None)
            weight_exp = getattr(self, "__WEIGHT_EXP__", None)
            assert weight_cosn is not None
            assert weight_cosm is not None
            assert weight_exp is not None
        else:
            weight_cosn = self.get_cos_map(H, device=x.device).detach_()
            weight_cosm = self.get_cos_map(W, device=x.device).detach_()
            weight_exp = self.get_decay_map((H, W), device=x.device).detach_()
            setattr(self, "__RES__", (H, W))
            setattr(self, "__WEIGHT_COSN__", weight_cosn)
            setattr(self, "__WEIGHT_COSM__", weight_cosm)
            setattr(self, "__WEIGHT_EXP__", weight_exp)

        N, M = weight_cosn.shape[0], weight_cosm.shape[0]

        x = F.conv1d(x.contiguous().view(B, H, -1), weight_cosn.contiguous().view(N, H, 1))
        x = F.conv1d(x.contiguous().view(-1, W, C), weight_cosm.contiguous().view(M, W, 1)).contiguous().view(B, N, M, -1)

        if self.infer_mode:
            x = torch.einsum("bnmc,nmc->bnmc", x, self.k_exp)
        else:
            otfeat = outline_feat.mean(dim=0)
            weight_exp = torch.pow(weight_exp[:, :, None], self.to_k(freq_embed + otfeat))
            x = torch.einsum("bnmc,nmc -> bnmc", x, weight_exp)  # exp decay

        x = F.conv1d(x.contiguous().view(B, N, -1), weight_cosn.t().contiguous().view(H, N, 1))
        x = F.conv1d(x.contiguous().view(-1, M, C), weight_cosm.t().contiguous().view(W, M, 1)).contiguous().view(B, H, W, -1)


         #####################second heat conduct################################
        
        x = torch.cat((x, outline_feat), dim=-1) # bs*h*w*2c 

        if ((H, W) == getattr(self, "__RES2__", (0, 0))) and (getattr(self, "__WEIGHT_COSN2__", None).device == x.device):
            weight_cosn2 = getattr(self, "__WEIGHT_COSN2__", None)
            weight_cosm2 = getattr(self, "__WEIGHT_COSM2__", None)
            weight_exp2 = getattr(self, "__WEIGHT_EXP2__", None)
            assert weight_cosn2 is not None
            assert weight_cosm2 is not None
            assert weight_exp2 is not None
        else:
            weight_cosn2 = self.get_cos_map(H, device=x.device).detach_()
            weight_cosm2 = self.get_cos_map(W, device=x.device).detach_()
            weight_exp2 = self.get_decay_map((H, W), device=x.device).detach_()
            setattr(self, "__RES2__", (H, W))
            setattr(self, "__WEIGHT_COSN2__", weight_cosn2)
            setattr(self, "__WEIGHT_COSM2__", weight_cosm2)
            setattr(self, "__WEIGHT_EXP2__", weight_exp2)

        N, M = weight_cosn2.shape[0], weight_cosm2.shape[0]

        skfeat = self.linear2(x)
        skfeat = skfeat.mean(dim=0)

        x = F.conv1d(x.contiguous().view(B, H, -1), weight_cosn2.contiguous().view(N, H, 1))
        x = F.conv1d(x.contiguous().view(-1, W, C), weight_cosm2.contiguous().view(M, W, 1)).contiguous().view(B, N, M, -1)
        if self.infer_mode:
            x = torch.einsum("bnmc,nmc->bnmc", x, self.k_exp2)
        else:
            weight_exp2 = torch.pow(weight_exp2[:, :, None], self.to_k2(skfeat))
            x = torch.einsum("bnmc,nmc -> bnmc", x, weight_exp2)
        x = F.conv1d(x.contiguous().view(B, N, -1), weight_cosn2.t().contiguous().view(H, N, 1))
        x = F.conv1d(x.contiguous().view(-1, M, C), weight_cosm2.t().contiguous().view(W, M, 1)).contiguous().view(B, H, W, -1)

        x = self.linear3(x) 
        x = self.out_norm(x)
        x = x * nn.functional.silu(z)
        x = self.out_linear(x)

        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class HeatBlock(nn.Module):
    def __init__(
        self,
        res: int = 14,
        infer_mode=False,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        use_checkpoint: bool = False,
        drop: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        mlp_ratio: float = 4.0,
        post_norm=True,
        layer_scale=None,
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = norm_layer(hidden_dim)
        self.op = Heat2D(res=res, dim=hidden_dim, hidden_dim=hidden_dim, infer_mode=infer_mode)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, channels_first=True)
        self.post_norm = post_norm
        self.layer_scale = layer_scale is not None

        self.infer_mode = infer_mode

        if self.layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(hidden_dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(hidden_dim), requires_grad=True)

    def _forward(self, x: torch.Tensor, freq_embed, outline_feat):
        if not self.layer_scale:
            if self.post_norm:
                x = x + self.drop_path(self.norm1(self.op(x, freq_embed, outline_feat)))
                if self.mlp_branch:
                    x = x + self.drop_path(self.norm2(self.mlp(x)))  # FFN
            else:
                x = x + self.drop_path(self.op(self.norm1(x), freq_embed, outline_feat))
                if self.mlp_branch:
                    x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
            return x
        if self.post_norm:
            x = x + self.drop_path(self.gamma1[:, None, None] * self.norm1(self.op(x, freq_embed, outline_feat)))
            if self.mlp_branch:
                x = x + self.drop_path(self.gamma2[:, None, None] * self.norm2(self.mlp(x)))  # FFN
        else:
            x = x + self.drop_path(self.gamma1[:, None, None] * self.op(self.norm1(x), freq_embed, outline_feat))
            if self.mlp_branch:
                x = x + self.drop_path(self.gamma2[:, None, None] * self.mlp(self.norm2(x)))  # FFN
        return x

    def forward(self, input: torch.Tensor, freq_embed=None, outline_feat=None):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input, freq_embed, outline_feat)
        else:
            return self._forward(input, freq_embed, outline_feat)


class AdditionalInputSequential(nn.Sequential):
    def forward(self, x, *args, **kwargs):
        for module in self[:-1]:
            if isinstance(module, nn.Module):
                x = module(x, *args, **kwargs)
            else:
                x = module(x)
        x = self[-1](x)
        return x


@register
class GvHeat(nn.Module):
    def __init__(
        self,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        depths=[2, 2, 9, 2],
        dims=[96, 192, 384, 768],
        graph_patch=32,
        distance_thres=100,
        node_thres=5,
        k_near=3,
        node_num=7,
        input_dim=3072,
        hidden_dim=1024,
        output_dim=96,
        drop_path_rate=0.2,
        patch_norm=True,
        post_norm=True,
        layer_scale=None,
        use_checkpoint=False,
        mlp_ratio=4.0,
        img_size=224,
        act_layer="GELU",
        infer_mode=False,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2**i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        self.depths = depths

        self.patch_embed = StemLayer(in_chans=in_chans, out_chans=self.embed_dim, act_layer="GELU", norm_layer="LN")
        self.outlineGraph = OLGraph(
            patch_size=graph_patch,
            distance_thres=distance_thres,
            node_thres=node_thres,
            k=k_near,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )

        self.gnns = [GAT(input_dim, hidden_dim, dim) for dim in output_dim]

        res0 = img_size / patch_size
        self.res = [int(res0), int(res0 // 2), int(res0 // 4), int(res0 // 8)]

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.infer_mode = infer_mode

        self.freq_embed = nn.ParameterList()
        for i in range(self.num_layers):
            self.freq_embed.append(nn.Parameter(torch.zeros(self.res[i], self.res[i], self.dims[i]), requires_grad=True))
            trunc_normal_(self.freq_embed[i], std=0.02)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            self.layers.append(
                self.make_layer(
                    res=self.res[i_layer],
                    dim=self.dims[i_layer],
                    depth=depths[i_layer],
                    drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                    use_checkpoint=use_checkpoint,
                    norm_layer=LayerNorm2d,
                    post_norm=post_norm,
                    layer_scale=layer_scale,
                    downsample=(
                        self.make_downsample(
                            self.dims[i_layer],
                            self.dims[i_layer + 1],
                            norm_layer=LayerNorm2d,
                        )
                        if (i_layer < self.num_layers - 1)
                        else nn.Identity()
                    ),
                    mlp_ratio=mlp_ratio,
                    infer_mode=infer_mode,
                )
            )

        self.convs = [nn.Conv1d(in_channels=node_num, out_channels=res, kernel_size=1) for res in self.res]

        self.classifier = nn.Sequential(
            LayerNorm2d(self.num_features),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(self.num_features, num_classes),
        )

        self.apply(self._init_weights)

    @staticmethod
    def make_downsample(dim=96, out_dim=192, norm_layer=LayerNorm2d):
        return nn.Sequential(
            # norm_layer(dim),
            # nn.Conv2d(dim, out_dim, kernel_size=2, stride=2)
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(out_dim),
        )

    @staticmethod
    def make_layer(
        res=14,
        dim=96,
        depth=2,
        drop_path=[0.1, 0.1],
        use_checkpoint=False,
        norm_layer=LayerNorm2d,
        post_norm=True,
        layer_scale=None,
        downsample=nn.Identity(),
        mlp_ratio=4.0,
        infer_mode=False,
        **kwargs,
    ):
        assert depth == len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(
                HeatBlock(
                    res=res,
                    hidden_dim=dim,
                    drop_path=drop_path[d],
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                    mlp_ratio=mlp_ratio,
                    post_norm=post_norm,
                    layer_scale=layer_scale,
                    infer_mode=infer_mode,
                )
            )

        return AdditionalInputSequential(
            *blocks,
            downsample,
        )

    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in VSSBlock, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, VSSBlock initialization is useless

        Conv2D is not intialized !!!
        """
        # print(m, getattr(getattr(m, "weight", nn.Identity()), "INIT", None), isinstance(m, nn.Linear), "======================")
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def infer_init(self):
        for i, layer in enumerate(self.layers):
            for block in layer[:-1]:
                block.op.infer_init_heat2d(self.freq_embed[i])
        del self.freq_embed

    def forward_features(self, x):
        #########################################################
        featuremap_folder = 'feat_map/pingpong'
        def normalize_image(image):
            img_min = image.min()
            img_max = image.max()
            normalized_img = (image - img_min) / (img_max - img_min)
            return normalized_img
        
        import matplotlib.pyplot as plt
        import os
        img = x.detach().cpu().float().numpy()[0][0]
        img = normalize_image(img)
        # plt.imshow(img)
        plt.imshow(img, cmap='gray') 
        plt.axis("off")
        plt.grid(False)
        plt.tight_layout()
        os.system(f'mkdir -p {featuremap_folder}')
        path1= f'{featuremap_folder}/orin.png'
        plt.savefig(path1,dpi=300)
        plt.close()
        layer_name = 0
        ############################################
        graph_data = self.outlineGraph(x) 
        # outline_data = self.outlineGraph(x)
        x = self.patch_embed(x)
        if self.infer_mode:
            for layer in self.layers:
                x = layer(x)
        else:
            for i, layer in enumerate(self.layers):
                patch_resolution = self.res[i]
                ol_feat, batch= self.gnns[i](graph_data[i])
                ol_feat, mask = to_dense_batch(ol_feat, batch)

                ol_feat = F.interpolate(ol_feat.transpose(1, 2), size=patch_resolution, mode='linear', align_corners=False).transpose(1, 2)
                ol_feat = ol_feat.unsqueeze(2).expand(-1, -1, patch_resolution, -1)

                x = layer(x, self.freq_embed[i], ol_feat)  # (B, C, H, W)
##################################################################################
                import numpy as np
                import matplotlib.pyplot as plt
                import torchvision.transforms.functional as TF
                folder_name= f'{featuremap_folder}/layername{str(layer_name)}'
                layer_name=layer_name+1
                
                feature_map = x.detach().cpu().float()[0]
                # feature_map_normalized = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
                # if layer_name<2:
                #     continue
                for channel in range(feature_map.size(0)):
                    fm= feature_map.mean(dim=0)
                    # fm = feature_map[0][channel]
                    # fm= feature_map[0].mean(dim=2)
                    # fm= feature_map[0, :, :, channel] ## all
                    # plt.imshow(fm.numpy().copy(), cmap='viridis')
                    channel_image = TF.resize(fm.unsqueeze(0).unsqueeze(0), size=(640, 640))[0][0].numpy().copy()
                    channel_image = normalize_image(channel_image)##归一化
                    channel_image = img*0.5+ channel_image*0.5 
                    plt.imshow(channel_image, cmap='viridis')
                    plt.axis("off")
                    plt.grid(False)
                    plt.tight_layout()
                    path2 = f'{folder_name}_mean_{int(time.time())}.png'
                    # path2 = f'{folder_name}/channel{channel}.png'
                    plt.savefig(path2,dpi=300)
                    plt.close()
                    break
##################################################################################

        return x

    def forward(self, x):
        device  = x.device
        for gnn in self.gnns:
            gnn.to(device)
        x = self.forward_features(x)
        # x = self.classifier(x)
        return x


if __name__ == "__main__":
    from fvcore.nn import flop_count_table, flop_count_str, FlopCountAnalysis

    model = GvHeat().cuda()
    input = torch.randn((2, 3, 640, 640), device=torch.device("cuda:0"))
    res = model(input)
    print(res)
    # analyze = FlopCountAnalysis(model, (input,))
    # print(flop_count_str(analyze))
