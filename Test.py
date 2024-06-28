from functools import partial
from itertools import repeat
import logging
import os
from collections import OrderedDict
import torchvision
import numpy as np
import scipy
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset,DataLoader
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, trunc_normal_
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
# from .registry import register_model
class VolumesDataset_test(Dataset):
    def __init__(self, dataSourcePath, totalTimesteps, nTimesteps_test, dim,
                 fileStartVal, fileIncrement, constVal, float32DataType=np.float32,
                 transform=None):
        self.dataSourcePath = dataSourcePath
        self.totalTimesteps = totalTimesteps  # totalTimesteps=100.
        self.nTimesteps_test = nTimesteps_test  # number of timesteps used for test.
        self.fileStartVal = fileStartVal
        self.fileIncrement = fileIncrement
        self.constVal = constVal
        self.float32DataType = float32DataType
        self.transform = transform
        self.dim = dim
        # correct: 2022.10.14.

    def __len__(self):
        return self.nTimesteps_test  # =30.
        # correct: 2022.10.14.

    # given an index, return 1 pair of (ct, mr).
    def __getitem__(self, index):  # index: [0, 29].
        # (i)if index is outside normal range.
        if index < 0 or index >= self.nTimesteps_test:
            print('index is outside the normal range.\n')
            return

        # (ii)convert index range from [0, 29] to [70, 99].
        index += (self.totalTimesteps - self.nTimesteps_test)  # index: [70, 99].

        # 1. at index, read original a pair of (volume_ct, volume_mr) .raw files.
        # (1.1)read original volume_ct.
        fileName = '%snorm_ct_enContrast.%.3d.raw' % (
        self.dataSourcePath, (self.fileStartVal + index * self.fileIncrement) / self.constVal)
        volume_ct = np.fromfile(fileName, dtype=self.float32DataType)
        # convert numpy ndarry to tensor.
        volume_ct = torch.from_numpy(volume_ct)
        # reshape.
        volume_ct = volume_ct.view([1, self.dim[0], self.dim[1], self.dim[2]])  # [channels, depth, height, width].

        # (1.2)read original volume_mr.
        fileName = '%snorm_mr_enContrast.%.3d.raw' % (
        self.dataSourcePath, (self.fileStartVal + index * self.fileIncrement) / self.constVal)
        volume_mr = np.fromfile(fileName, dtype=self.float32DataType)
        # convert numpy ndarry to tensor.
        volume_mr = torch.from_numpy(volume_mr)
        # reshape.
        volume_mr = volume_mr.view([1, self.dim[0], self.dim[1], self.dim[2]])  # [channels, depth, height, width].

        # 2. given volume_ct/_mr, crop them to get cropped volumes, for data augmentation.
        if self.transform:
            crop_ct, crop_mr = self.transform(volume_ct, volume_mr)

        # make sure crop_ct and crop_mr are the same size.
        assert crop_ct[0].shape == crop_mr[0].shape

        return crop_ct, crop_mr, index
        # correct: 2023.5.22.


def saveRawFile_6x(t, t_letter, volume):
    fileName = '%sMulti-scale_v2_2_%d/cvtGAN_%s_%.4d.raw' % (
        dataSavePath, final_train_step, t_letter, (fileStartVal + t * fileIncrement) / constVal)
    volume = volume.view(dim[0], dim[1], dim[2])
    # copy tensor from gpu to cpu.
    volume = volume.cpu()
    # convert tensor to numpy ndarray.
    volume = volume.detach().numpy()
    volume.astype('float32').tofile(fileName)

def crop(image, new_shape):
    '''
    Function for cropping an image tensor: Given an image tensor and the new shape,
    crops to the center pixels (assumes that the input's size and the new size are
    even numbers).
    Parameters:
        image: image tensor of shape (batch size, channels, depth, height, width)
        new_shape: a torch.Size object with the shape you want x to have
    '''
    middle_depth=image.shape[2] //2
    middle_height = image.shape[3] // 2
    middle_width = image.shape[4] // 2
    starting_depth=middle_depth-new_shape[2]//2
    final_depth=starting_depth+new_shape[2]
    starting_height = middle_height - new_shape[3] // 2
    final_height = starting_height + new_shape[3]
    starting_width = middle_width - new_shape[4] // 2
    final_width = starting_width + new_shape[4]
    cropped_image = image[:, :,starting_depth:final_depth, starting_height:final_height, starting_width:final_width]
    return cropped_image

def saveRawFile10(cur_step, res, t, l, volume):
    fileName = '%s%s_%.4d_%s_%d.raw' % (dataSavePath, res, t, l, cur_step)
    volume = volume.view(dim_crop[0], dim_crop[1], dim_crop[2])
    # copy tensor from gpu to cpu.
    volume = volume.cpu()
    # convert tensor to numpy ndarray.
    volume = volume.detach().numpy()
    volume.astype('float32').tofile(fileName)
# add on 2022.9.26.


def saveRawFile2(cur_step, t, volume):
    fileName = '%sH_%s_%d.raw' % (dataSavePath, t, cur_step)
    volume = volume.view(dim_crop[0], dim_crop[1], dim_crop[2])
    # copy tensor from gpu to cpu.
    volume = volume.cpu()
    # convert tensor to numpy ndarray.
    volume = volume.detach().numpy()
    volume.astype('float32').tofile(fileName)

def saveModel(cur_step):
    # if save_model:
    fileName = "%scvtGAN_%d.pth" % (dataSavePath, cur_step)
    torch.save({'gen': gen.state_dict(),
                'gen_opt': gen_opt.state_dict(),
                }, fileName)
class MyRandomCrop3D3(object):
    def __init__(self, volume_sz, cropVolume_sz):
        c, d, h, w = volume_sz
        assert (d, h, w) >= cropVolume_sz
        self.volume_sz = tuple((d, h, w))
        self.cropVolume_sz = tuple(cropVolume_sz)

    def __call__(self, volume_ct, volume_mr):
        slice_dhw = [self._get_slice(i, k) for i, k in zip(self.volume_sz, self.cropVolume_sz)]
        return self._crop(volume_ct, volume_mr, *slice_dhw)

    @staticmethod
    def _get_slice(volume_sz, cropVolume_sz):
        try:
            lower_bound = torch.randint(volume_sz - cropVolume_sz, (1,)).item()
            return lower_bound, lower_bound + cropVolume_sz
        except:
            return (None, None)

    @staticmethod
    def _crop(volume_ct, volume_mr, slice_d, slice_h, slice_w):
        # print(f"slice_d[0]:slice_d[1], slice_h[0]:slice_h[1], slice_w[0]:slice_w[1]: {slice_d[0], slice_d[1], slice_h[0], slice_h[1], slice_w[0], slice_w[1]}")
        return volume_ct[:, slice_d[0]:slice_d[1], slice_h[0]:slice_h[1], slice_w[0]:slice_w[1]], \
               volume_mr[:, slice_d[0]:slice_d[1], slice_h[0]:slice_h[1], slice_w[0]:slice_w[1]]


class VolumesDataset(Dataset):
    def __init__(self, dataSourcePath, nTimesteps_train, dim,
                 fileStartVal, fileIncrement, constVal, float32DataType=np.float32,
                 transform=None):
        self.dataSourcePath = dataSourcePath
        self.nTimesteps_train = nTimesteps_train  # number of timesteps used for training.
        self.fileStartVal = fileStartVal
        self.fileIncrement = fileIncrement
        self.constVal = constVal
        self.float32DataType = float32DataType
        self.transform = transform
        self.dim = dim

    def __len__(self):
        return self.nTimesteps_train  # =70.

    # given an index, return a pair of (ct, mr).
    def __getitem__(self, index):  # index: [0, 69].
        # if index is outside normal range.
        if index < 0 or index >= self.nTimesteps_train:
            print('index is outside the normal range.\n')
            return


        # 1. at index, read original a pair of (volume_ct, volume_mr) .raw files.
        # (1.1)read original volume_ct.
        fileName = '%snorm_ct_enContrast.%.3d.raw' % (self.dataSourcePath, (self.fileStartVal + index * self.fileIncrement) / self.constVal)
        volume_ct = np.fromfile(fileName, dtype=self.float32DataType)
        # convert numpy ndarry to tensor.
        volume_ct = torch.from_numpy(volume_ct)
        # reshape.
        volume_ct = volume_ct.view([1, self.dim[0], self.dim[1], self.dim[2]])  # [channels, depth, height, width].

        # (1.2)read original volume_mr.
        fileName = '%snorm_mr_enContrast.%.3d.raw' % (self.dataSourcePath, (self.fileStartVal + index * self.fileIncrement) / self.constVal)
        volume_mr = np.fromfile(fileName, dtype=self.float32DataType)
        # convert numpy ndarry to tensor.
        volume_mr = torch.from_numpy(volume_mr)
        # reshape.
        volume_mr = volume_mr.view([1, self.dim[0], self.dim[1], self.dim[2]])  # [channels, depth, height, width].


        # 2. given volume_ct/_mr, crop them to get cropped volumes, for data augmentation.
        if self.transform:
            crop_ct, crop_mr = self.transform(volume_ct, volume_mr)

        #make sure crop_ct, crop_mr are the same size.
        assert crop_ct.shape == crop_mr.shape


        return crop_ct, crop_mr, index
        #correct: 2023.5.22.

#MSA与FFN之间的layerNorm操作
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

#线性层之间的GELU激活函数
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class Squeeze_excitation_block(nn.Module):
    def __init__(self, input_channels, ratio=8):
        super(Squeeze_excitation_block, self).__init__()

        self.avgpool1 = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(input_channels, input_channels//ratio),
            nn.ReLU(inplace=True),
            nn.Linear(input_channels//ratio, input_channels),
            nn.PReLU()  #nn.Sigmoid()   #notice: may need to change back.
        )

    def forward(self, x):
        batch_size, num_channels, _, _, _ = x.size()
        y = self.avgpool1(x).view(batch_size, num_channels)
        y = self.fc1(y).view(batch_size, num_channels, 1, 1, 1)

        return x*y.expand_as(x)

#EfficientChannelAttention:An improvement compared to the Squeeze excitation block
class EfficientChannelAttention(nn.Module):
    def __init__(self,channel,b=1,gamma=2):
        super(EfficientChannelAttention,self).__init__()

        t=int(abs((np.log2(channel)+b)/gamma))
        k=t if t % 2 else t+1
        self.avg_pool=nn.AdaptiveAvgPool3d(1)
        self.conv1=nn.Conv2d(1,1,kernel_size=k,padding=int(k/2),bias=False)
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):#shape=1,16,depth,height,width
        y=self.avg_pool(x)#shape=1,16,1,1,1
        y=self.conv1(y.squeeze(-1).transpose(-1,-3)).transpose(-1,-3).unsqueeze(-1)
        #shape=1,16,1,1--->shape=1,1,1,16--conv-->shape=1,1,1,16--->shape=1,16,1,1--->shape=1,16,1,1,1
        y=self.sigmoid(y)
        return x*y.expand_as(x)
#线性层
class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

#MSA层
class Attention(nn.Module):

    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 method='dw_bn',
                 kernel_size=3,
                 stride_kv=2,  # 用于k和v的project步长大小
                 stride_q=1,  # 用于q的project步长大小
                 padding_kv=1,  # 与stride同理
                 padding_q=1,
                 with_cls_token=False,
                 **kwargs
                 ):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim_out
        self.num_heads = num_heads
        # head_dim = self.qkv_dim // num_heads
        self.scale = dim_out ** -0.5
        self.with_cls_token = with_cls_token

        self.conv_proj_q = self._build_projection(
            dim_in, dim_out, kernel_size, padding_q,
            stride_q, 'linear' if method == 'avg' else method
        )
        self.conv_proj_k = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv, method
        )
        self.conv_proj_v = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv, method
        )

        self.proj_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out//4, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    #projection操作
    def _build_projection(self,
                          dim_in,
                          dim_out,
                          kernel_size,
                          padding,
                          stride,
                          method):
        if method == 'dw_bn':  
            proj = nn.Sequential(OrderedDict([
                ('depth_conv', nn.Conv3d(
                    dim_in,
                    dim_in,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=False,
                    groups=dim_in
                )),
                 ('point_conv',nn.Conv3d(
                    dim_in,
                    dim_out,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=1
                )),
                ('bn', nn.BatchNorm3d(dim_in)),
                ('rearrage', Rearrange('b c d h w -> b (d h w) c')),
            ]))
        elif method == 'avg':
            proj = nn.Sequential(OrderedDict([
                ('avg', nn.AvgPool3d(
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    ceil_mode=True
                )),
                ('rearrage', Rearrange('b c d h w-> b (d h w) c')),
            ]))
        elif method == 'linear':
            proj = None
        else:
            raise ValueError('Unknown method ({})'.format(method))

        return proj
    def conv_reduceChannel(self,x):
        b = x.shape[0]
        d = x.shape[1]
        t = x.shape[2]
        h = x.shape[3]
        x = x.view(b, d, t, h)
        
        i_padding=False
        if (t % 2 != 0):
            x=F.pad(x,(0,0,1,0))
            i_padding = True
        # print('x:', x.shape)
        conv_1 = nn.Conv2d(in_channels=d, out_channels=d // 2, kernel_size=1).cuda()
        x = conv_1(x)
       
        if (i_padding):
            x = x.reshape(b, h, (t + 1) // 2, d)
        else:
            x = x.reshape(b, h, t // 2, d)
        # print("After X1:", x.shape)
        return x
    def forward_conv(self, x, d, h, w):

        x = rearrange(x, 'b (d h w) c -> b c d h w', h=h, w=w, d=d)
        # print('x:',x.shape)
        if self.conv_proj_q is not None:
            q = self.conv_proj_q(x)
            # print("Conv_proj_q:",q.shape)
        else:
            q = rearrange(x, 'b c d h w -> b (d h w) c')  # 在hwd三个维度合并并将c和(hwd)维度调换,producing token sequences

        if self.conv_proj_k is not None:
            k = self.conv_proj_k(x)
           
        else:
            k = rearrange(x, 'b c d h w -> b (d h w) c')

        if self.conv_proj_v is not None:
            v = self.conv_proj_v(x)
            
        else:
            v = rearrange(x, 'b c d h w -> b (d h w) c')

        return q, k, v

    def forward(self, x, d, h, w):
        if (
                self.conv_proj_q is not None
                or self.conv_proj_k is not None
                or self.conv_proj_v is not None
        ):
            q, k, v = self.forward_conv(x, d, h, w)


       
        q = rearrange(self.proj_q(q), 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(self.proj_k(k), 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(self.proj_v(v), 'b t (h d) -> b h t d', h=self.num_heads)
    
        avg_pool = nn.AvgPool3d(2,stride=2)
        q=avg_pool(q)
        k=avg_pool(k)
        v=avg_pool(v)
       
        attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale
        attn = F.softmax(attn_score, dim=-1)
        attn = self.attn_drop(attn)
     
        x = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
        #skip connection add original Q,K,V to the final x
        # x=x+q
        # print("before rearrange x shape",x.shape)
        x = rearrange(x, 'b h t d -> b t (h d)')
        
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    @staticmethod
    def compute_macs(module, input, output):
        # T: num_token
        # S: num_token
        input = input[0]
        flops = 0

        _, T, C = input.shape
        H = W = int(np.sqrt(T - 1)) if module.with_cls_token else int(np.sqrt(T))

        H_Q = H / module.stride_q
        W_Q = H / module.stride_q
        T_Q = H_Q * W_Q + 1 if module.with_cls_token else H_Q * W_Q

        H_KV = H / module.stride_kv
        W_KV = W / module.stride_kv
        T_KV = H_KV * W_KV + 1 if module.with_cls_token else H_KV * W_KV

        # C = module.dim
        # S = T
        # Scaled-dot-product macs
        # [B x T x C] x [B x C x T] --> [B x T x S]
        # multiplication-addition is counted as 1 because operations can be fused
        flops += T_Q * T_KV * module.dim
        # [B x T x S] x [B x S x C] --> [B x T x C]
        flops += T_Q * module.dim * T_KV

        if (
                hasattr(module, 'conv_proj_q')
                and hasattr(module.conv_proj_q, 'conv')
        ):
            params = sum(
                [
                    p.numel()
                    for p in module.conv_proj_q.conv.parameters()
                ]
            )
            flops += params * H_Q * W_Q

        if (
                hasattr(module, 'conv_proj_k')
                and hasattr(module.conv_proj_k, 'conv')
        ):
            params = sum(
                [
                    p.numel()
                    for p in module.conv_proj_k.conv.parameters()
                ]
            )
            flops += params * H_KV * W_KV

        if (
                hasattr(module, 'conv_proj_v')
                and hasattr(module.conv_proj_v, 'conv')
        ):
            params = sum(
                [
                    p.numel()
                    for p in module.conv_proj_v.conv.parameters()
                ]
            )
            flops += params * H_KV * W_KV

        params = sum([p.numel() for p in module.proj_q.parameters()])
        flops += params * T_Q
        params = sum([p.numel() for p in module.proj_k.parameters()])
        flops += params * T_KV
        params = sum([p.numel() for p in module.proj_v.parameters()])
        flops += params * T_KV
        params = sum([p.numel() for p in module.proj.parameters()])
        flops += params * T

        module.__flops__ += flops


class Block(nn.Module):

    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 **kwargs):
        super().__init__()

        self.norm1 = norm_layer(dim_in)
        self.attn = Attention(
            dim_in, dim_out, num_heads, qkv_bias, attn_drop, drop,
            **kwargs
        )

        self.drop_path = DropPath(drop_path) \
            if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim_out)
        dim_mlp_hidden = int(dim_out * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim_out,
            hidden_features=dim_mlp_hidden,
            act_layer=act_layer,
            drop=drop
        )

    def forward(self, x, d, h, w):
        res = x
        # print("X",x.shape)

        x = self.norm1(x)
        #Transformer Attention计算
        attn = self.attn(x, d, h, w)
        attn=self.drop_path(attn)

        target_size=x.shape
        # print("Target Size",target_size[1:])
        attn=F.interpolate(attn.unsqueeze(0),size=target_size[1:],mode='bilinear',align_corners=False)
        # print("After Interpolate:",attn.shape)
        x = res + attn#attn token sequence比原始X小一半
        x=x.squeeze(0)
        # print("Squeeze X:",x.shape)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # print("Final X:",x.shape)
        return x


class ConvEmbed(nn.Module):
    """ Image to Conv Embedding

    """

    def __init__(self,
                 kernel_size=3,
                 in_chans=3,
                 embed_dim=64,
                 stride=4,
                 padding=2,
                 norm_layer=None):
        super().__init__()
        #projection
        self.proj = nn.utils.spectral_norm(nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        ))
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.proj(x)

        B, C, D, H, W = x.shape

        x = rearrange(x, 'b c d h w -> b (d h w) c')
        if self.norm:
            x = self.norm(x)
        x = rearrange(x, 'b (d h w) c -> b c d h w', h=H, w=W, d=D)

        return x


#
class TransposeConvEmbed(nn.Module):
    """ Image to Conv Embedding

    """

    def __init__(self,
                 kernel_size=3,
                 in_chans=3,
                 embed_dim=64,
                 stride=4,
                 padding=2,
                 norm_layer=None):
        super().__init__()
        #revised on 2023.12.5 replace ConvTranspose3d with upsample operation,since it's better
        self.proj = nn.utils.spectral_norm(nn.ConvTranspose3d(
            in_chans, embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        ))
        self.proj2=nn.Sequential(
            nn.Upsample(scale_factor=stride,mode="trilinear",align_corners=True),
            nn.utils.spectral_norm(nn.Conv3d(in_chans,embed_dim,kernel_size=3,stride=1,padding=1)),
        )
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        # x = self.proj(x)
        x = self.proj2(x)
        B, C, H, W, D = x.shape
        x = rearrange(x, 'b c h w d -> b (h w d) c')
        if self.norm:
            x = self.norm(x)
        x = rearrange(x, 'b (h w d) c -> b c h w d', h=H, w=W, d=D)

        return x


#
class VisionTransformer_up(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 kernel_size=16,
                 stride=16,
                 padding=0,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='xavier',
                 **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.rearrage = None

        self.patch_embed = TransposeConvEmbed(
            # img_size=img_size,
            # patch_size=patch_size,
            kernel_size=kernel_size,
            in_chans=in_chans,
            stride=stride,
            padding=padding,
            embed_dim=embed_dim,
            norm_layer=norm_layer
        )

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        blocks = []
        for j in range(depth):
            blocks.append(
                Block(
                    dim_in=embed_dim,
                    dim_out=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    **kwargs
                )
            )
        self.blocks = nn.ModuleList(blocks)

        if init == 'xavier':
            self.apply(self._init_weights_xavier)
        else:
            self.apply(self._init_weights_trunc_normal)

    def _init_weights_trunc_normal(self, m):
        if isinstance(m, nn.Linear):
            logging.info('=> init weight of Linear from trunc norm')
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                logging.info('=> init bias of Linear to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_weights_xavier(self, m):
        if isinstance(m, nn.Linear):
            logging.info('=> init weight of Linear from xavier uniform')
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                logging.info('=> init bias of Linear to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        B, C, H, W, D = x.size()

        x = rearrange(x, 'b c h w d -> b (h w d) c')

        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x, H, W, D)

        x = rearrange(x, 'b (h w d) c -> b c h w d', h=H, w=W, d=D)

        return x
    #


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 kernel_size=16,
                 stride=16,
                 padding=0,
                 in_chans=3,
                 embed_dim=768,#Transformer 中嵌入向量的维度，也是注意力模型的输出特征维度
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4., 
                 #隐藏层维度与嵌入维度的比例，使用一个全连接的多层感知机(MLP)来对Transformer中的特征进行线性变换
                 qkv_bias=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='xavier',
                 **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.rearrage = None
        ##ConvEmbed equals to patch embedding
        self.patch_embed = ConvEmbed(
            # img_size=img_size,
            # patch_size=patch_size,
            kernel_size=kernel_size,
            in_chans=in_chans,
            stride=stride,
            padding=padding,
            embed_dim=embed_dim,#equals to out_chans
            norm_layer=norm_layer
        )

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        blocks = []
        for j in range(depth):
            blocks.append(
                Block(
                    dim_in=embed_dim,
                    dim_out=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    **kwargs
                )
            )
        self.blocks = nn.ModuleList(blocks)

        if init == 'xavier':
            self.apply(self._init_weights_xavier)
        else:
            self.apply(self._init_weights_trunc_normal)

    def _init_weights_trunc_normal(self, m):
        if isinstance(m, nn.Linear):
            logging.info('=> init weight of Linear from trunc norm')
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                logging.info('=> init bias of Linear to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_weights_xavier(self, m):
        if isinstance(m, nn.Linear):
            logging.info('=> init weight of Linear from xavier uniform')
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                logging.info('=> init bias of Linear to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # print("X's shape before unsqueeze:"+str(x.shape))
        # x = torch.unsqueeze(x, dim=0)
        # print("X's shape after unsqueeze:" + str(x.shape))
        x = self.patch_embed(x)
        # B, C, H, W, D = x.size()
        B, C, D, H, W = x.size()
        x = rearrange(x, 'b c d h w -> b (d h w) c')

        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x, D, H, W)

        x = rearrange(x, 'b (d h w) c -> b c d h w', h=H, w=W, d=D)

        return x
    #
class SelfAttentionBlock(nn.Module):
    def __init__(self, input_channels):
        super(SelfAttentionBlock, self).__init__()

        # add on 2022.10.12.
        out_channels_mult = 2
        # add on 2022.10.12.

        # construct the conv layers (here query_conv, key_conv, value_conv must be conv3d).
        self.query_conv = nn.Conv3d(in_channels=input_channels, out_channels=input_channels // out_channels_mult,
                      kernel_size=(1, 1, 1))  # [B, C', depth, height, width].
        self.key_conv = nn.Conv3d(in_channels=input_channels, out_channels=input_channels // out_channels_mult,
                      kernel_size=(1, 1, 1))  # [B, C', depth, height, width].
        self.value_conv = nn.Conv3d(in_channels=input_channels, out_channels=input_channels,
                                                           kernel_size=(1, 1, 1))  # [B, C'', depth, height, width].

        # initialize gamma as 0.
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax1 = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, depth, height, width = x.size()  # [batch_size, channels, depth, height, width].

        proj_query = self.query_conv(x).view(B, -1, depth * height * width).permute(0, 2, 1)  # [B, N, C'].
        proj_key = self.key_conv(x).view(B, -1, depth * height * width)  # [B, C', N].
        energy = torch.bmm(proj_query, proj_key)  # [B, N, N].
        attention_map = self.softmax1(energy)  # [B, N, N].

        proj_value = self.value_conv(x).view(B, -1, depth * height * width)  # [B, C'', N].
        attention_weight = torch.bmm(proj_value, attention_map.permute(0, 2, 1))  # [B, C'', N].
        attention_weight = attention_weight.view(B, C, depth, height, width)  # [B, C'', depth, height, width].

        # add attention_with to input.
        output = self.gamma * attention_weight + x  # output: [batch, channels, depth, height, width].

        return output  # , attention_map
        # correct: 2022.9.12.
# add on 2022.9.9.
class TransformBlockAndSelfAttentionCombinationBlock(nn.Module):
    def __init__(self, input_channels, output_channels, scale_factor=(2,2,2)):
        super(TransformBlockAndSelfAttentionCombinationBlock, self).__init__()

        self.trans_block1 = TransformBlock(input_channels, output_channels//2)
        self.maxpool1 = nn.Upsample(scale_factor=(1.0/scale_factor[0],1.0/scale_factor[1],1.0/scale_factor[2]), mode='trilinear', align_corners=True)
        self.sa1 = SelfAttentionBlock(input_channels)
        self.up1 = nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=True)
        self.conv1 = nn.Conv3d(input_channels, output_channels//2, kernel_size=(1,1,1))

    def forward(self, x):
        #path1.
        x_path1 = self.trans_block1(x)

        #path2.
        # x_path2 = self.maxpool1(x)
        # x_path2 = self.sa1(x_path2)
        # x_path2 = self.up1(x_path2)
        x_path2 = self.conv1(x)

        #multiply path1 and path2 together.
        #x = x_path1 + x_path2   #x_path1 * x_path2
        x = torch.cat([x_path1, x_path2], dim=1)

        return x
        #correct: 2023.4.12.
class TransformBlock(nn.Module):
    def __init__(self, input_channels, output_channels, use_n=True, kernel_size=(3,3,3), padding=(1,1,1)):
        super(TransformBlock, self).__init__()

        self.conv1 = nn.utils.spectral_norm(nn.Conv3d(input_channels, output_channels, kernel_size=kernel_size, padding=padding))
        self.conv2 = nn.utils.spectral_norm(nn.Conv3d(output_channels, output_channels, kernel_size=kernel_size, padding=padding))
        self.use_n = use_n
        if self.use_n:
            self.instancenorm1 = nn.InstanceNorm3d(output_channels)  #nn.InstanceNorm3d(input_channels)
            self.instancenorm2 = nn.InstanceNorm3d(output_channels)  #nn.InstanceNorm3d(input_channels)
        self.activation1 = nn.ReLU(inplace=True)
        self.activation2 = nn.ReLU(inplace=True)

        self.conv3 = nn.utils.spectral_norm(nn.Conv3d(output_channels, output_channels, kernel_size=kernel_size, padding=padding))
        self.conv4 = nn.utils.spectral_norm(nn.Conv3d(input_channels, output_channels, kernel_size=kernel_size, padding=padding))
        self.activation_final = nn.ReLU(inplace=True)

    def forward(self, x):
        original_x = x.clone()

        #path1.
        x = self.conv1(x)
        if self.use_n:
            x = self.instancenorm1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        if self.use_n:
            x = self.instancenorm2(x)
        x = self.activation2(x)
        x = self.conv3(x)

        #path2.
        original_x = self.conv4(original_x)

        #add two paths together.
        y = self.activation_final(x + original_x)

        return y
class ResPath_v2(nn.Module):
    def __init__(self, input_channels, output_channels, length):
        super(ResPath_v2, self).__init__()
        self.length = length

        self.blocks = nn.Sequential(*[TransformBlock(input_channels, output_channels) for i in range(length)])

    def forward(self, x):
        if self.length == 0:
            output = x
        else:
            output = self.blocks(x)

        return output

class ResBlock3D(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.conv = nn.Conv3d(in_chan, out_chan, kernel_size=1, stride=1, padding=0)
        self.transconv = nn.ConvTranspose3d(in_chan, out_chan, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.conv(x)
# class RDBlock(nn.Module,in_channels=hidden_channels):
#     def __init__(self):
#         super().__init__()
#         self.conv1=nn.Conv3d(in_channels=hidden_channels,out_channels=hidden_channels*2,kernel_size=3,stride=2,padding=1)
#         self.insNorm=nn.InstanceNorm3d(hidden_channels*2)
#         self.activation=nn.GELU(0.2)
#         self.conv2=nn.Conv3d(in_channels=hidden_channels*2,out_channels=hidden_channels*4,kernel_size=3,stride=2,padding=1)
#         self.conv3=nn.Conv3d(in_channels=hidden_channels*4,out_channels=hidden_channels*4,kernel_size=1,stride=1,padding=0)
#     def forward(self,x):
#         x1=x
#         x=self.conv1(x)
#         x=self.conv2(x+x1)
#         x2=x
#
#         return x
class HighFeaExtraBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Sequential(
            nn.utils.spectral_norm(nn.Conv3d(in_channels=1,out_channels=hidden_channels,kernel_size=3,stride=2,padding=1)),
            nn.InstanceNorm3d(hidden_channels),
            nn.ReLU(),
        )
        self.conv2=nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv3d(in_channels=hidden_channels,out_channels=hidden_channels*2,kernel_size=3,stride=2,padding=1)
            ),
            nn.InstanceNorm3d(hidden_channels*2),
            nn.ReLU(),
        )
        self.conv3=nn.Sequential(
            nn.utils.spectral_norm(nn.Conv3d(in_channels=hidden_channels*2,out_channels=hidden_channels*4,kernel_size=(2,2,1),stride=(2,2,1),padding=0)),
            nn.InstanceNorm3d(hidden_channels * 4),
            nn.ReLU(),
        )
        self.conv4=nn.Sequential(
            nn.utils.spectral_norm(nn.Conv3d(in_channels=hidden_channels*4,out_channels=hidden_channels*8,kernel_size=(1,1,2),stride=(1,1,3),padding=0)),
            nn.InstanceNorm3d(hidden_channels * 8),
            nn.ReLU(),
        )
        self.conv5=nn.Sequential(
            nn.utils.spectral_norm(nn.Conv3d(in_channels=hidden_channels*8,out_channels=hidden_channels*16,kernel_size=(2,2,1),stride=(2,2,1),padding=0)),
            nn.InstanceNorm3d(hidden_channels * 16),
            nn.ReLU(),
        )
        self.conv6=nn.Sequential(
            nn.utils.spectral_norm(nn.Conv3d(in_channels=hidden_channels*16,out_channels=hidden_channels*16,kernel_size=1,stride=1,padding=0)),
            nn.InstanceNorm3d(hidden_channels*16),
            nn.ReLU(),
        )
        self.Tconv5=nn.Sequential(
            nn.ConvTranspose3d(in_channels=hidden_channels*16,out_channels=hidden_channels*8,kernel_size=(2,2,1),stride=(2,2,1),padding=0),
            nn.InstanceNorm3d(hidden_channels * 8),
            nn.ReLU(),
        )
        self.Tconv4 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=hidden_channels * 8, out_channels=hidden_channels * 4,
                               kernel_size=(1,1,3), stride=(1,1,3), padding=0),
            nn.InstanceNorm3d(hidden_channels * 4),
            nn.ReLU(),
        )
        self.Tconv3=nn.Sequential(
            nn.ConvTranspose3d(in_channels=hidden_channels * 4, out_channels=hidden_channels * 2,
                               kernel_size=(2,2,1),stride=(2,2,1),padding=0),
            nn.InstanceNorm3d(hidden_channels * 2),
            nn.ReLU(),
        )
        self.Tconv2=nn.Sequential(
            nn.ConvTranspose3d(in_channels=hidden_channels * 2, out_channels=hidden_channels ,
                               kernel_size=2,stride=2,padding=0),
            nn.InstanceNorm3d(hidden_channels ),
            nn.ReLU(),
        )
#
    def forward(self, x):
        #x=(1,1,80,112,84)
        en1=self.conv1(x)#(1,8,40,56,42)
        en2=self.conv2(en1)#(1,16,20,28,21)
        en3=self.conv3(en2)#(1,32,10,14,21)
        en4=self.conv4(en3)#(1,64,10,14,7)
        en5=self.conv5(en4)#(1,128,5,7,7)
        en6=self.conv6(en5)#(1,128,5,7,7)
        de5=self.Tconv5(en5+en6)#(1,64,10,14,7)
        # print("de5 size",de5.shape)
        de4=self.Tconv4(de5+en4)#(1,32,10,14,21)
        # print("de4 size", de4.shape)
        # print("en3 size", en3.shape)
        de3=self.Tconv3(de4+en3)#(1,16,20,28,21)
        # print("de3 size",de3.shape)
        de2=self.Tconv2(de3+en2)#(1,8,40,56,42)
        # print("de2 size",de2.shape)

        return de2+en1#(1,8,40,56,42)

class HighFeaExtraBlockV2(nn.Module):
    def __init__(self):
        super().__init__()
        #######Encoder#######
        #layer x1
        self.featuremap1=nn.Conv3d(in_channels=1,out_channels=hidden_channels,kernel_size=3,padding=1)
        self.maxpool1=nn.MaxPool3d(kernel_size=(3,3,3),stride=(2,2,2),padding=(1,1,1))
        self.avgpool1=nn.AvgPool3d(kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1))
        self.res_path1_decoder1=ResPath_v2(hidden_channels,hidden_channels,length=1)
        self.eca1 = EfficientChannelAttention(hidden_channels)
        #layer x2
        self.trans_block2=TransformBlock(hidden_channels,hidden_channels*2)
        self.maxpool2=nn.MaxPool3d(kernel_size=(3,3,3),stride=(2,2,2),padding=(1,1,1))
        self.avgpool2=nn.AvgPool3d(kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1))
        self.res_path2_decoder1=ResPath_v2(hidden_channels*2,hidden_channels*2,length=2)
        self.eca2 = EfficientChannelAttention(hidden_channels*2)
        #layer x3
        self.trans_block3=TransformBlock(hidden_channels*2,hidden_channels*4)
        self.maxpool3=nn.MaxPool3d(kernel_size=(2,2,1),stride=(2,2,1),padding=0)
        self.avgpool3 = nn.AvgPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1), padding=0)
        self.res_path3_decoder1=ResPath_v2(hidden_channels*4,hidden_channels*4,length=3)
        self.eca3 = EfficientChannelAttention(hidden_channels*4)
        #layer x4
        self.trans_block4=TransformBlock(hidden_channels*4,hidden_channels*8)
        self.maxpool4=nn.MaxPool3d(kernel_size=(1,1,2),stride=(1,1,3),padding=0)
        self.avgpool4=nn.AvgPool3d(kernel_size=(1,1,2), stride=(1,1,3), padding=0)
        self.res_path4_decoder1=ResPath_v2(hidden_channels*8,hidden_channels*8,length=4)
        self.eca4 = EfficientChannelAttention(hidden_channels*8)
    
        #######Decoder#######
        #layer x1
        self.com1=TransformBlockAndSelfAttentionCombinationBlock(hidden_channels*8,hidden_channels*8,scale_factor=(2,2,1))
        self.up1=nn.Upsample(scale_factor=(1,1,3),mode="trilinear",align_corners=True)
        #layer x2
        self.com2=TransformBlockAndSelfAttentionCombinationBlock(hidden_channels*16,hidden_channels*4,scale_factor=(2,2,3))
        self.up2=nn.Upsample(scale_factor=(2,2,1),mode="trilinear",align_corners=True)
        #layer x3
        self.com3=TransformBlockAndSelfAttentionCombinationBlock(hidden_channels*8,hidden_channels*2,scale_factor=(2,2,3))
        self.up3=nn.Upsample(scale_factor=(2,2,2),mode="trilinear",align_corners=True)
        #layer x4
        self.com4=TransformBlockAndSelfAttentionCombinationBlock(hidden_channels*4,hidden_channels,scale_factor=(2,2,2))
        self.up4=nn.Upsample(scale_factor=(2,2,2),mode="trilinear",align_corners=True)

    def forward(self, x):
        #x:[80,112,84,1]
        #encoder part#
        #layer x1
        x1=self.featuremap1(x)#[80,112,84,8]
        p1=self.maxpool1(x1)#[40,56,42,8]
        x1_decoder1=self.res_path1_decoder1(x1)#[80,112,84,8]
        p1=self.eca1(p1)
        #layer x2
        x2=self.trans_block2(p1)#[40,56,42,16]
        p2=self.maxpool2(x2)#[20,28,21,16]
        x2_decoder1=self.res_path2_decoder1(x2)#[40,56,42,16]
        p2 = self.eca2(p2)
        #layer x3
        x3=self.trans_block3(p2)#[20,28,21,32]
        p3=self.maxpool3(x3)#[10,14,21,32]
        x3_decoder1=self.res_path3_decoder1(x3)#[20,28,21,32]
        p3 = self.eca3(p3)
        #layer x4
        x4=self.trans_block4(p3)#[10,14,21,64]
        p4=self.maxpool4(x4)#[10,14,7,64]
        x4_decoder1=self.res_path4_decoder1(x4)#[10,14,21,64]
        p4 = self.eca4(p4)
        #decoder part#
        #layer x1
        u1=self.com1(p4)#[10,14,7,64]
        u1=self.up1(u1)#[10,14,21,64]
        u1=torch.cat([u1,x4_decoder1],dim=1)#[10,14,21,128]
        #layer x2
        u2=self.com2(u1)#[10,14,21,32]
        u2=self.up2(u2)#[20,28,21,32]
        u2=torch.cat([u2,x3_decoder1],dim=1)#[20,28,21,64]
        #layer x3
        u3=self.com3(u2)#[20,28,21,16]
        u3=self.up3(u3)#[40,56,42,16]
        u3=torch.cat([u3,x2_decoder1],dim=1)#[40,56,42,32]
        #layer x4
        u4=self.com4(u3)#[40,56,42,8]
        u4=self.up4(u4)#[80,112,84,8]
        u4=torch.cat([u4,x1_decoder1],dim=1)#[80,112,84,16]
        return u4
class Generator(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        #High Res Part
        self.HighResBlock=nn.Sequential(HighFeaExtraBlockV2())
      
        self.upsample1 = nn.Upsample(scale_factor=(2,2,2), mode='trilinear',
                                     align_corners=True)
        #Low Res Part
        # layer0 1->64
        # 64*64*64
        self.avgpool = nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder_layer0_down = nn.Sequential(
            # nn.Conv3d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.utils.spectral_norm(nn.Conv3d(1, hidden_channels, kernel_size=1, stride=1, padding=0)),
            #reverse the batchnorm and activation func
            nn.InstanceNorm3d(hidden_channels),
            nn.ReLU(),
        )
   
        self.encoder_layer1_down = nn.Sequential(
            VisionTransformer(kernel_size=3, stride=2, padding=1, in_chans=hidden_channels,
                              embed_dim=hidden_channels * 2,
                              depth=1, num_heads=4, mlp_ratio=4, ))
     
        self.encoder_layer2_down = nn.Sequential(
            VisionTransformer(kernel_size=(2, 2, 1), stride=(2, 2, 1), padding=0, in_chans=hidden_channels * 2,
                              embed_dim=hidden_channels * 4,
                              depth=2, num_heads=4, mlp_ratio=4))

     
        self.encoder_layer3_down = nn.Sequential(
            VisionTransformer(kernel_size=2, stride=(2, 2, 3), padding=0, in_chans=hidden_channels * 4,
                              embed_dim=hidden_channels * 8,
                              depth=2, num_heads=4, mlp_ratio=4))
  
        self.decoder_layer1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv3d(hidden_channels * 8, hidden_channels * 8, kernel_size=1, stride=1, padding=0)),
            nn.InstanceNorm3d(hidden_channels * 8),
            nn.ReLU(),

        )
        # self.resup1 = ResBlock3D(512, 512)
        self.resup1 = ResBlock3D(hidden_channels * 8, hidden_channels * 8)
        # self.decoder_layer1_up = VisionTransformer_up(kernel_size=(2,2,3), stride=(2,2,3), in_chans=512, embed_dim=256, depth=2,
        #                                               num_heads=4)
        self.decoder_layer1_up = VisionTransformer_up(kernel_size=(2, 2, 3), stride=(2, 2, 3),
                                                      in_chans=hidden_channels * 8,
                                                      embed_dim=hidden_channels * 4, depth=2, drop_path_rate=0.2,
                                                      num_heads=4)

        self.decoder_layer2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv3d(hidden_channels * 8, hidden_channels * 4, kernel_size=3, stride=1, padding=1)),
            nn.Dropout3d(0.2),
            nn.InstanceNorm3d(hidden_channels * 4),
            nn.ReLU(),

        )
        # self.resup2 = ResBlock3D(512, 256)
        self.resup2 = ResBlock3D(hidden_channels * 8, hidden_channels * 4)
        # self.decoder_layer2_up = VisionTransformer_up(kernel_size=(2,2,1), stride=(2,2,1), in_chans=256, embed_dim=128, depth=1,
        #                                               num_heads=4)
        self.decoder_layer2_up = VisionTransformer_up(kernel_size=(2,2,1), stride=(2,2,1), in_chans=hidden_channels*4, embed_dim=hidden_channels*2, depth=1,
                                                      num_heads=4,drop_path_rate=0.2)
        # self.decoder_layer2_up = nn.ConvTranspose3d(hidden_channels * 4, hidden_channels * 2, kernel_size=(2, 2, 1),
        #                                             stride=(2, 2, 1))
        #
        # self.decoder_layer3 = nn.Sequential(
        #     nn.Conv3d(256, 128, kernel_size=3, stride=1, padding=1),
        #     nn.LeakyReLU(0.2),
        #     nn.BatchNorm3d(128),
        # )
        self.decoder_layer3 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv3d(hidden_channels * 4, hidden_channels * 2, kernel_size=3, stride=1, padding=1)),
            nn.InstanceNorm3d(hidden_channels * 2),
            nn.ReLU(),

        )
        # self.resup3 = ResBlock3D(256, 128)
        self.resup3 = ResBlock3D(hidden_channels * 4, hidden_channels * 2)
        # self.decoder_layer3_up = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.decoder_layer3_up = nn.ConvTranspose3d(hidden_channels * 2, hidden_channels, kernel_size=2, stride=2)
        #

        self.decoder_layer4 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv3d(hidden_channels * 4, hidden_channels * 2, kernel_size=3, stride=1, padding=1)),
            nn.InstanceNorm3d(hidden_channels * 2),
            nn.ReLU(),
        )
        # self.resup4 = ResBlock3D(256, 128)
        self.resup4 = ResBlock3D(hidden_channels * 4, hidden_channels * 2)
        # self.decoder_layer4_up = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.decoder_layer4_up = nn.ConvTranspose3d(hidden_channels * 2, hidden_channels, kernel_size=1, stride=1)
      
        self.decoder_layer5 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv3d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, stride=1, padding=1)),
            # nn.Conv3d(64 , 64, kernel_size=3, stride=1, padding=1),
            nn.utils.spectral_norm(nn.Conv3d(hidden_channels * 2, hidden_channels, kernel_size=3, stride=1, padding=1)),
      
            nn.ReLU(),
        )
        self.after_conv=nn.Sequential(
            nn.utils.spectral_norm(nn.Conv3d(hidden_channels*3,1,kernel_size=3,stride=1,padding=1)),
            # nn.ReLU()
            nn.Tanh()
        )
        self.up_low=nn.Upsample(scale_factor=(2,2,2),mode="trilinear",align_corners=True)

    def forward(self, x):
        # print(x.shape)#80*112*84*1
        # x=self.encoder_conv(x)
        #High Res Part
        x1=self.HighResBlock(x)#[80,112,84,2]
        #Low Res Part
        x2=self.avgpool(x)

        en0 = self.encoder_layer0_down(x2)  # 40*56*42*8

        # print("en0:",en0.shape)
        en1 = self.encoder_layer1_down(en0)  # 20*28*21*16
        # print("en1:",en1.shape)
        #

        en2 = self.encoder_layer2_down(en1)  # 10*14*21*32
        # print("en2:",en2.shape)

        en3 = self.encoder_layer3_down(en2)  # 5*7*7*64

        #
        #有问题
        #

        de0 = self.decoder_layer1(en3) + self.resup1(en3)
        de0 = self.decoder_layer1_up(de0)  # 10*14*21*64
        cat1 = torch.cat([en2, de0], 1)  # 10*14*21*128

        de1 = self.decoder_layer2(cat1) + self.resup2(cat1)  # 10*14*21*64
        de1 = self.decoder_layer2_up(de1)  # 20*28*21*32
        cat2 = torch.cat([en1, de1], 1)  # 20*28*21*64

        de2 = self.decoder_layer3(cat2) + self.resup3(cat2)  # 20*28*21*32
        de2 = self.decoder_layer3_up(de2)  # 40*56*42*16
        cat3 = torch.cat([en0, de2], 1)  # 40*56*42*32

        de4 = self.decoder_layer5(cat3)  # 40*56*42*8
        de4=self.up_low(de4)
        # print("d3:",de3.shape)
        #without final res connection
        # return de4+x
        add_x=torch.cat([de4,x1],dim=1)
        add_x=self.after_conv(add_x)
        return add_x
    #


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv3d(1, hidden_channels, kernel_size=3, stride=1, padding=1)),
            nn.InstanceNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.2),
        )
        self.eca0 = EfficientChannelAttention(hidden_channels)
        self.maxpool0=nn.MaxPool3d(kernel_size=3,stride=2,padding=1)
        # self.squeeze0=Squeeze_excitation_block(8)


        self.conv1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv3d(hidden_channels, hidden_channels*2, kernel_size=3, stride=1, padding=1)),
            nn.InstanceNorm3d(hidden_channels*2),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.2),
        )
        self.eca1 = EfficientChannelAttention(hidden_channels * 2)
        self.maxpool1=nn.MaxPool3d(kernel_size=3,stride=2,padding=1)
        # self.squeeze1=Squeeze_excitation_block(16)



        self.conv2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv3d(hidden_channels*2, hidden_channels*4, kernel_size=3, stride=1, padding=1)),
            nn.InstanceNorm3d(hidden_channels*4),
            nn.ReLU(inplace=True),
        )
        self.eca2 = EfficientChannelAttention(hidden_channels * 4)
        self.maxpool2=nn.MaxPool3d(kernel_size=3,stride=(2,2,3),padding=1)
      
        self.conv3 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv3d(hidden_channels*4, hidden_channels*8, kernel_size=3, stride=1, padding=1)),
            nn.InstanceNorm3d(hidden_channels*8),
            nn.ReLU(inplace=True),
        )
        self.eca3 = EfficientChannelAttention(hidden_channels * 8)
        self.maxpool3=nn.MaxPool3d(kernel_size=3,stride=(2,2,1),padding=1)

        self.conv4 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv3d(hidden_channels*8, hidden_channels*8, kernel_size=3, stride=1, padding=1)),
            nn.InstanceNorm3d(hidden_channels*8),
            nn.ReLU(inplace=True),
        )
        self.eca4 = EfficientChannelAttention(hidden_channels * 8)
        self.maxpool4=nn.MaxPool3d(kernel_size=3,stride=1,padding=1)

        self.final_conv=nn.Conv3d(hidden_channels*8,1,kernel_size=(1,1,1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv0(x)
        x = self.eca0(x)
        x = self.maxpool0(x)
        x = self.conv1(x)
        x = self.eca1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.eca2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.eca3(x)
        x = self.maxpool3(x)
        x = self.conv4(x)
        x = self.eca4(x)
        x = self.maxpool4(x)
        x = self.final_conv(x)
        return x



def weights_init(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm3d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

totalTimesteps = 100
ratio_train = 0.7  # according to deep learning specialization, if you have a small collection of data, then use 70%/30% for train/test.
nTimesteps_train = round(totalTimesteps * ratio_train)  # nTimesteps_train=70.
nTimesteps_test = totalTimesteps - nTimesteps_train  # nTimestamps_test=30.
dataSourcePath = 'Your datasource Path'
dataSavePath = 'Your data saving path'
fileStartVal = 1
fileIncrement = 1
constVal = 1
cropScaleFactor_train = (0.5, 0.5, 0.5)  # [depth, height, width].
cropScaleFactor_test = (1, 1, 1)  # to be recovered!!!
dim = (160, 224, 168)  # [depth, height, width].
dim_A = 1
dim_B = 1
dim_crop = (int(dim[0] * cropScaleFactor_test[0]),
         int(dim[1] * cropScaleFactor_test[1]),
         int(dim[2] * cropScaleFactor_test[2]))
float32DataType = np.float32

myRandCrop3D = MyRandomCrop3D3(volume_sz=(1, dim[0], dim[1], dim[2]),
                               cropVolume_sz=dim_crop)

BCELoss=nn.BCEWithLogitsLoss()
L1=nn.L1Loss()
hidden_channels=8
#ms_ssim_loss = pytorch_ssim.SSIM(window_size=11)
chunk_size=1960
n_epochs = 3000 #100
batch_size = 1
display_step1 = np.ceil(np.ceil(nTimesteps_train / batch_size) * n_epochs / 100)
display_step2 = np.ceil(np.ceil(nTimesteps_train / batch_size) * n_epochs / 20)   #10
lr = 0.0002  #0.0002
lr_gen=0.0001
lr_dis=0.0004
device1=torch.device('cuda:0')
final_train_step = 46400 # need to be modified every time.

dataset_test = VolumesDataset_test(dataSourcePath=dataSourcePath, totalTimesteps=totalTimesteps,
                                   nTimesteps_test=nTimesteps_test,
                                   dim=dim,
                                   fileStartVal=fileStartVal, fileIncrement=fileIncrement, constVal=constVal,
                                   float32DataType=float32DataType,
                                   transform=myRandCrop3D
                                   )
dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)  # make sure change shuffle=True to be shuffle=False.


gen = Generator(hidden_channels=hidden_channels)

fileName_state_dict =torch.load("%load the training pth" % (dataSavePath,  final_train_step))
torch.cuda.set_device(0)
gen.load_state_dict(fileName_state_dict['gen'])
gen.to(device1)


def predict():
    start_time = time.time()
    with torch.no_grad():
        for ct,mri,index in dataloader:
            ct = ct.to(device1)
            fake_MRI=gen(ct)
            saveRawFile_6x(index,"pred_norm_mr",fake_MRI)
    # compute elapsed time.
    elapsed = time.time() - start_time
    elapsedTime = str(datetime.timedelta(seconds=elapsed))
    print(f"predict/infer time consumed: {elapsed} seconds, {elapsedTime}")

predict()
