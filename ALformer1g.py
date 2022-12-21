"""
## Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808
"""

import torch
import torch.nn as nn
from subpixel import shuffle_down, shuffle_up###################
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange
from Blanced_attention import BlancedAttention, BlancedAttention_CAM_SAM_ADD


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        self.act1 = nn.PReLU()
        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = self.act1(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)#nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)#nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)#nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
        #self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, q_fea, k_fea, v_fea):
        b,c,h,w = q_fea.shape
        q = self.q(q_fea)
        k = self.k(k_fea)
        v = self.v(v_fea)
        #qkv = self.qkv_dwconv(self.qkv(x))
        #q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class MDTA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm_key = LayerNorm(dim, LayerNorm_type)
        self.norm_query = LayerNorm(dim, LayerNorm_type)
        self.norm_value = LayerNorm(dim, LayerNorm_type)
        
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, key, query, value):
        x = value + self.attn(self.norm_key(key),self.norm_query(query),self.norm_value(value))
        x = x + self.ffn(self.norm2(x))

        return x
        
##########################################################################
class RestormerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(RestormerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = MDTA(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x
        
##########################################################################
class Scale(nn.Module):

    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale
##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

def st_conv(in_channels, out_channels, kernel_size, bias=False, stride = 2):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)		
##########################################################################
class depthwise_separable_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(depthwise_separable_conv, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.depth_conv = nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1, groups=ch_in)
        self.point_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x
##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
		
# contrast-aware channel attention module
def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)
	
class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )


    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
##########################################################################
## Global context Layer
# class GCLayer(nn.Module):
    # def __init__(self, channel, reduction=16, bias=False):
        # super(GCLayer, self).__init__()
        # # global average pooling: feature --> point
        # #self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # # feature channel downscale and upscale --> channel weight
        # self.conv_phi = nn.Conv2d(channel, 1, 1, stride=1,padding=0, bias=False)
        # self.softmax = nn.Softmax(dim=1)
		
        # self.conv_du = nn.Sequential(
                # nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                # nn.ReLU(inplace=True),
                # nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                # nn.Sigmoid()
        # )

    # def forward(self, x):
        # b, c, h, w = x.size()
        # #y = self.avg_pool(x)
        # y_1 = self.conv_phi(x).view(b, 1, -1).permute(0, 2, 1).contiguous()### b,hw,1
        # y_1_att = self.softmax(y_1)
        # print(y_1.size)
        # x_1 = x.view(b, c, -1)### b,c,hw
        # mul_context = torch.matmul(x_1, y_1_att)#### b,c,1
        # mul_context = mul_context.view(b, c, 1, -1)

        # y = self.conv_du(mul_context)
        # return x * y
		
##########################################################################
## Semantic-guidance Texture Enhancement Module
class STEM(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias=False):
        super(STEM, self).__init__()
        # global average pooling: feature --> point
        
        act=nn.PReLU()
        #num_blocks = 1
        heads = 1
        ffn_expansion_factor = 2.66
        LayerNorm_type = 'WithBias'  ## Other option 'BiasFree'

        #self.down_img = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)#DownSample1()
        self.conv_stem0 = st_conv(3, n_feat, kernel_size, bias=bias)
        #self.conv_stem1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv_stem2 = conv(3, n_feat, kernel_size, bias=bias)
        self.former = TransformerBlock(dim=n_feat, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias=False, LayerNorm_type=LayerNorm_type)
        self.conv_stem3 = conv(3, n_feat, kernel_size, bias=bias)
        self.S2FB = S2FB_2(n_feat, reduction, bias=bias, act=act)
        #self.CA_fea = CALayer(n_feat, reduction, bias=bias)

    def forward(self, img_rain, res, img):
        #img_down = self.down_img(img_rain)
        img_down = self.conv_stem0(img_rain)
        #img_down_fea = self.conv_stem1(img_down)
        res_fea = self.conv_stem2(res)
        #rain_mask = torch.sigmoid(res_fea)
        #rain_mask = self.CA_fea(res_fea)
        #att_fea = img_down * rain_mask + img_down
        att_fea = self.former(res_fea, img_down, img_down)
        img_fea = self.conv_stem3(img)
        S2FB2_FEA = self.S2FB(img_fea, att_fea)
        return S2FB2_FEA
        #return torch.cat([img_down_fea * rain_mask, img_fea],1) 
		
##########################################################################
## S2FB
class S2FB_2(nn.Module):
    def __init__(self, n_feat, reduction, bias, act):
        super(S2FB_2, self).__init__()
        self.DSC = depthwise_separable_conv(n_feat*2, n_feat)
        #self.CON_FEA = nn.Conv2d(n_feat*2, n_feat, kernel_size=1, bias=bias)
        self.CA_fea = CALayer(n_feat, reduction, bias=bias)
        #self.CA_fea = BlancedAttention_CAM_SAM_ADD(n_feat, reduction)
        #self.CA_fea = CCALayer(n_feat, reduction, bias=bias)
    def forward(self, x1, x2):
        FEA_1 = self.DSC(torch.cat((x1,x2), 1))
        #FEA_2 = self.CON_FEA(torch.cat((x1,x2), 1))
        #resin = FEA_1 + FEA_2
        res= self.CA_fea(FEA_1) + x1
        #res += resin
        return res#x1 + resin

##########################################################################
## S2FB
class S2FB_4(nn.Module):
    def __init__(self, n_feat, reduction, bias, act):
        super(S2FB_4, self).__init__()
        #self.CA_fea1 = CALayer(n_feat, reduction, bias=bias)
        #self.CA_fea2 = CALayer(n_feat, reduction, bias=bias)
        #self.CA_fea3 = CALayer(n_feat, reduction, bias=bias)
        self.DSC = depthwise_separable_conv(n_feat*4, n_feat)
        #self.CON_FEA = nn.Conv2d(n_feat*3, n_feat, kernel_size=1, bias=bias)
        self.CA_fea = CALayer(n_feat, reduction, bias=bias)
        #self.CA_fea = BlancedAttention_CAM_SAM_ADD(n_feat, reduction)
        #self.CA_fea = CCALayer(n_feat, reduction, bias=bias)
    def forward(self, x1, x2, x3, x4):
        FEA_1 = self.DSC(torch.cat((x1, x2,x3,x4), 1))
        #FEA_2 = self.CON_FEA(torch.cat((x2,x3,x4), 1))
        #resin = FEA_1 + FEA_2
        res= self.CA_fea(FEA_1) + FEA_1
        #res += resin
        #res1= self.CA_fea1(FEA_1)
        #res2= self.CA_fea2(x3)
        #res3= self.CA_fea3(x4)
        return res#x1 + res

class S2FB_p(nn.Module):
    def __init__(self, n_feat, reduction, bias, act):
        super(S2FB_p, self).__init__()
        #self.CA_fea1 = CALayer(n_feat, reduction, bias=bias)
        #self.CA_fea2 = CALayer(n_feat, reduction, bias=bias)
        #self.CA_fea3 = CALayer(n_feat, reduction, bias=bias)
        self.DSC1 = depthwise_separable_conv(n_feat*2, n_feat)
        self.DSC2 = depthwise_separable_conv(n_feat*2, n_feat)
        self.DSC3 = depthwise_separable_conv(n_feat*2, n_feat)
        self.CA_fea = CALayer(n_feat, reduction, bias=bias)
        #self.CA_fea = BlancedAttention_CAM_SAM_ADD(n_feat, reduction)
        #self.CA_fea = CCALayer(n_feat, reduction, bias=bias)
    def forward(self, x1, x2, x3, x4):
        FEA_34 = self.DSC1(torch.cat((x3, x4), 1))
        FEA_34_2 = self.DSC2(torch.cat((x2, FEA_34), 1))
        FEA_34_2_1 = self.DSC3(torch.cat((x1, FEA_34_2), 1))
        res= self.CA_fea(FEA_34_2_1) + FEA_34_2_1
        #res += resin
        #res1= self.CA_fea1(FEA_1)
        #res2= self.CA_fea2(x3)
        #res3= self.CA_fea3(x4)
        return res#x1 + res
##########################################################################
## stage_encoder_module
class stage_encoder_module_dn(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(stage_encoder_module_dn, self).__init__()
        self.CAB1 = CAB_dsc(n_feat, kernel_size, reduction, bias=bias, act=act)
        #self.S2FB1 = S2FB_2(n_feat, reduction, bias=bias, act=act)
        self.CAB2 = CAB_dsc(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.S2FB2 = S2FB_2(n_feat, reduction, bias=bias, act=act)
        #self.CAB3 = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        #self.S2FB3 = S2FB_2(n_feat, bias=bias, act=act)
		
    def forward(self, x):
        en_CAB1_FEA = self.CAB1(x)
        #en_S2FB1_FEA = self.S2FB1(x, en_CAB1_FEA)
        en_CAB2_FEA = self.CAB2(en_CAB1_FEA)
        en_S2FB2_FEA = self.S2FB2(x, en_CAB2_FEA)
        #en_CAB3_FEA = self.CAB3(en_S2FB2_FEA)
        #en_S2FB3_FEA = self.S2FB3(en_S2FB2_FEA, en_CAB3_FEA)
        #en_S2FB1_FEA += x
        return en_S2FB2_FEA
##########################################################################
## stage_decoder_module
class stage_decoder_module_dn(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(stage_decoder_module_dn, self).__init__()
        self.S2FB0 = S2FB_p(n_feat, reduction, bias=bias, act=act)
        self.CAB1 = ECAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        #self.S2FB1 = S2FB_2(n_feat, reduction, bias=bias, act=act)
        self.CAB2 = ECAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.S2FB2 = S2FB_2(n_feat, reduction, bias=bias, act=act)
        #self.CAB3 = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        #self.S2FB3 = S2FB_2(n_feat, bias=bias, act=act)
		
    def forward(self, x, stage_skip1, stage_skip2, stage_skip3):
        de_S2FB0_FEA = self.S2FB0(x, stage_skip1, stage_skip2, stage_skip3)
        de_CAB1_FEA = self.CAB1(de_S2FB0_FEA)
        #de_S2FB1_FEA = self.S2FB1(de_S2FB0_FEA, de_CAB1_FEA)
        de_CAB2_FEA = self.CAB2(de_CAB1_FEA)
        de_S2FB2_FEA = self.S2FB2(de_S2FB0_FEA, de_CAB2_FEA)
        #de_CAB3_FEA = self.CAB3(de_S2FB2_FEA)
        #de_S2FB3_FEA = self.S2FB2(de_S2FB2_FEA, de_CAB3_FEA)
        #de_S2FB2_FEA += x
        return de_S2FB2_FEA
		
## stage_encoder_module
class stage_encoder_module(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(stage_encoder_module, self).__init__()
        self.CAB1 = CAB_dsc(n_feat, kernel_size, reduction, bias=bias, act=act)
        #self.S2FB1 = S2FB_2(n_feat, reduction, bias=bias, act=act)
        #self.CAB2 = CAB_dsc(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.S2FB2 = S2FB_2(n_feat, reduction, bias=bias, act=act)
        #self.CAB3 = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        #self.S2FB3 = S2FB_2(n_feat, bias=bias, act=act)
		
    def forward(self, x):
        en_CAB1_FEA = self.CAB1(x)
        #en_S2FB1_FEA = self.S2FB1(x, en_CAB1_FEA)
        #en_CAB2_FEA = self.CAB2(en_CAB1_FEA)
        en_S2FB2_FEA = self.S2FB2(x, en_CAB1_FEA)
        #en_CAB3_FEA = self.CAB3(en_S2FB2_FEA)
        #en_S2FB3_FEA = self.S2FB1(en_S2FB2_FEA, en_CAB3_FEA)
        #en_S2FB2_FEA += x
        return en_S2FB2_FEA
##########################################################################
## stage_decoder_module
class stage_decoder_module(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(stage_decoder_module, self).__init__()
        self.S2FB0 = S2FB_p(n_feat, reduction, bias=bias, act=act)
        self.CAB1 = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        #self.S2FB1 = S2FB_2(n_feat, reduction, bias=bias, act=act)
        #self.CAB2 = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.S2FB2 = S2FB_2(n_feat, reduction, bias=bias, act=act)
        #self.CAB3 = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        #self.S2FB3 = S2FB_2(n_feat, bias=bias, act=act)
		
    def forward(self, x, stage_skip1, stage_skip2, stage_skip3):
        de_S2FB0_FEA = self.S2FB0(x, stage_skip1, stage_skip2, stage_skip3)
        de_CAB1_FEA = self.CAB1(de_S2FB0_FEA)
        #de_S2FB1_FEA = self.S2FB1(de_S2FB0_FEA, de_CAB1_FEA)
        #de_CAB2_FEA = self.CAB2(de_CAB1_FEA)
        de_S2FB2_FEA = self.S2FB2(de_S2FB0_FEA, de_CAB1_FEA)
        #de_CAB3_FEA = self.CAB3(de_S2FB2_FEA)
        #de_S2FB3_FEA = self.S2FB2(de_S2FB2_FEA, de_CAB3_FEA)
        #de_S2FB2_FEA += x
        return de_S2FB2_FEA
##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res
		
## Enhanced Channel Attention Block (ECAB)
class ECAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(ECAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)
        self.S2FB2 = S2FB_2(n_feat, reduction, bias=bias, act=act)
    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res = self.S2FB2(res, x)
        #res += x
        return res
		
## Channel Attention Block (CAB)
class CAB_dsc(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB_dsc, self).__init__()
        modules_body = []
        modules_body.append(depthwise_separable_conv(n_feat, n_feat))
        modules_body.append(act)
        modules_body.append(depthwise_separable_conv(n_feat, n_feat))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)
        self.S2FB2 = S2FB_2(n_feat, reduction, bias=bias, act=act)
    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res = self.S2FB2(res, x)
        #res += x
        return res

##########################################################################
## Supervised Attention Module
# class SAM(nn.Module):
    # def __init__(self, n_feat, kernel_size, bias):
        # super(SAM, self).__init__()
        # self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        # self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        # self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    # def forward(self, x, x_img):
        # x1 = self.conv1(x)
        # img = self.conv2(x) + x_img
        # x2 = torch.sigmoid(self.conv3(img))
        # x1 = x1*x2
        # x1 = x1+x
        # return x1, img

##########################################################################
## U-Net

class Encoder_dn(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(Encoder_dn, self).__init__()

        #self.encoder_level1 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        #self.encoder_level2 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        #self.encoder_level3 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
		
        #self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        #self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        #self.encoder_level3 = nn.Sequential(*self.encoder_level3)
        self.encoder_level1 = stage_encoder_module_dn(n_feat, kernel_size, reduction, bias, act)
        self.encoder_level2 = stage_encoder_module_dn(n_feat, kernel_size, reduction, bias, act)
        self.encoder_level3 = stage_encoder_module_dn(n_feat, kernel_size, reduction, bias, act)
		
        self.down_stage1  = DownSample(n_feat)
        self.down_stage2  = DownSample(n_feat)
        self.down_stage3  = DownSample(n_feat)

        # # Cross Stage Feature Fusion (CSFF)
        # if csff:
            # self.csff_enc1 = nn.Conv2d(n_feat,                     n_feat,                     kernel_size=1, bias=bias)
            # self.csff_enc2 = nn.Conv2d(n_feat+scale_unetfeats,     n_feat+scale_unetfeats,     kernel_size=1, bias=bias)
            # self.csff_enc3 = nn.Conv2d(n_feat+(scale_unetfeats*2), n_feat+(scale_unetfeats*2), kernel_size=1, bias=bias)

            # self.csff_dec1 = nn.Conv2d(n_feat,                     n_feat,                     kernel_size=1, bias=bias)
            # self.csff_dec2 = nn.Conv2d(n_feat+scale_unetfeats,     n_feat+scale_unetfeats,     kernel_size=1, bias=bias)
            # self.csff_dec3 = nn.Conv2d(n_feat+(scale_unetfeats*2), n_feat+(scale_unetfeats*2), kernel_size=1, bias=bias)

    def forward(self, x):

        x_1 = self.down_stage1(x)
        enc1 = self.encoder_level1(x_1)

        x_2 = self.down_stage2(enc1)
        enc2 = self.encoder_level2(x_2)
		
        x_3 = self.down_stage3(enc2)
        enc3 = self.encoder_level3(x_3)

        return [enc1, enc2, enc3]

class Decoder_dn(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(Decoder_dn, self).__init__()

        self.decoder_level1 = stage_decoder_module_dn(n_feat, kernel_size, reduction, bias, act)
        self.decoder_level2 = stage_decoder_module_dn(n_feat, kernel_size, reduction, bias, act)
        self.decoder_level3 = stage_decoder_module_dn(n_feat, kernel_size, reduction, bias, act)

        self.up_stage1  = UpSample(n_feat)
        self.up_stage2  = UpSample(n_feat)
        self.up_stage3  = UpSample(n_feat)
		
        self.up_32  = UpSample(n_feat)
        self.up_31  = UpSample4(n_feat)
        self.up_21  = UpSample(n_feat)
		
        self.down_13  = DownSample4(n_feat)
        self.down_23  = DownSample(n_feat)
        self.down_12  = DownSample(n_feat)
		
        #self.down_01  = DownSample(n_feat)
        #self.down_02  = DownSample4(n_feat)
		
    def forward(self, outs):
        enc1, enc2, enc3 = outs
		
        enc13_down  = self.down_13(enc1)
        enc23_down  = self.down_23(enc2)
        dec3 = self.decoder_level3(enc3, enc3, enc23_down, enc13_down)
        upstage32 = self.up_stage3(dec3)
		
        enc32_up = self.up_32(enc3)
        #osr02_down  = self.down_02(or_fea)
        enc12_down  = self.down_12(enc1)
        dec2 = self.decoder_level2(upstage32, enc2, enc12_down, enc32_up)
        upstage21 = self.up_stage2(dec2)
		
        enc31_up = self.up_31(enc3)
        enc21_up = self.up_21(enc2)
        #osr01_down  = self.down_01(or_fea)
        dec1 = self.decoder_level1(upstage21, enc1, enc21_up, enc31_up)
		
        upstage10 = self.up_stage1(dec1)

        return upstage10

		
class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(Encoder, self).__init__()


        self.encoder_level1 = stage_encoder_module(n_feat, kernel_size, reduction, bias, act)
        self.encoder_level2 = stage_encoder_module(n_feat, kernel_size, reduction, bias, act)
        self.encoder_level3 = stage_encoder_module(n_feat, kernel_size, reduction, bias, act)
		
        self.down_stage1  = DownSample(n_feat)
        self.down_stage2  = DownSample(n_feat)
        self.down_stage3  = DownSample(n_feat)

    def forward(self, x):

        x_1 = self.down_stage1(x)
        enc1 = self.encoder_level1(x_1)

        x_2 = self.down_stage2(enc1)
        enc2 = self.encoder_level2(x_2)
		
        x_3 = self.down_stage3(enc2)
        enc3 = self.encoder_level3(x_3)

        return [enc1, enc2, enc3]

class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(Decoder, self).__init__()

        self.decoder_level1 = stage_decoder_module(n_feat, kernel_size, reduction, bias, act)
        self.decoder_level2 = stage_decoder_module(n_feat, kernel_size, reduction, bias, act)
        self.decoder_level3 = stage_decoder_module(n_feat, kernel_size, reduction, bias, act)

        self.up_stage1  = UpSample(n_feat)
        self.up_stage2  = UpSample(n_feat)
        self.up_stage3  = UpSample(n_feat)
		
        self.up_32  = UpSample(n_feat)
        self.up_31  = UpSample4(n_feat)
        self.up_21  = UpSample(n_feat)
		
        self.down_13  = DownSample4(n_feat)
        self.down_23  = DownSample(n_feat)
        self.down_12  = DownSample(n_feat)
		
        #self.down_01  = DownSample(n_feat)
        #self.down_02  = DownSample4(n_feat)
		
    def forward(self, outs):
        enc1, enc2, enc3 = outs
		
        enc13_down  = self.down_13(enc1)
        enc23_down  = self.down_23(enc2)
        dec3 = self.decoder_level3(enc3, enc3, enc23_down, enc13_down)
        upstage32 = self.up_stage3(dec3)
		
        enc32_up = self.up_32(enc3)
        #osr02_down  = self.down_02(or_fea)
        enc12_down  = self.down_12(enc1)
        dec2 = self.decoder_level2(upstage32, enc2, enc12_down, enc32_up)
        upstage21 = self.up_stage2(dec2)
		
        enc31_up = self.up_31(enc3)
        enc21_up = self.up_21(enc2)
        #osr01_down  = self.down_01(or_fea)
        dec1 = self.decoder_level1(upstage21, enc1, enc21_up, enc31_up)
		
        upstage10 = self.up_stage1(dec1)

        return upstage10
##########################################################################
##---------- Resizing Modules ----------    
# class DownSample1(nn.Module):
    # def __init__(self):
        # super(DownSample, self).__init__()
        # self.down = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)

    # def forward(self, x):
        # x = self.down(x)
        # return x

class DownSample(nn.Module):
    def __init__(self, in_channels):
    #def __init__(self, in_channels,s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x
		
class DownSample4(nn.Module):
    def __init__(self, in_channels):
    #def __init__(self, in_channels,s_factor):
        super(DownSample4, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x
	
class DownSample8(nn.Module):
    def __init__(self, in_channels):
    #def __init__(self, in_channels,s_factor):
        super(DownSample4, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.125, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x
# class UpSample1(nn.Module):
    # def __init__(self, in_channels):
    # #def __init__(self, in_channels,s_factor):
        # super(UpSample, self).__init__()
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    # def forward(self, x):
        # x = self.up(x)
        # return x
	
class UpSample(nn.Module):
    def __init__(self, in_channels):
    #def __init__(self, in_channels,s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x

class UpSample4(nn.Module):
    def __init__(self, in_channels):
    #def __init__(self, in_channels,s_factor):
        super(UpSample4, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x
		
class SkipUpSample(nn.Module):
    #def __init__(self, in_channels,s_factor):
    def __init__(self, in_channels):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x

##########################################################################
## Original Resolution Block (ORB)
class ORB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(ORB, self).__init__()
        num_blocks = num_cab
        heads = 2
        ffn_expansion_factor = 2.66
        LayerNorm_type = 'WithBias'  ## Other option 'BiasFree'
        #modules_body = []
        #modules_body = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_cab)]
        self.modules_body = nn.Sequential(*[RestormerBlock(dim=n_feat, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks)])

        self.conv = conv(n_feat, n_feat, kernel_size)
        #self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.modules_body(x)
        res = self.conv(res)
        res += x
        return res

##########################################################################
class ORSNet(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(ORSNet, self).__init__()

        self.orb = ORB(n_feat, kernel_size, reduction, act, bias, num_cab)

    def forward(self, x):
        x = self.orb(x)
        return x

##########################################################################
class DSNet(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(DSNet, self).__init__()

        act=nn.PReLU()
        num_blocks = 1
        heads = 2
        ffn_expansion_factor = 2.66
        LayerNorm_type = 'WithBias'  ## Other option 'BiasFree'
        
        self.down_1 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)#DownSample1()
        self.shallow_feat1 = nn.Sequential(conv(3, n_feat, kernel_size, bias=bias), CAB(n_feat,kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat2 = nn.Sequential(conv(3*4, n_feat, kernel_size, bias=bias), CAB(n_feat,kernel_size, reduction, bias=bias, act=act))
        self.fuse_conv  = conv(n_feat*2, n_feat, kernel_size, bias=bias)
        self.former = nn.Sequential(*[RestormerBlock(dim=n_feat, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks)])
        
        self.orsnet = ORSNet(n_feat, kernel_size, reduction, act, bias, num_cab)
		
        self.dsnnet_encoder = Encoder(n_feat,kernel_size, reduction, bias=bias, act=act)
        self.dsnnet_decoder = Decoder(n_feat,kernel_size, reduction, bias=bias, act=act)
		
        self.S2FB_fuse = S2FB_2(n_feat, reduction, bias=bias, act=act)
        self.tail     = conv(n_feat, 3, kernel_size, bias=bias)

    def forward(self, x):

        H = x.size(2)
        W = x.size(3)

        # Multi-Patch Hierarchy: Split Image into four non-overlapping patches

        # Two Patches for Stage 2
        xtop_img  = x[:,:,0:int(H/2),:]
        xbot_img  = x[:,:,int(H/2):H,:]
		
        x2_img_down = self.down_1(x)
        x2_img_down_fea = self.shallow_feat1(x2_img_down)
		
        # Four Patches for Stage 1
        x1ltop_img = xtop_img[:,:,:,0:int(W/2)]
        x1rtop_img = xtop_img[:,:,:,int(W/2):W]
        x1lbot_img = xbot_img[:,:,:,0:int(W/2)]
        x1rbot_img = xbot_img[:,:,:,int(W/2):W]
		
        stage1_input = torch.cat([x1ltop_img, x1rtop_img, x1lbot_img, x1rbot_img],1) 
        x1fea = self.shallow_feat2(stage1_input)
        
        former_fea = self.former(x1fea)
        
        
        stage1_fuse = torch.cat([x2_img_down_fea, former_fea],1) 
        fuse_fea = self.fuse_conv(stage1_fuse)
		
        or_fea = self.orsnet(fuse_fea)
		
        feadsn_e = self.dsnnet_encoder(fuse_fea)
        feadsn_d = self.dsnnet_decoder(feadsn_e)
		
        fused_fea = self.S2FB_fuse(or_fea, feadsn_d)
        #fused_fea = self.S2FB_fuse(or_fea, fused_fea)
		
        stage1_res = self.tail(fused_fea)
		
        return [stage1_res, x2_img_down-stage1_res]

##########################################################################
class SSNet(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(SSNet, self).__init__()
        num_blocks = 1
        heads = 2
        ffn_expansion_factor = 2.66
        LayerNorm_type = 'WithBias'  ## Other option 'BiasFree'
        
        self.stem = STEM(n_feat, kernel_size, reduction, bias=bias)
        self.former = nn.Sequential(*[RestormerBlock(dim=n_feat, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks)])
        
        self.orsnet = ORSNet(n_feat, kernel_size, reduction, act, bias, num_cab)
        self.up_recon_b  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)#UpSample1(n_feat)
		
        self.ssnnet_encoder = Encoder(n_feat,kernel_size, reduction, bias=bias, act=act)
        self.ssnnet_decoder = Decoder(n_feat,kernel_size, reduction, bias=bias, act=act)
		
        self.S2FB_fuse = S2FB_2(n_feat, reduction, bias=bias, act=act)
        self.deep_conv  = conv(n_feat, n_feat*2, kernel_size, bias=bias)
        #self.up_recon  = UpSample(n_feat*2)

        self.tail     = conv(int(n_feat*2/4), 3, kernel_size, bias=bias)

    def forward(self, x, DSNet_outs):
        res, backgound = DSNet_outs
		
        #backgound_up = self.up_recon_b(backgound)
        res_up = self.up_recon_b(res)
        enhance_fea = self.stem(x, res, backgound)
        
        former_fea = self.former(enhance_fea)
        
        
        or_fea = self.orsnet(former_fea)
		
        feassn_e = self.ssnnet_encoder(former_fea)
        feassn_d = self.ssnnet_decoder(feassn_e)
		
        fused_fea = self.S2FB_fuse(or_fea, feassn_d)
        #fused_fea = self.S2FB_fuse(or_fea, fused_fea)
		
        deep_fea = self.deep_conv(fused_fea)
        #recon_up = self.up_recon(deep_fea)
        recon_up  = shuffle_up(deep_fea, 2)
        recon_res = self.tail(recon_up)

        return recon_res+res_up
		
##########################################################################
class ALformer(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=48, kernel_size=3, reduction=4, num_cab=10, bias=False):
        super(ALformer, self).__init__()

        act=nn.PReLU()
        self.dsnet = DSNet(n_feat, kernel_size, reduction, act, bias, num_cab)
        self.ssnet = SSNet(n_feat, kernel_size, reduction, act, bias, num_cab)

    def forward(self, x_img): #####b,c,h,w
        #print(x_img.shape)
        DSNet_out = self.dsnet(x_img)
        imitation = x_img - self.ssnet(x_img, DSNet_out)
        #print(imitation.shape)
        return [imitation, DSNet_out[1]]