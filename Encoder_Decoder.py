import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
import numbers

##########################################################################
## Layer Norm
##########################################################################
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

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
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
## Overlapped patch embedding 
##########################################################################
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x

##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
##########################################################################
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
    
##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
##########################################################################
class MDTA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

##########################################################################
# Frequency domain-based Self-Attention(FDSA)
##########################################################################
class FDSA(nn.Module):
    def __init__(self, dim, bias=False):
        super(FDSA, self).__init__()

        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)

        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        self.norm = LayerNorm(dim * 2, LayerNorm_type='WithBias')

        self.patch_size = 8

    def forward(self, x):

        b, c, h, w = x.shape
        
        # Calculate required padding to make dimensions divisible by patch_size
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        hidden = self.to_hidden(x)

        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)

        q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        q_fft = torch.fft.rfft2(q_patch.float())
        k_fft = torch.fft.rfft2(k_patch.float())

        out = q_fft * k_fft
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))
        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                        patch2=self.patch_size)
        
        out = out[:, :, :h, :w]
        v = v[:, :, :h, :w]

        out = self.norm(out)

        output = v * out
        output = self.project_out(output)

        return output

##########################################################################
## Lite Transformer(LITE)
##########################################################################
class LITE(nn.Module):
    def __init__(self,
                 dim,   
                 num_heads=8,
                 qkv_bias=False,):
        super(LITE, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim*3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim*3, dim*3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv2(self.qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.proj(out)
        return out

##########################################################################
## Dual-Domain Filter Block
##########################################################################
class FilterBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(FilterBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = MDTA(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)  
        self.fft = FDSA(dim, bias)
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x1 = self.fft(self.norm2(x))
        x2 = x + self.ffn(self.norm3(x1))
        
        return x2

##########################################################################  
##  MLP-Mixer
##########################################################################  
class Mlp(nn.Module):
    def __init__(self, 
                 in_features, 
                 hidden_features=None, 
                 ffn_expansion_factor = 2,
                 bias = False):
        super().__init__()
        hidden_features = int(in_features*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            in_features, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, in_features, kernel_size=1, bias=bias)
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

##########################################################################
## learnable descriptive convolution(LDC)
##########################################################################
class LDC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        
        super(LDC, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)  # [12,3,3,3]

        self.center_mask = torch.tensor([[0, 0, 0],
                                         [0, 1, 0],
                                         [0, 0, 0]]).cuda()
        self.base_mask = nn.Parameter(torch.ones(self.conv.weight.size()), requires_grad=False)  # [12,3,3,3]
        self.learnable_mask = nn.Parameter(torch.ones([self.conv.weight.size(0), self.conv.weight.size(1)]),
                                           requires_grad=True)  # [12,3]
        self.learnable_theta = nn.Parameter(torch.ones(1) * 0.5, requires_grad=True)  # [1]

    def forward(self, x):
        mask = self.base_mask - self.learnable_theta * self.learnable_mask[:, :, None, None] * \
               self.center_mask * self.conv.weight.sum(2).sum(2)[:, :, None, None]

        out_diff = F.conv2d(input=x, weight=self.conv.weight * mask, bias=self.conv.bias, stride=self.conv.stride,
                            padding=self.conv.padding,
                            groups=self.conv.groups)
        return out_diff

##########################################################################
## Structural Feature Extraction Module
##########################################################################
class SgExtraction(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_expansion_factor=2.,     
                 qkv_bias=False,):
        super(SgExtraction, self).__init__()
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn = LITE(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = LayerNorm(dim, 'WithBias')
        self.mlp = Mlp(in_features=dim,
                       ffn_expansion_factor=ffn_expansion_factor)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        sg = x + self.mlp(self.norm2(x))
        return sg

##########################################################################
## local Feature Extraction Module
##########################################################################
class TlExtraction(nn.Module):
    def __init__(self, dim=32):
        super(TlExtraction, self).__init__()
        self.pool_kernel = 3
        self.Conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.Enhance_texture = LDC(in_channels = dim, out_channels = dim)
        self.avgpool = nn.AvgPool2d(
            kernel_size=self.pool_kernel,
            stride=1,  
            padding=(self.pool_kernel - 1) // 2  
        ).cuda()
    def forward(self, x):
        x_ori = self.Conv(x)
        x_enhance = self.Enhance_texture(x)
        x_diff = x_enhance - x_ori
        global_avg_feat = torch.mean(x_diff)
        weight = torch.sigmoid(global_avg_feat)
        Tl = x_ori * weight
        
        return Tl

##########################################################################
## Discrepancy-Adaptive Aggregation Layer(DAAL)
##########################################################################    
class DAAL(nn.Module):
    def __init__(self, indim, outdim):
        super(DAAL, self).__init__()

        self.Conv_1 = nn.Sequential(nn.Conv2d(indim*2, outdim, kernel_size=1),
                                    nn.BatchNorm2d(outdim),
                                    nn.SiLU(inplace=True)
                                    )

        self.Conv = nn.Sequential(nn.Conv2d(outdim, outdim, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(outdim),
                                  nn.SiLU(inplace=True),
                                  )
        self.silu = nn.SiLU(inplace=True)

    def forward(self, diff, accom):
        cat = torch.cat([accom, diff], dim=1)
        cat = self.Conv_1(cat) + diff + accom
        c = self.Conv(cat) + cat
        c = self.silu(c) + diff
        return c

##########################################################################
## Simple Attention Module(SimAM)
##########################################################################
class simam_module(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)

##########################################################################
## Change Boundary-Aware Module(CBM)
##########################################################################
class diff_moudel(nn.Module):
    def __init__(self, in_channel):
        super(diff_moudel, self).__init__()
        self.avg_pool = nn.AvgPool2d((3, 3), stride=1, padding=1)
        self.conv_1 = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.sigmoid = nn.Sigmoid()
        self.simam = simam_module()

    def forward(self, x):
        x = self.simam(x)
        edge = x - self.avg_pool(x)  # Xi=X-Avgpool(X)
        weight = self.sigmoid(self.bn1(self.conv_1(edge)))
        # weight = self.conv_1(edge)
        out = weight * x + x
        out = self.simam(out)
        return out

class CBM(nn.Module):
    def __init__(self, in_channel):
        super(CBM, self).__init__()
        self.diff_1 = diff_moudel(in_channel)
        self.diff_2 = diff_moudel(in_channel)
        self.simam = simam_module()

    def forward(self, x1, x2):
        d1 = self.diff_1(x1)
        d2 = self.diff_2(x2)
        d = torch.abs(d1 - d2)
        d = self.simam(d)
        return d

##########################################################################
## Bitemporal Feature Aggregation Module(BFAM)
##########################################################################
class BFAM(nn.Module):
    def __init__(self, inp, out):
        super(BFAM, self).__init__()

        self.pre_siam = simam_module()
        self.lat_siam = simam_module()

        out_1 = int(inp / 2)

        self.conv_1 = nn.Conv2d(inp*2, out_1, padding=1, kernel_size=3, groups=out_1,
                                dilation=1)
        self.conv_2 = nn.Conv2d(inp*2, out_1, padding=2, kernel_size=3, groups=out_1,
                                dilation=2)
        self.conv_3 = nn.Conv2d(inp*2, out_1, padding=3, kernel_size=3, groups=out_1,
                                dilation=3)
        self.conv_4 = nn.Conv2d(inp*2, out_1, padding=4, kernel_size=3, groups=out_1,
                                dilation=4)

        self.fuse = nn.Sequential(
            nn.Conv2d(out_1 * 4, out_1*2, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_1*2),
            nn.SiLU(inplace=True)
        )

        self.fuse_siam = simam_module()

        self.out = nn.Sequential(
            nn.Conv2d(out_1*2, out, kernel_size=3, padding=1),
            nn.BatchNorm2d(out),
            nn.SiLU(inplace=True)
        )

    def forward(self, inp1, inp2, last_feature=None):
        x = torch.cat([inp1, inp2], dim=1)
        c1 = self.conv_1(x)
        c2 = self.conv_2(x)
        c3 = self.conv_3(x)
        c4 = self.conv_4(x)
        cat = torch.cat([c1, c2, c3, c4], dim=1)
        fuse = self.fuse(cat)
        inp1_siam = self.pre_siam(inp1)
        inp2_siam = self.lat_siam(inp2)

        inp1_mul = torch.mul(inp1_siam, fuse)
        inp2_mul = torch.mul(inp2_siam, fuse)
        fuse = self.fuse_siam(fuse)
        if last_feature is None:
            out = self.out(fuse + inp1 + inp2 + inp2_mul + inp1_mul)
        else:
            out = self.out(fuse + inp2_mul + inp1_mul + last_feature + inp1 + inp2)
        out = self.fuse_siam(out)

        return out

##########################################################################
## Global-Local to Difference-Accommodation Converter(GL-DAC)
##########################################################################  
class GL_DAC(nn.Module):
    def __init__(self,dim=32):

        super(GL_DAC, self).__init__()

        self.diff_extraction = CBM(in_channel=dim)
        self.accom_extraction = BFAM(inp=dim, out=dim)


    def forward(self,base_feature, detail_feature, mask):
        
        
        diff_feature = self.diff_extraction(base_feature, detail_feature)
        accom_feature = self.accom_extraction(base_feature, detail_feature)
        feature = base_feature * (1 - mask) + detail_feature * mask
        diff_feature = diff_feature + feature
        accom_feature = accom_feature + feature
        
        return diff_feature, accom_feature
##########################################################################
## Encoder-Decoder
##########################################################################
## Hierarchical Feature Encoder(HFE)
class HFE(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=32,
                 num_blocks=[1, 1],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):

        super(HFE, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.FilterBlock = nn.Sequential(*[FilterBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])   #feature dim 64

        self.Sg = SgExtraction(dim=dim, num_heads = heads[2])
        self.Tl = TlExtraction(dim=dim)
        
       
             
    def forward(self, inp_img):
        patch_embed = self.patch_embed(inp_img)
        enhance_feature = self.FilterBlock(patch_embed)
        sg = self.Sg(enhance_feature)
        tl = self.Tl(enhance_feature)

        return sg, tl
##########################################################################
## Difference-Accommodation Feature Decoder(DAFD)
class DAFD(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=32,
                 num_blocks=[1, 1],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):

        super(DAFD, self).__init__()

        self.daal = DAAL(indim=dim, outdim=dim)
        self.FilterBlock = nn.Sequential(*[FilterBlock(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.output = nn.Sequential(
            nn.Conv2d(int(dim), int(dim)//2, kernel_size=3,
                      stride=1, padding=1, bias=bias),
            nn.SiLU(),
            nn.Conv2d(int(dim)//2, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=bias)
            )
        self.sigmoid = nn.Sigmoid()  


    def forward(self, inp_img, diff_feature, accom_feature):
        
        aggregate_feature = self.daal(diff_feature, accom_feature)

        final_feature = self.FilterBlock(aggregate_feature)
        if inp_img is not None:
            final = self.output(final_feature) + inp_img
        else:
            final = self.output(final_feature)
        return self.sigmoid(final), final_feature
##########################################################################
## Edge Decoder
class EdgeDecoder(nn.Module):
    def __init__(self, in_channels=32, out_channels=1):
        super(EdgeDecoder, self).__init__()
        self.main_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels//2, in_channels//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels//4, 1, kernel_size=1)
        )
        self.side_branch = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        edge = self.main_branch(x)+self.side_branch(x)
        return self.sigmoid(edge)

if __name__ == '__main__':
   
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    height = 640
    width = 480
   
    modelE = HFE().to(device)
    modelD = DAFD().to(device)
    modelG = GL_DAC().to(device)
    batch_size = 1
    channels = 1
   
    test_input_I = torch.randn(batch_size, channels, height, width).to(device)
    test_input_V = torch.randn(batch_size, channels, height, width).to(device)
   
    
    print(f"input : {test_input_I.shape}")
    print(f"input : {test_input_V.shape}")
    
    
    modelE.eval()
    modelD.eval()
    with torch.no_grad():
            base_feature_I, detail_feature_I = modelE(test_input_I)
            base_feature_V, detail_feature_V = modelE(test_input_V)
            diff_feature, accom_feature = modelG(base_feature_I, base_feature_V)
            
            print(f"diff_feature: {diff_feature.shape}")
            print(f"accom_feature: {accom_feature.shape}")  
            output = modelD(test_input_V, diff_feature, accom_feature)
            print(f"output: {output.shape}")    



