
import numpy as np

from ex_fea import *


from MFMRM import MFMRM 


# ==============================================

# =====================================================
# 黎曼流行的COV协方差
class CovarianceModule(nn.Module):
    def __init__(self, in_channels):
        super(CovarianceModule, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print(x, 'x')
        b, c, _, _ = x.size()
        # 计算特征图的均值
        avg = self.pool(x).view(b, c, 1, 1)
        # 计算特征图的方差
        var = (self.pool(x * x).view(b, c, 1, 1) - avg ** 2).clamp(min=1e-8)
        # 计算归一化的特征图
        x_norm = (x - avg) / torch.sqrt(var)
        # 计算特征图的协方差矩阵
        cov = self.conv1(x_norm) * self.conv2(x_norm)
        # 计算特征图的平均协方差矩阵
        avg_cov = self.pool(cov).view(b, c, 1, 1)
        # 计算特征图的最终协方差矩阵
        final_cov = self.conv3(avg_cov)
        # 计算特征图的权重，使用 sigmoid 函数将权重限制在 [0, 1] 范围内
        weight = self.sigmoid(final_cov)
        # 对特征图进行加权
        x = x * weight
        return x


# 黎曼流行的自注意力机制
# 定义自注意力模块
# class SelfAttention_Covariance(nn.Module):
#     def __init__(self, in_dim, activation=F.relu):
#         super(SelfAttention_Covariance, self).__init__()
#         self.chanel_in = in_dim
#         self.activation = activation
#         self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
#         self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
#         self.gamma = nn.Parameter(torch.zeros(1))
#     def forward(self, x):
#         m_batchsize, C, width, height = x.size()
#         proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
#         proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
#         energy = torch.bmm(proj_query, proj_key)
#         attention = F.softmax(energy, dim=-1)
#         proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
#         out = torch.bmm(proj_value, attention.permute(0, 2, 1))
#         out = out.view(m_batchsize, C, width, height)
#         out = self.gamma*out + x
#         return out

# class ViT(nn.Module):
#     def __init__(self, out_channels):
#         self.out_channels = out_channels
#
#     def forward(self, x):
#         transforms.Resize((224, 224)),
#
#
#
#         return x


# class SelfAttention(nn.Module):
#     def __init__(self, channels, size, num_head):
#         super(SelfAttention, self).__init__()
#
#         self.channels = channels
#         self.size = size
#         self.num_head = num_head
#         self.head_dim = channels // num_head
#         self.q_proj = nn.Linear(channels, channels)
#         self.k_proj = nn.Linear(channels, channels)
#         self.v_proj = nn.Linear(channels, channels)
#         self.out_proj = nn.Linear(channels, channels)
#
#     def forward(self, x):
#         x = x.view(-1, self.channels, self.size * self.size).transpose(1, 2)
#         q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
#         q = q.reshape(x.size(0), x.size(1), self.num_head, self.head_dim)
#         k = k.reshape(x.size(0), x.size(1), self.num_head, self.head_dim)
#         v = v.reshape(x.size(0), x.size(1), self.num_head, self.head_dim)
#
#         out = flash_attn_func(q, k, v)
#         out = out.contiguous().view(x.size(0), x.size(1), self.channels)
#         out = self.out_proj(out)
#         return out.transpose(2, 1).contiguous().view(-1, self.channels, self.size, self.size)


class SelfAttention(nn.Module):
    def __init__(self, channels, size, use_DropKey=True, mask_ratio=0.1):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.use_DropKey = use_DropKey
        self.mask_ratio = mask_ratio
        self.mha = nn.MultiheadAttention(channels, 8, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).transpose(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)

        if self.use_DropKey:  # Only apply DropKey during training
            mask = torch.bernoulli(torch.ones_like(attention_value) * self.mask_ratio) * -1e12
            attention_value = attention_value + mask

        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        x = attention_value.transpose(1, 2).contiguous().view(-1, self.channels, self.size, self.size)
        return x


# --------------------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),

            # # 增加协方差矩阵
            # CovarianceModule(out_channels),
        )

    def forward(self, x):

        if self.residual:

            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )
        # self.relu=nn.ReLU()
        self.emb_layer = nn.Sequential(

            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

        # self.vit = VIT(out_channels)

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        # res=x
        #x=self.cpca(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        #  # vit(x)
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, attn=False, emb_dim=256, num=8):
        super().__init__()
        self.attn = attn
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.mfmrm=MFMRM(in_channels)

        self.emb_layer = nn.Sequential(

            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x=self.mfmrm(x, skip_x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb
        #return x


# class UNet_conditional(nn.Module):
#     def __init__(self, c_in=4, c_out=4, con_c_in=3, time_dim=256, device="cuda"):
#         super().__init__()
#         self.encoder = encoder(in_ch=con_c_in)
#         # self.x_vit = ViT(img_size=256, patch_size = 16, in_channels = c_in, num_classes= time_dim, embed_dim = 768, num_heads=12, num_layers = 12, hidden_dim = 3072, dropout = 0.1)
#         # self.y_vit = ViT(img_size=256, patch_size=16, in_channels=con_c_in, num_classes=time_dim, embed_dim=768, num_heads=12,
#         #                num_layers=12, hidden_dim=3072, dropout=0.1)
#         self.device = device
#         self.time_dim = time_dim
#         self.inc = DoubleConv(c_in, 32)  # 64
#         self.cov = CovarianceModule(c_in)

#         self.down1 = Down(32, 64)  # 64 128 112

#         self.down2 = Down(64, 128)  # 128  256  56

#         self.down3 = Down(128, 256)  # 256  256 28

#         self.down4 = Down(256, 512)  # 256  256 14

#         self.down5 = Down(512, 512)  # 256  256 7

#         self.bot1 = DoubleConv(512, 1024)  # 256  512
#         self.bot2 = DoubleConv(1024, 1024)  # 512  512
#         self.bot3 = DoubleConv(1024, 512)  # 512  256

#         self.gab1=group_aggregation_bridge(512,512)
#         self.gab2=group_aggregation_bridge(512,256)
#         self.gab3=group_aggregation_bridge(256,128)
#         self.gab4=group_aggregation_bridge(128,64)
#         self.gab5=group_aggregation_bridge(64,32)

#         self.gt_conv1 = nn.Sequential(nn.Conv2d(512, 1, 1))
#         self.gt_conv2 = nn.Sequential(nn.Conv2d(256, 1, 1))
#         self.gt_conv3 = nn.Sequential(nn.Conv2d(128, 1, 1))
#         self.gt_conv4 = nn.Sequential(nn.Conv2d(64, 1, 1))
#         self.gt_conv5 = nn.Sequential(nn.Conv2d(32, 1, 1))

#         self.dbn = nn.GroupNorm(4, 512)

#         self.up1 = Up(1024, 256)  # 512  256

#         self.up2 = Up(512, 128)  # 256 64

#         self.up3 = Up(256, 64)  # 128 64

#         self.up4 = Up(128, 32)  # 128 64

#         self.sa1 = SelfAttention(1024,8)
#         self.sa2 = SelfAttention(1024,8)
#         self.sa3 = SelfAttention(512,8)


#         self.up5 = Up(64, 32)  # 128 64

#         self.outc = nn.Conv2d(32, c_out, kernel_size=1)

#         # self.inLiner = nn.Linear(in_features=1024,out_features=256)

#     def pos_encoding(self, t, channels):
#         inv_freq = 1.0 / (
#                 10000
#                 ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
#         )
#         pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
#         pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
#         pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
#         return pos_enc

#     def forward(self, x, t, y):
#         # print(y.shape)
#         t = t.unsqueeze(-1).type(torch.float)
#         t = self.pos_encoding(t, self.time_dim)
#         # x = self.cov(x)
#         # y = self.y_vit(y)
#         # print(y.shape,'y.shape')
#         # x_t = self.x_vit(x)
#         # print(x_t.shape, 't')
#         # t = x_t + t
#         ex_x_fea = self.encoder(y)

#         # print(x.shape, 'x')
#         # x = self.atten_cov(x)
#         # print(x.shape, 'atten_x')
#         x1 = self.inc(x) + ex_x_fea[0]
#         x2 = self.down1(x1, t) + ex_x_fea[1]
#         x3 = self.down2(x2, t) + ex_x_fea[2]
#         x4 = self.down3(x3, t) + ex_x_fea[3]
#         x5 = self.down4(x4, t) + ex_x_fea[4]
#         x6 = self.down5(x5, t)+ ex_x_fea[5]

#         x6 = self.bot1(x6)
#         x6 = self.sa1(x6)


#         x6 = self.bot2(x6)
#         x6 = self.sa2(x6)


#         x6 = self.bot3(x6)
#         x6 = self.sa3(x6)

#         pre_t1=F.gelu(F.interpolate(self.gt_conv1(F.gelu(self.dbn(x6))), scale_factor=(2, 2), mode='bilinear',
#                                     align_corners=True))
#         t1=self.gab1(x6,x5,pre_t1)
#         x = self.up1(x6, t1, t)
#         # # x = self.att_4(x)

#         pre_t2=F.gelu(F.interpolate(self.gt_conv2(x), scale_factor=(2, 2), mode='bilinear',
#                                     align_corners=True))
#         t2=self.gab2(t1,x4,pre_t2)
#         x = self.up2(x, t2, t)

#         pre_t3=F.gelu(F.interpolate(self.gt_conv3(x), scale_factor=(2, 2), mode='bilinear',
#                                     align_corners=True))
#         t3=self.gab3(t2,x3,pre_t3)
#         x = self.up3(x, t3, t)

#         pre_t4=F.gelu(F.interpolate(self.gt_conv4(x), scale_factor=(2, 2), mode='bilinear',
#                                     align_corners=True))
#         t4=self.gab4(t3,x2,pre_t4)
#         x = self.up4(x, t4, t)

#         pre_t5=F.gelu(F.interpolate(self.gt_conv5(x), scale_factor=(2, 2), mode='bilinear',
#                                     align_corners=True))
#         t5=self.gab5(t4,x1,pre_t5)
#         x = self.up5(x, t5, t)

#         output = self.outc(x)
#=============================================
class UNet_conditional(nn.Module):
    def __init__(self, c_in=4, c_out=4, con_c_in=3, time_dim=256, device="cuda"):
        super().__init__()
        self.encoder = encoder(in_ch=con_c_in)
        # self.revln = RevIN(c_in)
        # self.revln2 = RevIN(12)
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 32)  # 64
        #self.cov = CovarianceModule(c_in)
        #self.cross0 = Cross_MultiAttention(32,32,2)
        self.down1 = Down(32, 64)  # 64 128 112
        #self.cross1=Cross_MultiAttention(64,64,2)
        self.down2 = Down(64, 128)  # 128  256  56
        #self.cross2 = Cross_MultiAttention(128, 128, 4)
        self.down3 = Down(128, 256)  # 256  256 28
        #self.cross3 = Cross_MultiAttention(256, 256, 8)
        self.down4 = Down(256, 512)  # 256  256 14
        self.down5 = Down(512, 512)  # 256  256 7
        # self.down4 = StarDown(256, 512)  # 256  256 14
        # self.down5 = StarDown(512, 512)  # 256  256 7
        #self.cross4 = Cross_MultiAttention(512, 512, 8)
        #self.cross5 = Cross_MultiAttention(512, 512, 8)
        self.bot1 = DoubleConv(512, 1024)  # 256  512
        self.bot2 = DoubleConv(1024, 1024, residual=True)  # 512  512
        self.bot3 = DoubleConv(1024, 512)  # 512  256
        # self.bot1 = Block(512, 1024)  # 256  512
        # self.bot2 = Block(1024, 1024)  # 512  512
        # self.bot3 = Block(1024, 512)  # 512  256

        self.sa1 = SelfAttention(1024, 8)
        self.sa2 = SelfAttention(1024, 8)
        self.sa3 = SelfAttention(512, 8)

        # self.up1 = StarUp(1024, 256)  # 512  256
        # self.up2 = StarUp(512, 128)  # 256 64

        self.up1 = Up(1024, 256)  # 512  256
        self.up2 = Up(512, 128)  # 256 64

        self.up3 = Up(256, 64)  # 128 64

        self.up4 = Up(128, 32)  # 128 64

        self.up5 = Up(64, 32)  # 128 64

        self.outc = nn.Conv2d(32, c_out, kernel_size=1)



    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    # def pos_encoding(self,timesteps, dim, max_period=10000):
    #     half = dim // 2
    #     freqs = torch.exp(
    #         -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    #     ).to(device=timesteps.device)
    #     args = timesteps[:, None].float() * freqs[None]
    #     embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    #     if dim % 2:
    #         embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    #     return embedding.squeeze(dim=1)

    def forward(self, x, t, y):
        # print(y.shape)
        t = t.unsqueeze(-1).type(torch.float)

        t = self.pos_encoding(t, self.time_dim)

        # x = self.cov(x)max_period
        # y = self.y_vit(y)
        # print(y.shape,'y.shape')
        # x_t = self.x_vit(x)
        # print(x_t.shape, 't')
        # t = x_t + t

        ex_x_fea = self.encoder(y)


        x1 = self.inc(x) + ex_x_fea[0]

        x2 = self.down1(x1, t) + ex_x_fea[1]

        x3 = self.down2(x2, t) + ex_x_fea[2]

        x4 = self.down3(x3, t) + ex_x_fea[3]

        x5 = self.down4(x4, t)+ ex_x_fea[4]

        x6 = self.down5(x5, t)+ ex_x_fea[5]
        # x1 = self.inc(x) + ex_x_fea[0]
        #
        # x2 = self.down1(x1, t) + ex_x_fea[1]
        #
        # x3 = self.down2(x2, t) + ex_x_fea[2]
        #
        # x4 = self.cross3(self.down3(x3, t) , ex_x_fea[3])
        #
        # x5 = self.cross4(self.down4(x4, t) , ex_x_fea[4])
        #
        # x6 = self.cross5(self.down5(x5, t) , ex_x_fea[5])

        x6 = self.bot1(x6)
        x6 = self.sa1(x6)

        x6 = self.bot2(x6)
        x6 = self.sa2(x6)

        x6 = self.bot3(x6)
        x6 = self.sa3(x6)

        x = self.up1(x6, x5, t)

        x = self.up2(x, x4, t)

        x = self.up3(x, x3, t)

        x = self.up4(x, x2, t)

        x = self.up5(x, x1, t)

        output = self.outc(x)

        return output



# if __name__ == '__main__':
#     # net = UNet(device="cpu")
#     net = UNet_conditional(num_classes=10, device="cpu")
#     print(sum([p.numel() for p in net.parameters()]))
#     x = torch.randn(3, 1, 256, 256)
#     t = x.new_tensor([500] * x.shape[0]).long()
#     y = x.new_tensor([1] * x.shape[0]).long()
#     print(net(x, t, y).shape)
