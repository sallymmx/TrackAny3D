import math
from .utils import pytorch_utils as pt_utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
from .MOE import  SparseMoE
class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()

        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size


    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center,id = misc.arange(xyz, self.num_group)  # B G 3

        # knn to get the neighborhood
        # _, idx = self.knn(xyz, center)  # B G M
        idx = knn_point(self.group_size, xyz,center)
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center,id


## Transformers
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 adapter_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, init_values=None, rank=64,
                 depth=0,num_group = 128):
        super().__init__()
        self.depth = depth
        self.norm1 = norm_layer(dim)
        self.num_group = num_group
        self.dim = dim
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.Adapter_MLP = Adapter(d_model=dim, bottleneck=rank, dropout=adapter_drop)
        self.Adapter_atten= Adapter(d_model=dim, bottleneck=rank, dropout=adapter_drop)
        self.non_linear_func = nn.GELU()
        if depth%2==0:
            self.expert=SparseMoE(n_embed=dim,num_experts=8,top_k=4)

        else:
            self.expert=None


    def forward(self, x,mask_refs):

        xx =self.norm1(x)
        adapter_attn = self.Adapter_atten(xx)
        x = x + self.drop_path1((self.attn(xx))+adapter_attn)
        h = x
        x = self.norm2(x)
        adapter_mlp = self.Adapter_MLP(x)
        x = self.mlp(x) + adapter_mlp
        if self.expert:
            cls_token = x[:, :1, :]
            search_token = x[:, -self.num_group:, :]
            xx = torch.cat([cls_token, search_token], dim=1)
            xx = self.expert(xx)
            cls_token = xx[:, :1, :]
            search_token = xx[:, -self.num_group:, :]
            x = torch.cat([cls_token, x[:, 1:-self.num_group], search_token], dim=1)
        x = h + self.drop_path2(x)

        return x


class Adapter(nn.Module):
    def __init__(self,
                 d_model=None,
                 out_dim=None,
                 bottleneck=None,
                 dropout=0.0,
                 adapter_layernorm_option="in",
                 use_square=False, ):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck
        self.use_square = use_square
        # _before
        self.adapter_layernorm_option = adapter_layernorm_option
        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)
        self.scale = nn.Linear(self.n_embd, 1)
        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.GELU()
        if out_dim is None:
            self.up_proj = nn.Linear(self.down_size, self.n_embd)
        else:
            self.up_proj = nn.Linear(self.down_size, out_dim)

        self.dropout = dropout

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)
            nn.init.kaiming_uniform_(self.scale.weight, a=math.sqrt(5))
            nn.init.zeros_(self.scale.bias)
            nn.init.constant_(nn.LayerNorm(self.n_embd).weight, 1.0)
            nn.init.constant_(nn.LayerNorm(self.n_embd).bias, 0.0)



    def forward(self, x, add_residual=False, residual=None):
        #


        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)
        scale = F.relu(self.scale(x))
        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)
        up = up * scale
        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)
        if add_residual:
            output = up + residual
        else:
            output = up


        return output





class PointTransformer_DAPT(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims
        self.drop_adapter_rate = config.drop_adapter_rate if hasattr(config, 'drop_adapter_rate') else 0.
        self.rank = config.rank

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.learn = nn.Parameter(torch.ones(1, self.num_group))

        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
            drop_adapter_rate=self.drop_adapter_rate,
            rank=self.rank,
            num_group=self.num_group
        )

        self.class_token = config.class_token
        self.prompt_token = config.prompt_token
        self.patch_token = config.patch_token

        self.HEAD_CHANEL = 0
        if self.class_token != 'None':
            self.HEAD_CHANEL += 1
        if self.prompt_token != 'None':
            self.HEAD_CHANEL += 1
        if self.patch_token != 'None':
            self.HEAD_CHANEL += 1

        assert self.HEAD_CHANEL != 0

        self.norm = nn.LayerNorm(self.trans_dim)
        self.fc_mask = (
            pt_utils.Seq(self.trans_dim)
            .conv1d(self.trans_dim, bn=True)
            .conv1d(self.trans_dim, bn=True)
            .conv1d(self.trans_dim, bn=True)
            .conv1d(128, activation=None)
        )

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)


        self.fc1 = (
            pt_utils.Seq(self.trans_dim)
            # .conv1d(cfg.out_channels, bn=True)
            .conv1d(128, activation=None)
        )

        self.mask_emb = (
            pt_utils.Seq(1)
            .conv1d(self.trans_dim, activation=None)
        )

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path,map_location=torch.device('cuda:1'))
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('ACT_encoder'):
                    base_ckpt[k[len('ACT_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('transformer_k'):
                    base_ckpt[k[len('transformer_k.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)


            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
            for name, param in self.named_parameters():  # Use `self.named_parameters()` here
                if name not in base_ckpt:
                    continue

                param.requires_grad = False
                print(f"Loaded {name} and frozen")

        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self,pts,mask_refs,cls_token,bc_refs):
        neighborhood, center, idx = self.group_divider(pts)

        mask_refs = mask_refs.reshape(-1, mask_refs.shape[-1])

        mask_refs = torch.gather(mask_refs, dim=1, index=idx.to(torch.int64))
        mask_refs = mask_refs.to(torch.float32)
        learn = self.learn.expand(mask_refs.size(0), -1)

        mask_refs = mask_refs * learn
        mask_refs = mask_refs.unsqueeze(1)

        mask_refs = self.mask_emb(mask_refs).permute(0, 2, 1)

        group_input_tokens = self.encoder(neighborhood)  # B G N





        group_input_tokens = group_input_tokens+mask_refs
        group_input_tokens = group_input_tokens.reshape(-1, 2 * group_input_tokens.shape[-2],
                                                        group_input_tokens.shape[-1])

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)

        if cls_token is None:
            cls_tokens = cls_tokens
        else:
            cls_tokens = cls_tokens + cls_token
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)  # 32,1,384
        pos = self.pos_embed(center).reshape(-1, group_input_tokens.shape[-2], group_input_tokens.shape[-1])
        pcd_pos = pos
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)

        x = x + pos
        pcd = center.reshape(-1, 2, center.shape[-2], center.shape[-1])

        x = self.blocks(x, mask_refs)

        x = self.norm(x)


        cls_token = x[:, 0, :].unsqueeze(1)
        search_feat = x[:, -self.num_group:, ]
        att = torch.matmul(search_feat, cls_token.transpose(1, 2))
        search_feat = (search_feat * att)
        search_feat = search_feat.permute(0, 2, 1)
        mask_feat = self.fc_mask(search_feat)


        return dict(
            xyz=center,
            feat=self.fc1(search_feat),
            idx=idx.long(),
            mask_feat=mask_feat,
            cls_token=cls_token,
            # mask_preds = mask_preds
        )


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., drop_adapter_rate=0., rank=64,num_group = 128):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                adapter_drop=drop_adapter_rate, rank=rank,
                depth=i,
                num_group = num_group
            )
            for i in range(depth)])

    def forward(self, x, mask_ref):

        for _, block in enumerate(self.blocks):

            x = block(x,mask_ref)

        return x




class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=4, mask=False):
        super().__init__()


        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads



    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.permute(0,1,3,2) # (num_heads * batch_size, ..., head_dim, src_length)exi

        attn_score = torch.matmul(query,key) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)



        return out





