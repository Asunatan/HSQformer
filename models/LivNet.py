import math

import torch
from matplotlib import pyplot as plt
from timm.layers import to_2tuple, trunc_normal_
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange,repeat
from .convnext_segmentaion import ConvNeXt
from timm.models.layers import  DropPath
# from .swin_transformer import SwinTransformer
from .swin_transformer import SwinTransformer
from .cswin_transformer_segmentation import CSWin


class SwitchTransformersTop1Router(nn.Module):
    """
    Router using tokens choose top-1 experts assignment.

    This router uses the same mechanism as in Switch Transformer (https://arxiv.org/abs/2101.03961) and V-MoE
    (https://arxiv.org/abs/2106.05974): tokens choose their top experts. Items are sorted by router_probs and then
    routed to their choice of expert until the expert's expert_capacity is reached. **There is no guarantee that each
    token is processed by an expert**, or that each expert receives at least one token.

    """

    def __init__(self,embed_dim, num_experts=4, top_k=1,dropout=0. ):
        super().__init__()
        self.num_experts = num_experts
        # self.expert_capacity = config.expert_capacity
        self.route_linear = nn.Linear(embed_dim, self.num_experts, bias=False)
        self.jitter_noise = 0.01
        self.expert_capacity=float('inf')

    def forward(self, hidden_states: torch.Tensor):

        if self.training and self.jitter_noise > 0:
            # Multiply the token inputs by the uniform distribution - adding some noise
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)

        # Shape: [batch_size, sequence_length, hidden_dim]
        router_logits = self.route_linear(hidden_states)

        # Apply Softmax and cast back to the original `dtype`
        router_probs = router_logits.softmax(dim=-1)

        expert_index = torch.argmax(router_probs, dim=-1)
        expert_index = torch.nn.functional.one_hot(expert_index, num_classes=self.num_experts)

        # Mask tokens outside expert capacity. Sum over each sequence
        # token_priority = torch.cumsum(expert_index, dim=-2)
        # # mask if the token routed to to the expert will overflow
        # expert_capacity_mask = token_priority <= self.expert_capacity
        # expert_index = expert_index * expert_capacity_mask

        router_probs = torch.max(router_probs, dim=-1).values.unsqueeze(-1)
        return expert_index, router_probs, router_logits

class SwitchTransformersSparseMLP(nn.Module):
    def __init__(self, embed_dim, num_experts=4, top_k=1,dropout=0.):
        super().__init__()
        self.router = SwitchTransformersTop1Router(embed_dim, num_experts, top_k)
        self.experts = nn.ModuleList([MLPLayer(embed_dim,dropout=dropout) for _ in range(num_experts)])
        self.top_k = top_k

    def forward(self, hidden_states):
        r"""
              Hold on, this will be slightly tricky to understand In the correct order, a MoE layer does the following:

              1- Gets the `router_mask` from the router. The shape of the mask is `(batch_size, sequence_length, num_expert)`
              and corresponds to the argmax of the `router_probs`. The probabilities are needed in the computation of the
              hidden states : they are broadcasted to the hidden states values (can be interpreted as a scaling factor).

              2- Dispatch the tokens to its associated experts. We do a classic for loop over the experts and assign for each
              expert the corresponding hidden states.

              """
        # Step 1: Get the router_mask from the router as wel as the probabilities
        router_mask, router_probs, router_logits = self.router(hidden_states)
        # expert_index = torch.argmax(router_mask, dim=-1)

        # The routers introduced might not always map all the tokens, to a router, which means that some hidden states
        # can be unchanged from one layer to another. That is why the hidden states are cloned before updating only the seleced ones.

        next_states = hidden_states.clone()

        router_mask = router_mask.bool()
        #router_mask -> [batch_size, sequence_length, 专家编号的one hot]
        #下面三行可以去掉，不考虑容量的话
        batch_size, seq_len, num_experts = router_mask.shape
        idx_mask = router_mask.transpose(1, 2).reshape(batch_size * seq_len, num_experts).sum(dim=0)
        idx_mask = torch.nonzero(idx_mask, as_tuple=True)[0].tolist()  # length: number of "activated" expert / value: index
        for idx in idx_mask:
            next_states[router_mask[:, :, idx]] = self.experts[idx](hidden_states[router_mask[:, :, idx]]).float()

        hidden_states = router_probs * next_states
        return hidden_states


# noisy top-k gating
class NoisyTopkRouter(nn.Module):
    def __init__(self, embed_dim, num_experts, top_k=1,dropout=0.):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        # layer for router logits
        self.topkroute_linear = nn.Linear(embed_dim, num_experts)
        self.noise_linear = nn.Linear(embed_dim, num_experts)

    def forward(self, mh_output):
        # mh_ouput is the output tensor from multihead self attention block
        logits = self.topkroute_linear(mh_output)

        # Noise logits
        noise_logits = self.noise_linear(mh_output)

        # Adding scaled unit gaussian noise to the logits

        # noise = torch.randn_like(logits) * F.softplus(noise_logits)
        # noisy_logits = logits + noise
        # noisy_logits = noisy_logits.softmax(-1)
        # # top_k_probs, indices = noisy_logits.topk(self.top_k, dim=-1)
        # return noisy_logits

        #一
        # noise = torch.randn_like(logits) * F.softplus(noise_logits)
        # noisy_logits = logits + noise
        # noisy_logits = noisy_logits.softmax(-1)
        # top_k_probs, indices = noisy_logits.topk(self.top_k, dim=-1)
        # return top_k_probs,indices

        #二
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices

class NoisyDenseRouter(nn.Module):
    def __init__(self, embed_dim, num_experts, top_k=1):
        super(NoisyDenseRouter, self).__init__()

        # layer for router logits
        self.route_linear = nn.Linear(embed_dim, num_experts)
        self.noise_linear = nn.Linear(embed_dim, num_experts)

    def forward(self, mh_output):
        # mh_ouput is the output tensor from multihead self attention block
        logits = self.route_linear(mh_output)

        # Noise logits
        noise_logits = self.noise_linear(mh_output)

        # Adding scaled unit gaussian noise to the logits
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices

#Now create the sparse mixture of experts module
class SparseMoE_token(nn.Module):
    def __init__(self, embed_dim, num_experts=4, top_k=1,dropout=0.,out_features=None,shared=False):
        super(SparseMoE_token, self).__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.topkroute_linear = nn.Linear(embed_dim, num_experts)
        self.noise_linear = nn.Linear(embed_dim, num_experts)
        self.out_features = out_features or embed_dim
        self.experts = nn.ModuleList([MLPLayer(embed_dim,dropout=dropout,out_features=self.out_features) for _ in range(num_experts)])
        self.shared = shared
        if shared:
            self.shared_expert = MLPLayer(embed_dim,dropout=dropout)
            self.shared_expert_gate = nn.Linear(embed_dim, 1)
            self.shared_expert_noise = nn.Linear(embed_dim, 1)



    def forward(self, x):

        #Shared expert
        if self.shared:
            shared_expert_gatelogit = self.shared_expert_gate(x)
            shared_expert_noise = torch.randn_like(shared_expert_gatelogit) * F.softplus(self.shared_expert_noise(x))
            shared_expert_noise_logits = shared_expert_gatelogit + shared_expert_noise
            shared_expert_output = self.shared_expert(x)
            shared_expert_output = F.sigmoid(shared_expert_noise_logits) * shared_expert_output


        #Sparse expert
        logits = self.topkroute_linear(x)
        # Noise logits
        noise_logits = self.noise_linear(x)
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_probs = F.softmax(sparse_logits, dim=-1)

        if self.out_features is not None:
            final_output = torch.zeros(x.size(0), x.size(1), self.out_features,device=x.device)
        else:
            final_output = torch.zeros_like(x)

        # Reshape inputs for batch processing
        flat_x = x.view(-1, x.size(-1))
        flat_router_probs = router_probs.view(-1, router_probs.size(-1))

        # Process each expert in parallel
        for i, expert in enumerate(self.experts):
            # Create a mask for the inputs where the current expert is in top-k
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)

            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)

                # Extract and apply gating scores
                gating_scores = flat_router_probs[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores

                # Update final output additively by indexing and adding
                final_output[expert_mask] += weighted_output.squeeze(1)
        if self.shared:
            return final_output + shared_expert_output
        return final_output

class SparseMoE_expert(nn.Module):
    def __init__(self, embed_dim, num_experts=4, top_k=1,dropout=0.,shared=False):
        super(SparseMoE_expert, self).__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.topkroute_linear = nn.Linear(embed_dim, num_experts)
        self.noise_linear = nn.Linear(embed_dim, num_experts)
        self.experts = nn.ModuleList([MLPLayer(embed_dim,dropout=dropout) for _ in range(num_experts)])
        self.shared = shared

        if shared:
            self.shared_expert = MLPLayer(embed_dim,dropout=dropout)
            self.shared_expert_gate = nn.Linear(embed_dim, 1)
            self.shared_expert_noise = nn.Linear(embed_dim, 1)



    def forward(self, x):

        #Shared expert
        if self.shared:
            shared_expert_gatelogit = self.shared_expert_gate(x)
            shared_expert_noise = torch.randn_like(shared_expert_gatelogit) * F.softplus(self.shared_expert_noise(x))
            shared_expert_noise_logits = shared_expert_gatelogit + shared_expert_noise
            shared_expert_output = self.shared_expert(x)
            shared_expert_output = F.sigmoid(shared_expert_noise_logits) * shared_expert_output


        #Sparse expert
        logits = self.topkroute_linear(x)
        # Noise logits
        noise_logits = self.noise_linear(x)
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise
        noisy_logits =noisy_logits.mean(dim=(0, 1))
        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_probs = F.softmax(sparse_logits, dim=-1)

        final_output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            expert_output = expert(x) * router_probs[i]
            final_output += expert_output

        if self.shared:
            return final_output + shared_expert_output
        return final_output

class DenseMoE_token(nn.Module):#SoftMoE
    def __init__(self, embed_dim, num_experts=4, top_k=1,dropout=0.,shared=False):
        super(DenseMoE_token, self).__init__()
        # self.router = NoisyTopkRouter(embed_dim, num_experts, top_k)
        self.top_k = top_k
        self.num_experts = num_experts
        self.topkroute_linear = nn.Linear(embed_dim, num_experts)
        self.noise_linear = nn.Linear(embed_dim, num_experts)
        self.experts = nn.ModuleList([MLPLayer(embed_dim,dropout=dropout) for _ in range(num_experts)])
        self.shared = shared
        if shared:
            self.shared_expert = MLPLayer(embed_dim,dropout=dropout)
            self.shared_expert_gate = nn.Linear(embed_dim, 1)
            self.shared_expert_noise = nn.Linear(embed_dim, 1)



    def forward(self, x):

        #Shared expert
        if self.shared:
            shared_expert_gatelogit = self.shared_expert_gate(x)
            shared_expert_noise = torch.randn_like(shared_expert_gatelogit) * F.softplus(self.shared_expert_noise(x))
            shared_expert_noise_logits = shared_expert_gatelogit + shared_expert_noise
            shared_expert_output = self.shared_expert(x)
            shared_expert_output = F.sigmoid(shared_expert_noise_logits) * shared_expert_output


        #Sparse expert
        logits = self.topkroute_linear(x)
        # Noise logits
        noise_logits = self.noise_linear(x)
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_probs = F.softmax(sparse_logits, dim=-1)

        final_output = torch.zeros_like(x)

        # Reshape inputs for batch processing
        flat_x = x.view(-1, x.size(-1))
        flat_router_probs = router_probs.view(-1, router_probs.size(-1))

        # Process each expert in parallel
        for i, expert in enumerate(self.experts):
            # Create a mask for the inputs where the current expert is in top-k
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)

            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)

                # Extract and apply gating scores
                gating_scores = flat_router_probs[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores

                # Update final output additively by indexing and adding
                final_output[expert_mask] += weighted_output.squeeze(1)
        if self.shared:
            return final_output + shared_expert_output
        return final_output

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # elif self.data_format == "channels_first":
        #     u = x.mean(1, keepdim=True)
        #     s = (x - u).pow(2).mean(1, keepdim=True)
        #     x = (x - u) / torch.sqrt(s + self.eps)
        #     x = self.weight[:, None, None] * x + self.bias[:, None, None]
        #     return x
        elif self.data_format == "channels_first":
            mean = x.mean(1, keepdim=True)
            std = x.std(1, keepdim=True)
            return self.weight[:, None, None] * (x - mean) / (std + self.eps) + self.bias[:, None, None]
class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x
class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.dwconv_small = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        # self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
        #                           requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)+self.dwconv_small(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        # if self.gamma is not None:
        #     x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x



class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        #context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        #context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        if mask is not None:
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class SelfAttentionLayer(nn.Module):

    def __init__(self, embed_dim, num_heads=8, dropout=0.0,normalize_before=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout,batch_first=True)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, query_key_value,query_pos=None,attn_mask = None,key_padding_mask = None):

        x = self.self_attn(query=query_key_value, key=query_key_value, value=query_key_value, attn_mask=attn_mask, need_weights=False,
                                  key_padding_mask=key_padding_mask)[0]
        return self.dropout(x)



class CrossAttentionLayer(nn.Module):

    def __init__(self, embed_dim, num_heads=8, dropout=0.0,normalize_before=True):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout,batch_first=True)

        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, query, key_value, memory_mask = None, memory_key_padding_mask = None,
                kv_pos = None,query_pos = None):
        att = self.multihead_attn(query=query,key=key_value,value=key_value, attn_mask=memory_mask,
                                       need_weights=False,key_padding_mask=memory_key_padding_mask)[0]
        return  self.dropout(att)

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (torch.div(dim_t, 2, rounding_mode='floor')) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        # _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)

class PatchEmbed(nn.Module):

    def __init__(self, patch_size=7,img_size=224,in_chans=3, out_channel=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        #self.proj = nn.Conv2d(in_chans, out_channel, kernel_size=patch_size, stride=patch_size,)
        self.proj = nn.Conv2d(in_chans, out_channel, kernel_size=3, stride=1,padding=1)
        self.norm = nn.LayerNorm(out_channel)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_patches, out_channel))
        #self.proj_linear=nn.Linear(out_channel,out_channel)
        #self.dropout = nn.Dropout(0.1)


    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(-1, -2)#+self.position_embeddings
        #x=self.proj_linear(x)
        #x = self.dropout(self.norm(x))
        #x=self.norm(x)
        return x


class SCAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.0, serial_parallel=None,):
        super().__init__()
        self.serial_parallel=serial_parallel

        self.conv_self_norm = nn.LayerNorm(embed_dim)
        self.swin_self_norm = nn.LayerNorm(embed_dim)
        self.conv_pre_norm = nn.LayerNorm(embed_dim)
        self.swin_pre_norm = nn.LayerNorm(embed_dim)
        self.self_attention_swin = SelfAttentionLayer(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.cross_attention_swin = CrossAttentionLayer(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)


        self.self_attention_conv = SelfAttentionLayer(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.cross_attention_conv = CrossAttentionLayer(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)


    def forward(self, x_swin,x_conv):
        #B, N, C = kv.shape
        if self.serial_parallel == 'serial':
            x_swin_norm = self.swin_pre_norm(x_swin)
            x_conv_norm = self.conv_pre_norm(x_conv)
            x_swin_ = self.cross_attention_swin(x_swin_norm,x_conv_norm) + x_swin
            x_conv_ = self.cross_attention_conv(x_conv_norm, x_swin_norm) + x_conv

            x_swin_ = self.self_attention_swin(self.swin_self_norm(x_swin_)) + x_swin_
            x_conv_ = self.self_attention_conv(self.conv_self_norm(x_conv_)) + x_conv_
            return x_conv_+x_swin_

        elif self.serial_parallel == 'parallel':
            x_swin_cross_norm = self.swin_pre_norm(x_swin)#cross_normal
            x_conv_cross_norm = self.conv_pre_norm(x_conv)
            x_conv2swin = self.cross_attention_swin(x_swin_cross_norm, x_conv_cross_norm)
            x_swin2conv = self.cross_attention_conv(x_conv_cross_norm, x_swin_cross_norm)
            x_swin_ = self.self_attention_swin(self.swin_self_norm(x_swin))
            x_conv_ = self.self_attention_conv(self.conv_self_norm(x_conv))
            return x_swin_+x_conv2swin+x_conv_+x_swin2conv+x_swin+x_conv

        elif self.serial_parallel == 'invert_serial':
            x_swin_ = self.self_attention_swin(self.swin_self_norm(x_swin)) + x_swin
            x_conv_ = self.self_attention_conv(self.conv_self_norm(x_conv)) + x_conv

            x_swin_norm = self.swin_pre_norm(x_swin_)#cross_normal
            x_conv_norm = self.conv_pre_norm(x_conv_)
            x_swin_ = self.cross_attention_swin(x_swin_norm,x_conv_norm) + x_swin_
            x_conv_ = self.cross_attention_conv(x_conv_norm, x_swin_norm) + x_conv_

            return x_conv_+x_swin_
        else:
            raise Exception("args.serial_parallel error")

class SC_MoEBlock(nn.Module):
    def __init__(self, embed_dim,seq_length=None,num_heads=8, num_experts=4, top_k=1,dropout=0.,shared=False,args=None):
        super().__init__()
        # self.pos_embedding_swin = nn.Parameter(torch.empty(1, seq_length, embed_dim).normal_(std=0.02))
        # self.pos_embedding_conv = nn.Parameter(torch.empty(1, seq_length, embed_dim).normal_(std=0.02))
        # self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, embed_dim).normal_(std=0.02))
        self.sclayer = SCAttentionLayer(embed_dim=embed_dim,num_heads=num_heads,serial_parallel=args.serial_parallel)


        self.moe_norm = nn.LayerNorm(embed_dim)
        if args.sparse_dense=='sparse_token':
            self.moe = SparseMoE_token(embed_dim=embed_dim,num_experts=num_experts,
                                       top_k=top_k,dropout=dropout,shared=shared)
        elif args.sparse_dense=='dense_token':
            self.moe = SparseMoE_token(embed_dim=embed_dim,num_experts=num_experts,
                                       top_k=num_experts,dropout=dropout,shared=shared)
        elif args.sparse_dense=='sparse_expert':
            self.moe = SparseMoE_expert(embed_dim=embed_dim,num_experts=num_experts,
                                       top_k=top_k,dropout=dropout,shared=shared)
        elif args.sparse_dense=='dense_expert':
            self.moe = SparseMoE_expert(embed_dim=embed_dim,num_experts=num_experts,
                                       top_k=num_experts,dropout=dropout,shared=shared)
        elif args.sparse_dense=='mlp':
            self.moe = MLPLayer(embed_dim,dropout=dropout)
        else:
            raise Exception("Moe mode error")

        # self.token_gate = nn.Linear(num_experts*seq_length, seq_length)
        # self.token_gate_noise = nn.Linear(num_experts*seq_length, seq_length)
        # self.token_topk=seq_length


    def forward(self, x_swin,x_conv):

        # x_swin = x_swin + self.pos_embedding
        # x_conv = x_conv + self.pos_embedding
        x = self.sclayer(x_swin,x_conv)
        #concat
        # x_swin_ += x_swin
        # x_conv2swin += x_swin
        # x_conv_ += x_conv
        # x_swin2conv += x_conv
        # x= torch.cat([x_swin_,x_conv2swin,x_conv_,x_swin2conv],dim=1)
        # x,_ = x.topk(self.token_topk, dim=1)
        # # logits = self.token_gate(x.transpose(-1,-2))
        # # # Noise logits
        # # noise_logits = self.token_gate_noise(x.transpose(-1,-2))
        # # noise = torch.randn_like(logits) * F.softplus(noise_logits)
        # # noisy_logits = logits + noise

        # top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        #add

        x = self.moe(self.moe_norm(x)) + x
        return x

class Q_former(nn.Module):
    def __init__(self, embed_dim,drop_path=0.,num_heads=8, num_experts=4, top_k=1,dropout=0.,shared=False,args=None):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.cross_pre_norm_kv = nn.LayerNorm(embed_dim)
        self.cross_pre_norm_query = nn.LayerNorm(embed_dim)
        self.self_pre_norm = nn.LayerNorm(embed_dim)
        self.cross_att = CrossAttentionLayer(embed_dim=embed_dim,num_heads=num_heads)
        self.self_att = SelfAttentionLayer(embed_dim=embed_dim,num_heads=num_heads)
        self.moe_norm = nn.LayerNorm(embed_dim)
        if args.sparse_dense=='sparse_token':
            self.moe = SparseMoE_token(embed_dim=embed_dim,num_experts=num_experts,
                                       top_k=top_k,dropout=dropout,shared=shared)
        elif args.sparse_dense=='dense_token':
            self.moe = SparseMoE_token(embed_dim=embed_dim,num_experts=num_experts,
                                       top_k=num_experts,dropout=dropout,shared=shared)
        elif args.sparse_dense=='sparse_expert':
            self.moe = SparseMoE_expert(embed_dim=embed_dim,num_experts=num_experts,
                                       top_k=top_k,dropout=dropout,shared=shared)
        elif args.sparse_dense=='dense_expert':
            self.moe = SparseMoE_expert(embed_dim=embed_dim,num_experts=num_experts,
                                       top_k=num_experts,dropout=dropout,shared=shared)
        elif args.sparse_dense=='mlp':
            self.moe = MLPLayer(embed_dim,dropout=dropout)
        else:
            raise Exception("Moe mode error")

    def forward(self, q, k_v):

        q = self.drop_path(self.cross_att(self.cross_pre_norm_query(q),self.cross_pre_norm_kv(k_v))) + q

        q = self.drop_path(self.self_att(self.self_pre_norm(q)))+q

        q = self.drop_path(self.moe(self.moe_norm(q))) + q
        return q

class SC_noMoEBlock(nn.Module):#消融实验，将4个注意力直接相加，然后通过一个mlp
    def __init__(self, embed_dim, num_experts=4, top_k=1, dropout=0.):
        super().__init__()
        self.conv_pre_norm = nn.LayerNorm(embed_dim)
        self.swin_pre_norm = nn.LayerNorm(embed_dim)
        self.sclayer = SCAttentionLayer(embed_dim=embed_dim,dropout=0.)
        self.ffn = MLPLayer(embed_dim,dropout=dropout)

    def forward(self, x_swin,x_conv):
        x_swin_,x_conv2swin,x_conv_,x_swin2conv = self.sclayer(self.swin_pre_norm(x_swin),self.conv_pre_norm(x_conv))
        x = x_swin_+x_conv2swin+x_conv_+x_swin2conv+x_swin+x_conv
        x = self.ffn(x)
        return x


class MLPLayer(nn.Module):
    r""" MLP layer of InternImage
    Args:
        in_features (int): number of input features
        hidden_features (int): number of hidden features
        out_features (int): number of output features
        act_layer (str): activation layer
        drop (float): dropout rate
    """

    def __init__(self,
                 in_features,
                 hidden_features_radio=4,
                 out_features=None,
                 dropout=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features_radio *  in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()#build_act_layer(act_layer)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LivNet_Baseline(nn.Module):
    def __init__(self,num_classes=2,aux_loss=False,drop_path_rate=0.0,return_feature=False,
                 k_fold=None,**kwargs):
        super().__init__()
        self.aux_loss=aux_loss
        self.k_fold=k_fold
        self.num_classes = num_classes
        self.return_feature = return_feature
        self.swin = SwinTransformer(embed_dim=96, depth=[2,2,18,2],drop_path_rate=drop_path_rate,
                                    return_feature=False,num_heads=[3, 6, 12, 24], **kwargs)
        self.convnext = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], drop_path_rate=drop_path_rate,
                                 return_feature=False,**kwargs)


        self.load_pretrained()

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):

        return self.swin(x)+self.convnext(x)


    def load_pretrained(self):
        swin_weight=torch.load(r'/home/uax/SCY/LiverClassification/Weights/swin_small_patch4_window7_224_22k.pth',map_location="cpu")['model']
        convnext_weight = torch.load(r'/home/uax/SCY/LiverClassification/Weights/convnext_small_22k_224.pth',map_location="cpu")['model']

        swin_loaded_state_dict=self.swin.load_state_dict(swin_weight,strict=False)
        convnext_loaded_state_dict=self.convnext.load_state_dict(convnext_weight,strict=False)
        swin_missing_keys = swin_loaded_state_dict.missing_keys
        print("未加载的权重键：", swin_missing_keys)
        convnext_missing_keys = convnext_loaded_state_dict.missing_keys
        print("未加载的权重键：", convnext_missing_keys)
        self.swin.head = nn.Linear(in_features=self.swin.head.in_features, out_features=self.num_classes)
        self.convnext.head = nn.Linear(in_features=self.convnext.head.in_features,
                                             out_features=self.num_classes)

        del swin_weight,convnext_weight
class LivNet_no_moe(nn.Module):
    def __init__(self,num_classes=2,aux_loss=False,drop_path_rate=0.,return_feature=False,
                 dropout=0.1,k_fold=None,**kwargs):
        super().__init__()
        self.aux_loss=aux_loss
        self.k_fold=k_fold
        self.num_classes = num_classes
        self.return_feature = return_feature
        self.swintransformer = SwinTransformer(embed_dim=96, depths=[2,2,18,2],drop_path_rate=drop_path_rate,
        num_heads=[3, 6, 12, 24],out_features=True, **kwargs)
        self.convnext = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], drop_path_rate=drop_path_rate,
                                 **kwargs)
        dims = [96, 192, 384, 768]
        self.sc_att_mlp = nn.ModuleList()
        for i in range(len(dims)):
            layer = SC_noMoEBlock(embed_dim=dims[i],dropout=dropout)
            self.sc_att_mlp.append(layer)
            # layer_norm = nn.LayerNorm(dims[i])
            # layer_name = f'norm{i}'
            # self.add_module(layer_name, layer_norm)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.head_mlp = MLPLayer(sum(dims))
        self.head_norm = nn.LayerNorm(sum(dims))
        self.head = nn.Linear(sum(dims),num_classes)
        self.apply(self._init_weights)
        self.load_pretrained()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        swin_feature = self.swintransformer(x)
        convnext_feature = self.convnext(x)
        out=[]
        for i,layer in enumerate(self.sc_att_mlp):
            x = layer(swin_feature[i],convnext_feature[i])
            x = self.avgpool(x.transpose(1, 2)).flatten(1)
            out.append(x)
            # norm_layer = getattr(self, f'norm{i}')
            # out.append(norm_layer(x))
        x = torch.cat(out,dim=1)
        # x = x + self.head_mlp(x)
        x = self.head(self.head_norm(x))


        return x


    def load_pretrained(self):
        swin_weight=torch.load(r'/home/uax/SCY/LiverClassification/Weights/swin_small_patch4_window7_224_22k.pth',map_location="cpu")['model']
        convnext_weight = torch.load(r'/home/uax/SCY/LiverClassification/Weights/convnext_small_22k_224.pth',map_location="cpu")['model']

        swin_loaded_state_dict=self.swintransformer.load_state_dict(swin_weight,strict=False)
        convnext_loaded_state_dict=self.convnext.load_state_dict(convnext_weight,strict=False)
        swin_missing_keys = swin_loaded_state_dict.missing_keys
        print("未加载的权重键：", swin_missing_keys)
        convnext_missing_keys = convnext_loaded_state_dict.missing_keys
        print("未加载的权重键：", convnext_missing_keys)
        self.swintransformer.head = nn.Linear(in_features=self.swintransformer.head.in_features, out_features=self.num_classes)
        self.convnext.head = nn.Linear(in_features=self.convnext.head.in_features,out_features=self.num_classes)
        del self.swintransformer.norm,self.swintransformer.avgpool,self.swintransformer.head
        del self.convnext.norm, self.convnext.head
        del swin_weight,convnext_weight


class LENet(nn.Module):
    def __init__(self,num_classes=2,aux_loss=False,drop_path_rate=0.2,dropout=0.5,k_fold=None,
                 image_size=224,patch_size=4,args=None,**kwargs):
        super().__init__()
        self.cat_moe_head =args.cat_moe_head

        self.aux_loss=aux_loss
        self.k_fold=k_fold
        self.num_classes = num_classes
        self.return_feature = args.visual_feature
        self.swintransformer = SwinTransformer(embed_dim=96, depths=[2,2,18,2],drop_path_rate=drop_path_rate,
        num_heads=[3, 6, 12, 24], **kwargs)
        self.convnext = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], drop_path_rate=drop_path_rate,
                                 **kwargs)
        q_former_head_num = args.q_former_head_num
        self.q_former_depths = args.q_former_depths
        self.q_former_len=len(self.q_former_depths)
        dims = args.stage_dims
        query_dim=args.query_dim
        for _ in range(4 - len(dims)):
            image_size //= 2

        self.sc_att_mlp = nn.ModuleList()
        self.stage_proj = nn.ModuleList()
        self.q_cross_stage = nn.ModuleList()
        self.pos_embedding = nn.ModuleList()
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.q_former_depths))]  # stochastic depth decay rule
        cur = 0
        for i in range(len(dims)):
            seq_length = (image_size // patch_size) ** 2
            layer = SC_MoEBlock(embed_dim=dims[i],num_experts=4,top_k=1,seq_length=seq_length,
                                num_heads=q_former_head_num[i],dropout=dropout,shared=False,args=args)
            self.sc_att_mlp.append(layer)
            image_size //= 2
            layer = nn.Linear(dims[i],query_dim)
            self.stage_proj.append(layer)
            layer = nn.ModuleList(Q_former(embed_dim=query_dim,num_heads=q_former_head_num[i],args=args,
            drop_path=dpr[cur + j]) for j in range(self.q_former_depths[i]))
            self.q_cross_stage.append(layer)
            cur += self.q_former_depths[i]

            self.pos_embedding.append(nn.Embedding(seq_length, dims[i]))
            # layer_norm = nn.LayerNorm(dims[i])
            # layer_name = f'norm{i}'
            # self.add_module(layer_name, layer_norm)

        self.query_tokens = nn.Parameter(torch.zeros(1, args.num_query_tokens, query_dim))
        self.pos_embedding_query_tokens = nn.Parameter(torch.empty(1, args.num_query_tokens, query_dim).normal_(std=0.02))



        self.level_embed = nn.Embedding(len(dims), query_dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        if args.head_type=='moe_head':
            self.head = SparseMoE_token(query_dim,args.num_experts,out_features=num_classes)
        if args.head_type=='linear':
            self.head = nn.Linear(query_dim,num_classes)
        if args.head_type=='mlp':
            self.head = MLPLayer(query_dim,out_features=num_classes)
        self.head_norm = nn.LayerNorm(query_dim)

        self.apply(self._init_weights)
        self.load_pretrained()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x):
        swin_feature = self.swintransformer(x)
        convnext_feature = self.convnext(x)
        out=[]
        for i,layer in enumerate(self.sc_att_mlp):
            x = layer(swin_feature[4-self.q_former_len+i]+self.pos_embedding[i].weight.unsqueeze(0),
                      convnext_feature[4-self.q_former_len+i]+self.pos_embedding[i].weight.unsqueeze(0))
            x = self.stage_proj[i](x)
            x = x + self.level_embed.weight[i][None, None,: ]#+self.pos_embedding_kv[i].weight.unsqueeze(0)
            # x = self.avgpool(x.transpose(1, 2)).flatten(1)
            out.append(x)
            # norm_layer = getattr(self, f'norm{i}')
            # out.append(norm_layer(x))
        x = self.query_tokens.expand(x.shape[0], -1, -1)+self.pos_embedding_query_tokens

        if self.cat_moe_head:
            for i, layer in enumerate(self.q_cross_stage):
                x = layer(x, out[i])
                out[i] = x
            x = torch.cat(out, dim=1)
        else:
            for i, layers in enumerate(self.q_cross_stage):
                for j , layer in enumerate(layers):
                    x = layer(x, out[i])


        x = self.head(self.head_norm(x))
        x = self.avgpool(x.transpose(1, 2)).flatten(1)

        if self.return_feature:
            return x,out
        return x


    def load_pretrained(self):
        swin_weight=torch.load(r'/home/uax/SCY/LiverClassification/Weights/swin_small_patch4_window7_224_22k.pth',map_location="cpu")['model']
        convnext_weight = torch.load(r'/home/uax/SCY/LiverClassification/Weights/convnext_small_22k_224.pth',map_location="cpu")['model']

        swin_loaded_state_dict=self.swintransformer.load_state_dict(swin_weight,strict=False)
        convnext_loaded_state_dict=self.convnext.load_state_dict(convnext_weight,strict=False)
        swin_missing_keys = swin_loaded_state_dict.missing_keys
        print("未加载的权重键：", swin_missing_keys)
        convnext_missing_keys = convnext_loaded_state_dict.missing_keys
        print("未加载的权重键：", convnext_missing_keys)
        self.swintransformer.head = nn.Linear(in_features=self.swintransformer.head.in_features, out_features=self.num_classes)
        self.convnext.head = nn.Linear(in_features=self.convnext.head.in_features,out_features=self.num_classes)
        del self.swintransformer.norm,self.swintransformer.avgpool,self.swintransformer.head
        del self.convnext.norm, self.convnext.head
        del swin_weight,convnext_weight