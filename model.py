import torch 
import torch.distributed.nn 
import torch.nn.functional as F
import torch.distributed as dist 
from torch import nn
from collections import OrderedDict

from modules.position_embedding import SinusoidalPositionalEmbedding


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=0.2, batch_first=True)
        self.ln_1 = LayerNorm(d_model)
        self.ln_12 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("relu", nn.ReLU()),
            ('dropout', nn.Dropout(p=0.1)),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.ln_22 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.dropout = nn.Dropout(p=0.1)

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.ln_12(self.attention(self.ln_1(x)))
        x = x + self.ln_22(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class EmoClassAttention(nn.Module):
    def __init__(
        self,
        M=1024,
        H=1024,
        C=7,
        widths=[1024]*3,
        heads=[16]*3,
        args=None,
    ):
        super(EmoClassAttention, self).__init__()
        self.M = M
        self.H = H
        self.C = C
        self.args = args

        self.net = nn.Sequential(
            nn.Linear(H, M),
            nn.Tanh()
        )
        self.W1 = nn.Linear(M, C)
        self.W2 = nn.Linear(C, C)

        self.ln = nn.LayerNorm(H)
        classifier_weights = torch.empty(C, H)
        nn.init.xavier_uniform_(classifier_weights, gain=1.)
        self.classifier = nn.Parameter(classifier_weights)

        self.mlp_a = nn.Sequential(nn.Linear(widths[0], int(widths[0] / 2)), nn.ReLU(), nn.Linear(int(widths[0] / 2), 32), nn.ReLU(), nn.Linear(32, C))
        self.mlp_t = nn.Sequential(nn.Linear(widths[1], int(widths[1] / 2)), nn.ReLU(), nn.Linear(int(widths[1] / 2), 32), nn.ReLU(), nn.Linear(32, C))
        self.mlp_v = nn.Sequential(nn.Linear(widths[2], int(widths[2] / 2)), nn.ReLU(), nn.Linear(int(widths[2] / 2), 32), nn.ReLU(), nn.Linear(32, C))

        self.mix_pool_m = nn.Parameter((C ** -0.5) * torch.rand(C))
        self.mlp_m = nn.Sequential(nn.Linear(H, 32), nn.ReLU(), nn.Linear(32, C))
        return

    def forward(self, x_a, x_t, x_v):
        args = self.args
        logits_a = self.mlp_a(x_a)
        logits_t = self.mlp_t(x_t)
        logits_v = self.mlp_v(x_v)
        x = torch.stack([x_a, x_t, x_v], dim=1)
        # x_a: (B, H)
        # logits_a: (B, C)
        # x: (B, N, H)

        part = self.net(x)
        scores = self.W1(part).transpose(2, 1)
        A = F.softmax(scores, dim=2)
        # part3: (B, N, M)
        # scores: (B, C, N)
        # A: (B, C, N)
        
        x_m = torch.matmul(A, x)
        # x_m: (B, C, H)
        x_m = torch.matmul(self.mix_pool_m, x_m)
        # x_m: (B, H)

        logits_m = self.mlp_m(x_m)
        logits_m = self.W2(logits_m)
        # x_m: (B, (1+N)*H)
        # logits_m: (B, C)

        outputs = {
            'atten': A,
            'logits_a': logits_a,
            'logits_t': logits_t,
            'logits_v': logits_v,
            'logits_m': logits_m,
        }
        return outputs


class EmoEncoder(nn.Module):
    def __init__(
        self,
        widths_encoder,
        widths=[1024]*3,
        dropout_rates=[0.25]*3,
        heads=[16]*3,
        num_layer=4,
        attn_mask=None,
        args=None,
    ):
        super(EmoEncoder, self).__init__()
        self.widths = widths
        self.dropout_rates = dropout_rates
        self.heads = heads
        self.num_layer = num_layer
        self.args = args

        self.fea_len_a = args.fea_len_a
        self.fea_len_t = args.fea_len_t
        self.fea_len_v = args.fea_len_v
        self.pool_method = args.pool_method

        self.fc_in_a = nn.Linear(widths_encoder[0], widths[0])
        self.fc_in_t = nn.Linear(widths_encoder[1], widths[1])
        self.fc_in_v = nn.Linear(widths_encoder[2], widths[2])

        self.emb_scale_a = widths[0] ** -0.5
        self.emb_scale_t = widths[1] ** -0.5
        self.emb_scale_v = widths[2] ** -0.5
        self.emb_a = nn.Parameter(self.emb_scale_a * torch.randn(widths[0]))
        self.emb_t = nn.Parameter(self.emb_scale_t * torch.randn(widths[1]))
        self.emb_v = nn.Parameter(self.emb_scale_v * torch.randn(widths[2]))

        self.ln_before_pe_a = nn.LayerNorm(widths[0])
        self.ln_before_pe_t = nn.LayerNorm(widths[1])
        self.ln_before_pe_v = nn.LayerNorm(widths[2])

        self.emb_pe_a = SinusoidalPositionalEmbedding(widths[0])
        self.emb_pe_t = SinusoidalPositionalEmbedding(widths[1])
        self.emb_pe_v = SinusoidalPositionalEmbedding(widths[2])

        self.ln_after_pe_a = nn.LayerNorm(widths[0])
        self.ln_after_pe_t = nn.LayerNorm(widths[1])
        self.ln_after_pe_v = nn.LayerNorm(widths[2])

        self.transformer_a = nn.Sequential(*[ResidualAttentionBlock(widths[0], heads[0], attn_mask) for _ in range(num_layer)])
        self.transformer_t = nn.Sequential(*[ResidualAttentionBlock(widths[1], heads[1], attn_mask) for _ in range(num_layer)])
        self.transformer_v = nn.Sequential(*[ResidualAttentionBlock(widths[2], heads[2], attn_mask) for _ in range(2*num_layer)])
        self.transformers = [self.transformer_a, self.transformer_t, self.transformer_v]

        self.avg_pool_a = nn.AvgPool1d(self.fea_len_a)
        self.avg_pool_t = nn.AvgPool1d(self.fea_len_t)
        self.avg_pool_v = nn.AvgPool1d(self.fea_len_v)
        self.max_pool_a = nn.MaxPool1d(self.fea_len_a)
        self.max_pool_t = nn.MaxPool1d(self.fea_len_t)
        self.max_pool_v = nn.MaxPool1d(self.fea_len_v)
        self.mix_pool_a = nn.Parameter((self.fea_len_a ** -0.5) * torch.rand(self.fea_len_a))
        self.mix_pool_t = nn.Parameter((self.fea_len_t ** -0.5) * torch.rand(self.fea_len_t))
        self.mix_pool_v = nn.Parameter((self.fea_len_v ** -0.5) * torch.rand(self.fea_len_v))

        self.initialize_parameters()
        return

    def forward(self, x_a, x_t, x_v):
        device = x_a.device
        dtype = x_a.dtype
        x_t = x_t[:, 0:80, :]
        x_a = x_a.to(torch.float32)
        x_t = x_t.to(torch.float32)
        x_v = x_v.to(torch.float32)
        x_a = self.fc_in_a(x_a)
        x_t = self.fc_in_t(x_t)
        x_v = self.fc_in_v(x_v)

        x_a = torch.cat([self.emb_a.to(dtype) + torch.zeros(x_a.shape[0], 1, x_a.shape[-1], dtype=dtype, device=device), x_a], dim=1)
        x_t = torch.cat([self.emb_t.to(dtype) + torch.zeros(x_t.shape[0], 1, x_t.shape[-1], dtype=dtype, device=device), x_t], dim=1)
        x_v = torch.cat([self.emb_v.to(dtype) + torch.zeros(x_v.shape[0], 1, x_v.shape[-1], dtype=dtype, device=device), x_v], dim=1)

        x_a = 1 / self.emb_scale_a * x_a
        x_t = 1 / self.emb_scale_t * x_t
        x_v = 1 / self.emb_scale_v * x_v

        x_a = self.ln_before_pe_a(x_a)
        x_t = self.ln_before_pe_a(x_t)
        x_v = self.ln_before_pe_a(x_v)

        x_a += self.emb_pe_a(x_a[:, :, 0])
        x_t += self.emb_pe_t(x_t[:, :, 0])
        x_v += self.emb_pe_v(x_v[:, :, 0])

        x_a = F.dropout(x_a, p=self.dropout_rates[0], training=self.training)
        x_t = F.dropout(x_t, p=self.dropout_rates[1], training=self.training)
        x_v = F.dropout(x_v, p=self.dropout_rates[2], training=self.training)

        x_a = self.ln_after_pe_a(x_a)
        x_t = self.ln_after_pe_t(x_t)
        x_v = self.ln_after_pe_v(x_v)

        for i_layer in range(self.num_layer):
            x_a = self.transformer_a[i_layer](x_a)
            x_t = self.transformer_t[i_layer](x_t)
            x_v = self.transformer_v[2*i_layer](x_v)
            x_v = self.transformer_v[2*i_layer+1](x_v)
        # x_a: (B, T, H)

        if self.pool_method == 'avg':
            x_a = self.avg_pool_a(x_a.transpose(2, 1)).squeeze(2)
            x_t = self.avg_pool_t(x_t.transpose(2, 1)).squeeze(2)
            x_v = self.avg_pool_v(x_v.transpose(2, 1)).squeeze(2)
        elif self.pool_method == 'max':
            x_a = self.max_pool_a(x_a.transpose(2, 1)).squeeze(2)
            x_t = self.max_pool_t(x_t.transpose(2, 1)).squeeze(2)
            x_v = self.max_pool_v(x_v.transpose(2, 1)).squeeze(2)
        elif self.pool_method == 'mix':
            x_a = torch.matmul(self.mix_pool_a, x_a)
            x_t = torch.matmul(self.mix_pool_t, x_t)
            x_v = torch.matmul(self.mix_pool_v, x_v)
        # x_a: (B, H)

        outputs = {
            'audio': x_a,
            'text': x_t,
            'video': x_v,
        }
        return outputs

    def initialize_parameters(self):
        num_layer = self.num_layer
        widths = self.widths
        for i in [0, 1]:
            proj_std = (widths[i] ** -0.5) * ((2 * num_layer) ** -0.5)
            attn_std = widths[i] ** -0.5
            fc_std = (2 * widths[i]) ** -0.5
            transformer = self.transformers[i]
            for block in transformer:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        for i in [2]:
            proj_std = (widths[i] ** -0.5) * ((2 * 2 * num_layer) ** -0.5)
            attn_std = widths[i] ** -0.5
            fc_std = (2 * widths[i]) ** -0.5
            transformer = self.transformers[i]
            for block in transformer:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        return


class EmoModel(nn.Module):
    def __init__(
        self,
        widths_encoder=[1024, 1024, 2048],
        widths=[1024]*3,
        dropout_rates=[0.25]*3,
        heads=[16]*3,
        num_layer=4,
        attn_mask=None,
        M=1024,
        H=1024,
        C=7,
        args=None,
    ):
        super(EmoModel, self).__init__()
        self.widths = widths
        self.dropout_rates = dropout_rates
        self.heads = heads
        self.num_layer = num_layer
        self.args = args

        self.fea_len_a = args.fea_len_a
        self.fea_len_t = args.fea_len_t
        self.fea_len_v = args.fea_len_v

        self.encoder = EmoEncoder(
            widths_encoder=widths_encoder,
            widths=widths,
            dropout_rates=dropout_rates,
            heads=heads,
            num_layer=num_layer,
            attn_mask=attn_mask,
            args=args,
        )
        self.class_attention = EmoClassAttention(M, H, C, args=args)

        self.last_local_batch_size = 0
        return 

    def evaluate(self, x_t, x_a, x_v):
        args = self.args
        # encode
        outputs_encoder_q = self.encoder(x_a, x_t, x_v)
        x_a_q = outputs_encoder_q['audio']
        x_t_q = outputs_encoder_q['text']
        x_v_q = outputs_encoder_q['video']
        # x_a_q: (B, H)

        # atten
        outputs_class_attention = self.class_attention(x_a_q, x_t_q, x_v_q)
        A = outputs_class_attention['atten']
        B, C, N = A.shape

        # modal
        logits_a = outputs_class_attention['logits_a']
        logits_t = outputs_class_attention['logits_t']
        logits_v = outputs_class_attention['logits_v']
        logits_m = outputs_class_attention['logits_m']

        probs_a = F.softmax(logits_a, dim=1)
        probs_t = F.softmax(logits_t, dim=1)
        probs_v = F.softmax(logits_v, dim=1)
        probs_m = F.softmax(logits_m, dim=1)

        pred_a = probs_a.argmax(dim=1)
        pred_t = probs_t.argmax(dim=1)
        pred_v = probs_v.argmax(dim=1)
        pred_m = probs_m.argmax(dim=1)
        return pred_a, pred_t, pred_v, pred_m
    
    def forward(self, x_t, x_a, x_v, label_t, label_a, label_v, label_m):
        args = self.args
        # encode
        outputs_encoder_q = self.encoder(x_a, x_t, x_v)
        x_a_q = outputs_encoder_q['audio']
        x_t_q = outputs_encoder_q['text']
        x_v_q = outputs_encoder_q['video']
        # x_a_q: (B, H)

        # atten
        outputs_class_attention = self.class_attention(x_a_q, x_t_q, x_v_q)
        A = outputs_class_attention['atten']
        B = A.shape[0]
        device = A.device
        # modal
        logits_a = outputs_class_attention['logits_a']
        logits_t = outputs_class_attention['logits_t']
        logits_v = outputs_class_attention['logits_v']
        logits_m = outputs_class_attention['logits_m']
        loss_a = F.cross_entropy(logits_a, label_a)
        loss_t = F.cross_entropy(logits_t, label_t)
        loss_v = F.cross_entropy(logits_v, label_v)
        loss_m = F.cross_entropy(logits_m, label_m)
        # logits_a: (B, C)
        
        A_tgt = A[range(B), label_m]
        label_s = torch.stack([label_a, label_t, label_v], dim=1)
        label_atten = (label_s == label_m.unsqueeze(1)).to(torch.float32)
        label_atten_sum = label_atten.sum(dim=1)
        flag_need_combine = (label_atten_sum == 0)
        label_atten[flag_need_combine] = 1
        A_tgt_right = (A_tgt * label_atten).sum(dim=1)
        A_tgt_right = torch.stack([A_tgt_right, 1 - A_tgt_right], dim=1)
        label_atten_tgt_right = torch.zeros(B).to(torch.int64).to(device)
        loss_atten_tgt = F.cross_entropy(A_tgt_right, label_atten_tgt_right)
        # A_tgt: (B, N)
        # label_s: (B, N)
        # label_atten: (B, N)
        # label_atten_sum: (B,)
        # flag_need_combine: (B,)
        # A_tgt_right: (B,) → (B, 2)
        # label_atten_tgt_right: (B,)

        # cl
        all_loss_cl = []
        for i, x_q, label, loss_s in zip(range(3), [x_a_q, x_t_q, x_v_q], [label_a, label_t, label_v], [loss_a, loss_t, loss_v]):
            flag_same = label == label_m 
            x_q.retain_grad()
            loss_s.backward(retain_graph=True)
            unnormalized_noise = x_q.grad.detach_()
            norm = unnormalized_noise.norm(p=2, dim=-1)
            normalized_noise = unnormalized_noise / (norm.unsqueeze(dim=-1) + 1e-8)
            x_k = x_q - 0.01 * normalized_noise
            mix_feat = torch.cat([x_q, x_k], dim=0)
            mix_feat = F.normalize(mix_feat, dim=-1, p=2, eps=1e-8)
            mix_label = torch.cat([label, label], dim=0)
            mix_flag_same = torch.cat([flag_same, flag_same], dim=0)
            # mix_feat: (2*B, H)
            # mix_label: (2*B,)
            # mix_flag_same: (2*B,)
            
            all_feat = torch.cat(torch.distributed.nn.all_gather(mix_feat), dim=0)
            all_label = concat_all_gather(mix_label)
            all_flag_same = concat_all_gather(mix_flag_same)
            # all_feat: (G*2*B, H)
            # all_label: (G*2*B,)
            # all_flag_same: (G*2*B,)

            flag_keep = torch.bitwise_and(mix_flag_same.reshape(-1, 1), all_flag_same)
            # flag_keep = torch.bitwise_and(mix_flag_same.reshape(-1, 1), torch.ones_like(all_flag_same))
            S = mix_flag_same.sum()

            mix_label_cl = torch.eq(mix_label.view(-1, 1), all_label.contiguous().view(1, -1).float().to(device))
            mask_self = torch.scatter(
                torch.ones_like(mix_label_cl),
                1,
                torch.arange(2*B).view(-1, 1).to(device) + 2 * B * get_rank(),
                0
            ).float()
            mix_label_cl = mix_label_cl * mask_self
            # mix_label_cl: (S, G*B)
            # mask_self: (S, G*B)

            mix_label_cl[:, 2*B*get_rank():2*B*(1+get_rank())] = 0
            cumsum = mix_label_cl.cumsum(dim=1)
            mix_label_cl[cumsum > 10] = 0

            mix_logits = (mix_feat @ all_feat.T) / args.temperature
            mix_logits = mix_logits - (1 - mask_self) * 1e8
            mix_prob = mix_label_cl / mix_label_cl.sum(1, keepdim=True).clamp(min=1.0)
            mix_logits = torch.masked_select(mix_logits, flag_keep).reshape(S, -1)
            mix_prob = torch.masked_select(mix_prob, flag_keep).reshape(S, -1)
            loss_cl = compute_cross_entropy(mix_prob, mix_logits)
            all_loss_cl.append(loss_cl)

        loss = 0.2 * loss_atten_tgt + loss_m + 0.33 * sum(all_loss_cl)
        return loss


def compute_cross_entropy(p, q):
    q = F.log_softmax(q, dim=-1)
    loss = torch.sum(p * q, dim=-1)
    return - loss.mean()

@torch.no_grad()
def concat_all_gather(ten):
    # no grad
    ten_gather = [torch.ones_like(ten) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(ten_gather, ten, async_op=False)
    output = torch.cat(ten_gather, dim=0)
    return output 

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0 
    return dist.get_rank()

def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()
