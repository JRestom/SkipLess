import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def linear_scheduler(step, total_steps, start, end):
    return max(end, start - (start - end) * (step / total_steps))

def cosine_scheduler(step, total_steps, start, end):
    return end + (start - end) * 0.5 * (1 + math.cos(math.pi * step / (total_steps - 1)))

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feats: int, head: int = 8, dropout: float = 0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.head = head
        self.feats = feats
        self.sqrt_d = feats ** 0.5
        
        self.q = nn.Linear(feats, feats)
        self.k = nn.Linear(feats, feats)
        self.v = nn.Linear(feats, feats)
        self.o = nn.Linear(feats, feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, f = x.size()
        q = self.q(x).view(b, n, self.head, self.feats // self.head).transpose(1, 2)
        k = self.k(x).view(b, n, self.head, self.feats // self.head).transpose(1, 2)
        v = self.v(x).view(b, n, self.head, self.feats // self.head).transpose(1, 2)

        score = F.softmax(torch.einsum("bhif, bhjf -> bhij", q, k) / self.sqrt_d, dim=-1)
        attn = torch.einsum("bhij, bhjf -> bihf", score, v)
        o = self.dropout(self.o(attn.flatten(2)))
        return o


class TransformerEncoder(nn.Module):
    def __init__(self, feats: int, mlp_hidden: int, head: int = 8, dropout: float = 0.1, skip_scalar:float=1.0):
        super(TransformerEncoder, self).__init__()
        self.la1 = nn.LayerNorm(feats)
        self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout)
        self.la2 = nn.LayerNorm(feats)
        self.mlp = nn.Sequential(
            nn.Linear(feats, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, feats),
            nn.GELU(), #chatgpt recommended to remove to match og vit
            nn.Dropout(dropout),
        )
        self.skip_scalar = skip_scalar
        print(f"TransformerEncoder initialized with skip_scalar: {self.skip_scalar}")
    
    def forward(self, x):
        #print(self.skip_scalar)
        out = self.msa(self.la1(x)) + self.skip_scalar * x
        out = self.mlp(self.la2(out)) + self.skip_scalar * out
        return out


class ViT(nn.Module):
    def __init__(self, in_c: int = 3, num_classes: int = 10, img_size: int = 32, patch: int = 8, dropout: float = 0.1, num_layers: int = 7, hidden: int = 384, mlp_hidden: int = 384 * 4, head: int = 8, is_cls_token: bool = True, skip_scalar:float=1.0, scheduler_type: str = 'linear', total_epochs: int = 100, final_skip: float = 1.0, start_value = 1.0):
        super(ViT, self).__init__()
        self.patch = patch
        self.is_cls_token = is_cls_token
        self.patch_size = img_size // self.patch
        f = (self.patch_size ** 2) * in_c
        num_tokens = (self.patch ** 2) + 1 if self.is_cls_token else (self.patch ** 2)

        self.skip_scalar = skip_scalar  
        self.scheduler_type = scheduler_type
        self.total_epochs = total_epochs
        self.final_skip = final_skip
        self.start_value = start_value

        print(f"Initializing ViT with skip_scalar: {self.skip_scalar}")

        self.emb = nn.Linear(f, hidden)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden)) if is_cls_token else None
        self.pos_emb = nn.Parameter(torch.randn(1, num_tokens, hidden))
        self.enc = nn.Sequential(*[TransformerEncoder(hidden, mlp_hidden=mlp_hidden, dropout=dropout, head=head, skip_scalar=skip_scalar) for _ in range(num_layers)])
        self.fc = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, num_classes))
    
    def forward(self, x):
        out = self._to_words(x)
        out = self.emb(out)
        if self.is_cls_token:
            out = torch.cat([self.cls_token.repeat(out.size(0), 1, 1), out], dim=1)
        out = out + self.pos_emb
        out = self.enc(out)
        out = out[:, 0] if self.is_cls_token else out.mean(1)
        out = self.fc(out)
        return out
    
    def _to_words(self, x):
        out = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size).permute(0, 2, 3, 4, 5, 1)
        out = out.reshape(x.size(0), self.patch ** 2, -1)
        return out

    def update_skip_scale(self, step, total_steps, start_value=None):
        if start_value is None:
            start_value = self.start_value

        if step >= total_steps:
            return   

        if self.scheduler_type == 'linear':
            self.skip_scalar = linear_scheduler(step, total_steps, start=start_value, end=self.final_skip)
        elif self.scheduler_type == 'cosine':
            self.skip_scalar = cosine_scheduler(step, total_steps, start=start_value, end=self.final_skip)
        elif self.scheduler_type == 'none':
            self.skip_scalar = 1.0
        else:
            raise ValueError(f"Unsupported scheduler type: {self.scheduler_type}")

        print(f"Updated skip_scalar: {self.skip_scalar}")

        for layer in self.enc:
            layer.skip_scalar = self.skip_scalar

    def get_skip_scales(self):
        return {"skip_scalar": self.skip_scalar}


def vit_base(num_classes=10, img_size=32, patch=8, hidden=384, num_layers=7, head=8, dropout=0.1, is_cls_token=True, skip_scalar=1.0, scheduler_type='linear', total_epochs=100, final_skip=1.0, start_value=1.0):
    return ViT(
        in_c=3,
        num_classes=num_classes,
        img_size=img_size,
        patch=patch,
        hidden=hidden,
        num_layers=num_layers,
        head=head,
        dropout=dropout,
        is_cls_token=is_cls_token,
        skip_scalar=skip_scalar,
        scheduler_type=scheduler_type,
        total_epochs=total_epochs,
        final_skip=final_skip,
        start_value=start_value
    )
