"""Model v1 (spec/model.md §2) and its losses: SIM pre-training, Phase 8.

One window of preprocessed CSI from every link of a scene goes in (the
batch of spec/model.md §1); presence, people count, the 3 x 3 zone grid and
up to three people's positions with an uncertainty come out.

1. **Per-link encoder.** `amp` (44 z-score channels at the reference bins
   `R`), `rssi` minus its per-link window mean (dB, no division) and
   `sample_mask`, over time: three convolution layers (kernel 3, width `D`,
   GELU; `TimeConv`, a Conv1d written as one GEMM), masked mean and max
   pooling over time, projected to `D`. The side
   features `mu`, `log(sigma + 1e-4)` (2 x 44), the per-link RSSI mean and
   `log(std + 0.1)` go through an MLP and are added.
2. **Geometry.** tx, rx (normalised by the room), their midpoint, the
   segment's length and direction (`cos`, `sin`) -> MLP, added to the link
   token; the room (`w, d, h`, m) -> MLP, a global token. Where
   `geometry_known` is false both are learned "unknown" tokens.
3. **Across links.** Two pre-norm transformer encoder layers over the room
   token and the link tokens (`D`, 4 heads, padding masked), then
   attention pooling with four learned queries: one pooled vector (presence,
   count, zone) and three slots (xy).
4. **Heads.** Presence 1 logit, count 4, zone 9; per slot an existence
   logit, a centre (sigmoid, room-normalised) and
   `sigma = 0.1225 + softplus(raw)` m, at most 7.4 m.

Choices the spec leaves open (flagged in the Phase 8 report):
- hidden activations are re-zeroed at masked time bins after every conv
  layer, so a masked bin acts exactly like the convolution's zero padding
  and appending masked bins or padded links changes nothing (tested);
- the side features and the room enter the MLPs through fixed affine
  scalings measured on SIM (roomsim-v1, 5 Hz bins: `mu` 1.00 +- 0.27,
  `log(sigma + 1e-4)` -3.87 +- 0.58, RSSI mean -44.5 +- 7.0 dBm,
  `log(std + 0.1)` -0.50 +- 0.28); they are part of the model config;
- the RSSI spread is the unbiased standard deviation over the link's bins
  with an RSSI (0 below two bins), as spec/preprocess.md §6 uses for
  `sigma`; the link length and direction are of the 2-D segment in
  room-normalised coordinates;
- the 7.4 m cap on `sigma` is a straight-through clamp: the forward value is
  `min(sigma, 7.4)`, the gradient that of the uncapped value (a hard clamp
  would freeze a slot whose `sigma` reached the cap);
- `2.448 * 0.1225 = 0.29988 m`, not quite 0.3 m: the reported 95% radius
  is `max(2.448 sigma, 0.3)` (spec/model.md decision 3's floor,
  `radius95`), which is what guarantees 0.3 m.

Matching (the loss's Hungarian assignment of slots to present people on
the Euclidean distance in metres) has two exact solvers: `scipy`
(`scipy.optimize.linear_sum_assignment` per sample, on the CPU) and
`enumerate` (every assignment of at most 3 people to 3 slots, 6 of them,
scored at once on the device; no host round trip). Both return a
minimum-cost assignment; the tests hold them equal.

Everything this model is trained and evaluated on is SIM.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import asdict, dataclass, fields

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

N_REF = 44                    # the reference subcarriers R (spec/preprocess.md §3) that reach the model
N_SLOTS = 3
N_ZONES = 9
N_COUNT = 4                   # people count 0..3
SIGMA_MIN_M = 0.1225          # spec/model.md §2.4
SIGMA_MAX_M = 7.4
R95 = 2.448                   # 95% radius of an isotropic 2-D Gaussian in sigmas: sqrt(-2 ln 0.05) = 2.4477
MIN_PLAUSIBLE_RADIUS_M = 0.3  # spec/locate.md §9: the reported radius is floored here (spec/model.md decision 3)
LOG_2PI = math.log(2.0 * math.pi)
MASKED_FILL = -1.0e4          # max-pooling fill at masked bins (finite in float16)

# Loss weights (spec/model.md §2 Losses): equal except the xy NLL.
DEFAULT_LOSS_WEIGHTS = {"presence": 1.0, "count": 1.0, "zone": 1.0, "xy_nll": 0.5, "exist": 1.0}

# Keys of the batch the model reads (spec/model.md §1); the ablations add `phase`/`phase_mask` or `gain_db`.
MODEL_KEYS = ("amp", "mu", "sigma", "sample_mask", "link_mask", "rssi", "rssi_mask", "link_pos", "room",
              "geometry_known")
LOSS_KEYS = ("presence", "count", "zone_occupied", "zone_reachable", "xy", "person_mask", "geometry_known", "room")
PER_LINK_KEYS = ("amp", "mu", "sigma", "sample_mask", "link_mask", "rssi", "rssi_mask", "link_pos", "phase",
                 "phase_mask", "gain_db", "links")


@dataclass
class ModelConfig:
    """Everything that fixes the architecture; saved with every checkpoint."""

    d: int = 128                  # width D
    heads: int = 4
    layers: int = 2
    ff_mult: int = 4              # transformer feed-forward width = ff_mult * D
    dropout: float = 0.1
    phase: bool = False           # ablation §5.4: 2 x 44 phase channels (cos, sin at R)
    gain: bool = False            # ablation §5.5: gain_db minus its per-link window mean
    n_ref: int = N_REF
    sigma_min: float = SIGMA_MIN_M
    sigma_max: float = SIGMA_MAX_M
    # fixed input scalings (module docstring): feature -> (feature - centre) / scale
    mu_centre: float = 1.0
    mu_scale: float = 0.25
    logsig_centre: float = -3.9
    logsig_scale: float = 0.6
    rssi_centre: float = -45.0
    rssi_scale: float = 7.0
    logstd_centre: float = -0.5
    logstd_scale: float = 0.3
    room_scale: float = 5.0

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown model config keys {sorted(unknown)}")
        return cls(**d)

    @property
    def in_channels(self) -> int:
        return self.n_ref + 2 + (2 * self.n_ref if self.phase else 0) + (1 if self.gain else 0)


def _mlp(n_in: int, d: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_in, d), nn.GELU(), nn.Linear(d, d))


class TimeConv(nn.Module):
    """`Conv1d(c_in, d, kernel_size=3, padding=1)` on time-major input
    [N, T, C]: the three shifted copies side by side, one Linear(3 C, d).
    The same function and parameter count as the Conv1d (its weight
    `[d, C, 3]` is this one's `[d, 3 C]` regrouped by tap), as one GEMM
    without cuDNN's channel-layout conversions (measured: they cost a fifth
    of the step at D = 256)."""

    def __init__(self, c_in: int, d: int):
        super().__init__()
        self.lin = nn.Linear(3 * c_in, d)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        hp = F.pad(h, (0, 0, 1, 1))                                          # zero padding in time
        return self.lin(torch.cat([hp[:, :-2], hp[:, 1:-1], hp[:, 2:]], dim=-1))


class LinkEncoder(nn.Module):
    """§2.1: a kernel-3 convolution x 3 over time (width D, GELU), masked
    mean and max pooling, projected to D. Hidden activations are zeroed at
    masked bins after every layer (module docstring)."""

    def __init__(self, c_in: int, d: int):
        super().__init__()
        self.convs = nn.ModuleList([TimeConv(c_in, d), TimeConv(d, d), TimeConv(d, d)])
        self.proj = nn.Linear(2 * d, d)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        # x [N, T, C] (time-major), valid [N, T] bool
        v = valid[..., None]
        h = x * v.to(x.dtype)
        for conv in self.convs:
            h = F.gelu(conv(h))
            h = h * v.to(h.dtype)
        cnt = valid.sum(-1, keepdim=True)                                   # [N, 1]
        mean = h.sum(1) / cnt.clamp(min=1).to(h.dtype)
        # max(dim) routes the gradient by the argmax index; amax's equality-based backward came out NaN
        # under torch.compile with float16 (measured). The fill is finite (GELU >= -0.17, so it never wins
        # on a link with a valid bin); a link without one (padding) gets 0.
        mx = h.masked_fill(~v, MASKED_FILL).max(dim=1).values
        mx = torch.where(cnt > 0, mx, torch.zeros_like(mx))
        return self.proj(torch.cat([mean, mx], dim=-1))


class Block(nn.Module):
    """A pre-norm transformer encoder layer (self-attention + GELU MLP) with
    a key padding mask; scaled_dot_product_attention underneath."""

    def __init__(self, d: int, heads: int, ff: int, dropout: float):
        super().__init__()
        if d % heads:
            raise ValueError(f"width {d} is not a multiple of {heads} heads")
        self.heads = heads
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, ff)
        self.ff2 = nn.Linear(ff, d)
        self.drop = nn.Dropout(dropout)
        self.p = dropout

    def forward(self, x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, s, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=keep[:, None, None, :],
                                           dropout_p=self.p if self.training else 0.0)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(b, s, d)))
        return x + self.drop(self.ff2(self.drop(F.gelu(self.ff1(self.n2(x))))))


class QueryPool(nn.Module):
    """§2.3: attention pooling with learned queries (1 pooled + 3 slots)."""

    def __init__(self, d: int, heads: int, n_queries: int):
        super().__init__()
        self.heads = heads
        self.queries = nn.Parameter(torch.randn(n_queries, d))
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.proj = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, tokens: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        b, s, d = tokens.shape
        nq = self.queries.shape[0]
        hd = d // self.heads
        q = self.q(self.queries).view(1, nq, self.heads, hd).transpose(1, 2).expand(b, -1, -1, -1)
        k, v = self.kv(tokens).view(b, s, 2, self.heads, hd).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q.to(k.dtype), k, v, attn_mask=keep[:, None, None, :])
        out = self.proj(a.transpose(1, 2).reshape(b, nq, d))
        return self.norm(out + self.queries.to(out.dtype))


class SaansModel(nn.Module):
    """spec/model.md §2. `forward(batch)` takes the §1 batch (tensors on the
    model's device; float16 inputs are read as float32) and returns float32
    tensors: `presence_logit` [B], `count_logits` [B, 4], `zone_logits`
    [B, 9], `slot_logit` [B, 3], `slot_centre` [B, 3, 2] (room-normalised),
    `slot_xy` [B, 3, 2] (m, room frame: centre * (w, d)), `slot_sigma`
    [B, 3] (m). An optional `pack_index` (`pack_index()`) makes the link
    encoder run on the batch's real links only (same outputs, less work)."""

    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or ModelConfig()
        d = cfg.d
        self.encoder = LinkEncoder(cfg.in_channels, d)
        self.side = _mlp(2 * cfg.n_ref + 2, d)
        self.geo = _mlp(12, d)
        self.room = _mlp(3, d)
        self.unknown_geo = nn.Parameter(torch.randn(d) * 0.02)
        self.unknown_room = nn.Parameter(torch.randn(d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, cfg.heads, cfg.ff_mult * d, cfg.dropout) for _ in range(cfg.layers)])
        self.norm = nn.LayerNorm(d)
        self.pool = QueryPool(d, cfg.heads, 1 + N_SLOTS)
        self.head_global = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1 + N_COUNT + N_ZONES))
        self.head_slot = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 4))

    # ---- inputs -------------------------------------------------------------

    def _link_inputs(self, batch: dict):
        cfg = self.cfg
        amp = batch["amp"].float()                                           # [B, L, 44, T]
        b, n_links, c, t = amp.shape
        if c != cfg.n_ref:
            raise ValueError(f"amp has {c} subcarriers, the model wants the {cfg.n_ref} reference bins")
        link_mask = batch["link_mask"].bool()
        valid = batch["sample_mask"].bool() & link_mask[..., None]          # [B, L, T]
        vf = valid.float()
        # RSSI: per-link window mean over the bins that carry one, the deviation from it, its spread.
        rmask = batch["rssi_mask"].bool() & valid
        rmf = rmask.float()
        rssi = torch.where(rmask, batch["rssi"].float(), torch.zeros_like(rmf))
        n_r = rmf.sum(-1)                                                    # [B, L]
        r_mean = rssi.sum(-1) / n_r.clamp(min=1)
        r_dev = (rssi - r_mean[..., None]) * rmf
        r_var = (r_dev * r_dev).sum(-1) / (n_r - 1).clamp(min=1)
        r_std = torch.where(n_r >= 2, r_var.clamp(min=0).sqrt(), torch.zeros_like(r_var))
        # time-major channels [B, L, T, C_in]: amp (44), RSSI deviation, sample mask, ablations
        chans = [amp.transpose(2, 3) * vf[..., None], r_dev[..., None], vf[..., None]]
        if cfg.phase:
            ph = batch["phase"].float()                                      # [B, L, 44, T, 2] (cos, sin)
            pm = batch["phase_mask"].bool() & valid[:, :, None, :]
            ph = ph * pm[..., None].float()
            chans.append(ph.permute(0, 1, 3, 4, 2).reshape(b, n_links, t, 2 * cfg.n_ref))  # [cos x 44, sin x 44]
        if cfg.gain:
            g = torch.where(valid, batch["gain_db"].float(), torch.zeros_like(vf))
            g_mean = g.sum(-1) / vf.sum(-1).clamp(min=1)
            chans.append(((g - g_mean[..., None]) * vf)[..., None])
        x = torch.cat(chans, dim=-1)                                         # [B, L, T, C_in]
        lm = link_mask[..., None].float()
        mu = batch["mu"].float()
        sig = batch["sigma"].float().clamp(min=0)
        side = torch.cat([
            (mu - cfg.mu_centre) / cfg.mu_scale,
            (torch.log(sig + 1e-4) - cfg.logsig_centre) / cfg.logsig_scale,
            ((r_mean - cfg.rssi_centre) / cfg.rssi_scale)[..., None],
            ((torch.log(r_std + 0.1) - cfg.logstd_centre) / cfg.logstd_scale)[..., None],
        ], dim=-1) * lm                                                      # [B, L, 90]; padded links 0
        return x, valid, side, link_mask

    def _geometry(self, batch: dict, link_mask: torch.Tensor):
        cfg = self.cfg
        known = batch["geometry_known"].bool()                              # [B]
        room = batch["room"].float()                                         # [B, 3] m
        ok = known[:, None] & (room > 0)
        safe = torch.where(ok, room, torch.ones_like(room))                  # never divide by an unknown room
        pos = batch["link_pos"].float() / safe[:, None, None, :]            # [B, L, 2, 3] normalised
        pos = pos * (known[:, None, None, None] & link_mask[..., None, None]).float()
        tx, rx = pos[:, :, 0], pos[:, :, 1]
        mid = 0.5 * (tx + rx)
        dxy = rx[..., :2] - tx[..., :2]
        length = (dxy * dxy).sum(-1, keepdim=True).clamp(min=1e-12).sqrt()
        far = length > 1e-6
        cos = torch.where(far, dxy[..., :1] / length, torch.ones_like(length))
        sin = torch.where(far, dxy[..., 1:] / length, torch.zeros_like(length))
        length = torch.where(far, length, torch.zeros_like(length))
        feats = torch.cat([tx, rx, mid, length, cos, sin], dim=-1)          # [B, L, 12]
        geo = torch.where(known[:, None, None], self.geo(feats), self.unknown_geo.to(feats.dtype))
        room_tok = torch.where(known[:, None], self.room(safe / cfg.room_scale),
                               self.unknown_room.to(safe.dtype))             # [B, D]
        return geo, room_tok, safe

    # ---- forward ------------------------------------------------------------

    def forward(self, batch: dict) -> dict:
        cfg = self.cfg
        x, valid, side, link_mask = self._link_inputs(batch)
        b, n_links, t, c_in = x.shape
        xf, vf = x.reshape(b * n_links, t, c_in), valid.reshape(b * n_links, t)
        pack = batch.get("pack_index")
        if pack is None:
            h = self.encoder(xf, vf)
        else:
            # Encode only the listed links (`pack_index`, flat b * L + l; entries equal to B * L are
            # padding): the encoder is per link, so this is the same function on fewer rows. The others
            # stay 0, which nothing reads (their tokens are masked).
            real = pack < b * n_links
            src = torch.where(real, pack, torch.zeros_like(pack))
            hp = self.encoder(xf[src], vf[src] & real[:, None])
            h = hp.new_zeros(b * n_links + 1, hp.shape[-1]).index_copy(0, pack, hp)[:-1]
        h = h.reshape(b, n_links, -1)
        geo, room_tok, room = self._geometry(batch, link_mask)
        tok = h + self.side(side).to(h.dtype) + geo.to(h.dtype)
        tokens = torch.cat([room_tok.to(tok.dtype)[:, None, :], tok], dim=1)  # [B, 1 + L, D]
        keep = torch.cat([torch.ones_like(link_mask[:, :1]), link_mask], dim=1)  # the room token is always kept
        for blk in self.blocks:
            tokens = blk(tokens, keep)
        tokens = self.norm(tokens)
        pooled = self.pool(tokens, keep)                                     # [B, 4, D]
        g = self.head_global(pooled[:, 0]).float()                          # [B, 14]
        s = self.head_slot(pooled[:, 1:]).float()                           # [B, 3, 4]
        centre = torch.sigmoid(s[..., 1:3])
        sigma = cfg.sigma_min + F.softplus(s[..., 3])
        # straight-through cap: the forward value is exactly sigma_max above it, the gradient sigma's own
        sigma = torch.where(sigma > cfg.sigma_max, cfg.sigma_max + (sigma - sigma.detach()), sigma)
        return {
            "presence_logit": g[:, 0],
            "count_logits": g[:, 1:1 + N_COUNT],
            "zone_logits": g[:, 1 + N_COUNT:],
            "slot_logit": s[..., 0],
            "slot_centre": centre,
            "slot_xy": centre * room[:, None, :2],
            "slot_sigma": sigma,
        }


def pack_index(link_mask: torch.Tensor, n_links: int | None = None, bucket: int = 256) -> torch.Tensor:
    """The flat indices `b * L + l` of the real links of a batch whose link
    axis is (or will be padded to) `n_links`, padded with `B * L` to a
    multiple of `bucket` (few distinct shapes for a compiled graph). [N] int64,
    on `link_mask`'s device."""
    b, n = link_mask.shape
    n_links = n if n_links is None else int(n_links)
    if n_links < n:
        raise ValueError(f"cannot pack {n} links into {n_links}")
    bi, li = torch.nonzero(link_mask.bool(), as_tuple=True)
    idx = bi * n_links + li
    size = max(bucket, -(-idx.numel() // bucket) * bucket)
    out = torch.full((size,), b * n_links, dtype=torch.int64, device=link_mask.device)
    out[:idx.numel()] = idx
    return out


# --------------------------------------------------------------------------
# Matching and losses (spec/model.md §2 Losses)
# --------------------------------------------------------------------------


def _assignments(n_slots: int, n_people: int, device) -> torch.Tensor:
    """Every injective map person -> slot: [A, n_people] slot indices."""
    return torch.tensor(list(itertools.permutations(range(n_slots), n_people)), dtype=torch.long, device=device)


def pairwise_dist(pred_xy: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    """[B, P, S] Euclidean distances (m) from person p to slot s."""
    diff = xy[:, :, None, :].float() - pred_xy[:, None, :, :].float()
    return (diff * diff).sum(-1).sqrt()


@torch.no_grad()
def match_enumerate(pred_xy: torch.Tensor, xy: torch.Tensor, person_mask: torch.Tensor) -> torch.Tensor:
    """Minimum total Euclidean distance assignment of the present people to
    slots, by scoring every injective assignment at once (6 for 3 x 3).
    Returns [B, P] slot index per person, -1 where the person is absent."""
    b, n_p, _ = xy.shape
    n_s = pred_xy.shape[1]
    if n_p > n_s:
        raise ValueError(f"{n_p} people for {n_s} slots")
    dist = pairwise_dist(pred_xy, xy)                                        # [B, P, S]
    perms = _assignments(n_s, n_p, dist.device)                              # [A, P]
    person = torch.arange(n_p, device=dist.device)
    cost_each = dist[:, person[None, :], perms]                              # [B, A, P]
    m = person_mask.bool()
    cost = (cost_each * m[:, None, :].float()).sum(-1)                       # [B, A]
    best = cost.argmin(dim=1)
    slot = perms[best]                                                       # [B, P]
    return torch.where(m, slot, torch.full_like(slot, -1))


@torch.no_grad()
def match_scipy(pred_xy: torch.Tensor, xy: torch.Tensor, person_mask: torch.Tensor) -> torch.Tensor:
    """The same assignment by scipy.optimize.linear_sum_assignment, per sample (CPU)."""
    from scipy.optimize import linear_sum_assignment

    dist = pairwise_dist(pred_xy, xy).cpu().numpy()
    m = person_mask.bool().cpu().numpy()
    slot = np.full(m.shape, -1, dtype=np.int64)
    for i in range(m.shape[0]):
        idx = np.flatnonzero(m[i])
        if idx.size:
            r, c = linear_sum_assignment(dist[i][idx])
            slot[i, idx[r]] = c
    return torch.from_numpy(slot).to(pred_xy.device)


MATCHERS = {"enumerate": match_enumerate, "scipy": match_scipy}


def gaussian_nll_2d(err2: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """-log N(x; mu, sigma^2 I_2) for squared error `err2` (m^2) and sigma (m)."""
    return LOG_2PI + 2.0 * torch.log(sigma) + err2 / (2.0 * sigma * sigma)


def compute_loss(out: dict, batch: dict, weights: dict | None = None, matcher: str = "enumerate") -> dict:
    """spec/model.md §2 Losses. Presence BCE + count CE + zone BCE over
    reachable cells + (Hungarian match of slots to present people in metres)
    0.5 x the matched people's 2-D Gaussian NLL + existence BCE (matched 1,
    others 0). Zone, NLL and existence are skipped where `geometry_known` is
    false; absent people, unreachable cells and padded links never enter.
    Returns float32 scalars: `total` and each term (unweighted), plus the
    matched people's `n_matched`."""
    w = dict(DEFAULT_LOSS_WEIGHTS)
    if weights:
        unknown = set(weights) - set(w)
        if unknown:
            raise ValueError(f"unknown loss weights {sorted(unknown)}")
        w.update(weights)
    known = batch["geometry_known"].bool()
    presence = batch["presence"].float()
    l_presence = F.binary_cross_entropy_with_logits(out["presence_logit"].float(), presence)
    l_count = F.cross_entropy(out["count_logits"].float(), batch["count"].long())

    zmask = batch["zone_reachable"].bool() & known[:, None]
    zl = F.binary_cross_entropy_with_logits(out["zone_logits"].float(), batch["zone_occupied"].float(),
                                            reduction="none")
    l_zone = (zl * zmask.float()).sum() / zmask.sum().clamp(min=1)

    pmask = batch["person_mask"].bool() & known[:, None]                     # [B, P]
    xy = torch.where(pmask[..., None], batch["xy"].float(), torch.zeros_like(batch["xy"].float()))
    pred_xy = out["slot_xy"].float()
    slot = MATCHERS[matcher](pred_xy, xy, pmask)                             # [B, P], -1 absent
    sel = slot.clamp(min=0)
    m_xy = torch.gather(pred_xy, 1, sel[..., None].expand(-1, -1, 2))        # [B, P, 2]
    m_sigma = torch.gather(out["slot_sigma"].float(), 1, sel)                # [B, P]
    err2 = ((m_xy - xy) ** 2).sum(-1)
    nll = gaussian_nll_2d(err2, m_sigma)
    pm = pmask.float()
    n_matched = pm.sum()
    l_nll = (nll * pm).sum() / n_matched.clamp(min=1)

    n_slots = out["slot_logit"].shape[1]
    exist_t = torch.zeros_like(out["slot_logit"].float())
    exist_t.scatter_add_(1, sel, pm)                                         # matched slots 1, others 0
    emask = known[:, None].expand(-1, n_slots).float()
    el = F.binary_cross_entropy_with_logits(out["slot_logit"].float(), exist_t.clamp(max=1), reduction="none")
    l_exist = (el * emask).sum() / emask.sum().clamp(min=1)

    total = (w["presence"] * l_presence + w["count"] * l_count + w["zone"] * l_zone
             + w["xy_nll"] * l_nll + w["exist"] * l_exist)
    return {"total": total, "presence": l_presence, "count": l_count, "zone": l_zone, "xy_nll": l_nll,
            "exist": l_exist, "n_matched": n_matched.detach()}


# --------------------------------------------------------------------------
# Predictions (the evaluation contract) and helpers
# --------------------------------------------------------------------------


def radius95(sigma):
    """The reported 95% radius: `max(2.448 sigma, 0.3)` m (spec/model.md
    decision 3; spec/locate.md §9's `min_plausible_radius_m`)."""
    if isinstance(sigma, torch.Tensor):
        return (R95 * sigma).clamp(min=MIN_PLAUSIBLE_RADIUS_M)
    return np.maximum(R95 * np.asarray(sigma, dtype=np.float64), MIN_PLAUSIBLE_RADIUS_M)


@torch.no_grad()
def predictions(out: dict) -> dict:
    """Model outputs -> the evaluation contract (numpy, float32):
    presence_prob [N], count_prob [N, 4], zone_prob [N, 9], slot_exist
    [N, 3], slot_xy [N, 3, 2] m (room frame), slot_sigma [N, 3] m."""
    return {
        "presence_prob": torch.sigmoid(out["presence_logit"].float()).cpu().numpy().astype(np.float32),
        "count_prob": torch.softmax(out["count_logits"].float(), -1).cpu().numpy().astype(np.float32),
        "zone_prob": torch.sigmoid(out["zone_logits"].float()).cpu().numpy().astype(np.float32),
        "slot_exist": torch.sigmoid(out["slot_logit"].float()).cpu().numpy().astype(np.float32),
        "slot_xy": out["slot_xy"].float().cpu().numpy().astype(np.float32),
        "slot_sigma": out["slot_sigma"].float().cpu().numpy().astype(np.float32),
    }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(cfg: ModelConfig | dict | None = None, seed: int | None = None) -> SaansModel:
    """A model with its initialisation fixed by `seed` (when given)."""
    if isinstance(cfg, dict):
        cfg = ModelConfig.from_dict(cfg)
    if seed is not None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            return SaansModel(cfg)
    return SaansModel(cfg)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Model v1 (spec/model.md §2): parameter counts. SIM.")
    ap.add_argument("--widths", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--phase", action="store_true")
    ap.add_argument("--gain", action="store_true")
    args = ap.parse_args(argv)
    rows = {}
    for d in args.widths:
        m = build_model(ModelConfig(d=d, phase=args.phase, gain=args.gain), seed=0)
        rows[d] = count_parameters(m)
    print(json.dumps({"label": "SIM", "params": rows, "config": asdict(ModelConfig())}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
