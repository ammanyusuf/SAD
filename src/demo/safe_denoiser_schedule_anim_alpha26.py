import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap

from third_party.mdlm.noise_schedule import (
    CosineNoise, CosineSqrNoise, Linear, GeometricNoise, LogLinearNoise
)
from third_party.mdlm.repellency.safe_denoiser import MaskKernelRepellency


# -------------------------
# Data generation (V tokens)
# -------------------------

def alpha_letters(V: int, alphabet: str) -> List[str]:
    if alphabet:
        letters = list(alphabet)
        assert len(letters) == V, f"--alphabet length {len(letters)} != V={V}"
        return letters
    # default: A..Z
    return [chr(ord('A') + i) for i in range(V)]

def parse_bumps26(arg: str, letter_to_idx: dict) -> List[Tuple[int, int, int, float]]:
    """
    Parse bumps like: "start:len:LETTER:p, start:len:LETTER:p, ..."
    Returns a list of (start, length, token_index, prob).
    """
    if not arg or arg.strip().lower() in {"", "none", "[]"}:
        return []
    out = []
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    for p in parts:
        st, ln, let, pr = p.split(":")
        tok = let.strip()
        assert tok in letter_to_idx, f"Unknown token '{tok}' in bumps"
        out.append((int(st), int(ln), int(letter_to_idx[tok]), float(pr)))
    return out

def build_profile_V(L: int, V: int, bumps: List[Tuple[int,int,int,float]]) -> np.ndarray:
    """
    Start uniform over V at all positions, then apply bumps:
      at each bump (start, len, tok_idx, p_tok):
        set that token's prob to p_tok, and distribute the remaining (1-p_tok)
        uniformly over the other V-1 tokens.
    Returns p of shape [L, V], rows sum to 1.
    """
    p = np.full((L, V), 1.0 / V, dtype=float)
    for start, length, tok_idx, p_tok in bumps:
        start = max(0, start); end = min(L, start + max(0, length))
        if start >= end: 
            continue
        rem = max(0.0, 1.0 - p_tok)
        if V > 1:
            filler = rem / (V - 1)
        else:
            filler = 0.0
        p[start:end, :] = filler
        p[start:end, tok_idx] = p_tok
    p /= np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
    return p

def sample_sequences_from_profile(p_pos: np.ndarray, n: int) -> np.ndarray:
    """
    Sample n sequences from per-position categorical p_pos: [L, V] -> [n, L]
    """
    L, V = p_pos.shape
    out = np.empty((n, L), dtype=np.int64)
    for i in range(L):
        out[:, i] = np.random.choice(V, size=n, p=p_pos[i])
    return out


# -------------------------
# Schedules (move(t))
# -------------------------

def move_from_noise(T: int, noise_type: str,
                    sigma_min: float, sigma_max: float, eps: float,
                    dtype=torch.float32) -> np.ndarray:
    """
    move(t) = 1 - exp(- total_noise(t)) evaluated on t∈[0,1], T steps.
    """
    nt = noise_type.lower()
    if   nt == "geometric": noise = GeometricNoise(sigma_min=sigma_min, sigma_max=sigma_max)
    elif nt == "loglinear": noise = LogLinearNoise(eps=eps)
    elif nt == "cosine":    noise = CosineNoise(eps=eps)
    elif nt == "cosinesqr": noise = CosineSqrNoise(eps=eps)
    elif nt == "linear":    noise = Linear(sigma_min=sigma_min, sigma_max=sigma_max, dtype=dtype)
    else: raise ValueError(f"Unsupported --noise-type '{noise_type}'")
    t = torch.linspace(0.0, 1.0, steps=T, dtype=dtype)
    sigma_t = noise.total_noise(t)
    move = 1.0 - torch.exp(-sigma_t)
    return move.detach().cpu().numpy()

def forward_mask_cumulative(x0, move, mask_index=-1):
    xt = x0.clone()
    mask = (torch.rand_like(xt.float()) < move)
    xt[mask] = mask_index
    return xt


# -------------------------
# Posteriors (baseline mix)
# -------------------------

def summarize_per_position(X_np, V):
    L = X_np.shape[1]
    counts = np.zeros((L, V), dtype=float)
    for v in range(V):
        counts[:, v] = (X_np == v).sum(axis=0)
    return counts / np.clip(counts.sum(axis=1, keepdims=True), 1e-12, None)

def empirical_data_posterior_mixture(X_safe_t, X_unsafe_t, alpha, xt, V):
    """
    Per-position baseline posterior combining SAFE/UNSAFE with weight alpha for UNSAFE.
    Unmasked positions get a delta at the observed token.
    """
    B, L_ = xt.shape
    safe_counts   = torch.stack([(X_safe_t==v).float().sum(dim=0)   for v in range(V)], dim=1)
    unsafe_counts = torch.stack([(X_unsafe_t==v).float().sum(dim=0) for v in range(V)], dim=1)
    mix_counts = (1.0 - alpha) * safe_counts + alpha * unsafe_counts
    p_pos = mix_counts / mix_counts.sum(dim=1, keepdim=True).clamp_min(1e-12)

    p = p_pos.unsqueeze(0).expand(B, L_, V).contiguous()
    is_mask = (xt == -1)
    for b in range(B):
        if (~is_mask[b]).any():
            p[b, ~is_mask[b]] = torch.nn.functional.one_hot(
                xt[b, ~is_mask[b]].clamp(min=0), num_classes=V
            ).to(p.dtype)
    return p


# -------------------------
# Visualization helpers (heatmap + mean±std; sample row)
# -------------------------

def _mean_std_from_probs(probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    probs: [L, V] over tokens 0..V-1. Returns mean (L,), std (L,) in token-index space.
    """
    L, V = probs.shape
    idx = np.arange(V)[None, :]
    mean = (probs * idx).sum(axis=1)
    var  = (probs * (idx - mean[:, None])**2).sum(axis=1)
    std  = np.sqrt(np.clip(var, 0.0, None))
    return mean, std

def heatmap_with_mean(ax, probs: np.ndarray, title: str, letters: list[str]):
    """
    probs: [L, V]; draws once (with colorbar) and then only updates data.
    Stores handles on the Axes to avoid re-creating artists every frame.
    """
    L, V = probs.shape
    first_time = not hasattr(ax, "_heat_handles")

    if first_time:
        img = ax.imshow(probs.T, aspect="auto", origin="lower", vmin=0.0, vmax=1.0)
        (line,) = ax.plot([], [], lw=2.0, alpha=0.95, color="white")
        cbar = ax.figure.colorbar(img, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("P(token)")
        ax.set_xlim(0, L - 1)
        ax.set_ylim(0, V - 1)
        ax.set_yticks(np.arange(V))
        ax.set_yticklabels(letters)
        ax.set_xlabel("Position")
        ax._heat_handles = {"img": img, "line": line, "cbar": cbar}
    else:
        img = ax._heat_handles["img"]
        line = ax._heat_handles["line"]
        img.set_data(probs.T)
        img.set_clim(0.0, 1.0)
    idx = np.arange(V)[None, :]
    mean = (probs * idx).sum(axis=1)
    x = np.arange(L)
    line.set_data(x, mean)

    ax.set_title(title)
    return ax._heat_handles["img"]

def draw_sample_row_color(ax, xt_np: np.ndarray, V: int, letters: List[str], mask_index=-1):
    """
    1xL row image showing the current sample: mask=gray, tokens colored.
    """
    ax.cla()
    # build a color palette (mask + 26 tokens)
    base = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors) + list(plt.cm.tab20c.colors)
    if len(base) < V:
        base = base * ((V // len(base)) + 1)
    colors = ['#bdc3c7'] + list(base[:V])  # [mask, tokens...]
    cmap = ListedColormap(colors)
    data = xt_np.copy()
    data = data + 1
    data[data < 0] = 0
    ax.imshow(data[np.newaxis, :], aspect='auto', origin='lower', cmap=cmap, vmin=0, vmax=V)
    ax.set_yticks([0]); ax.set_yticklabels(["sample"])
    ax.set_xlim(0, data.shape[-1]-1)
    ax.set_xlabel("Position")
    ax.set_title("Current sample (mask=gray)")
    return cmap


# -------------------------
# Reverse step (MDLM cache)
# -------------------------

def sample_categorical_exp_trick(weights: torch.Tensor) -> torch.Tensor:
    eps = 1e-10
    U = torch.rand_like(weights).clamp_min(eps)
    E = -torch.log(U)  # Exp(1)
    return torch.argmax(weights / (E + eps), dim=-1)

@torch.no_grad()
def ddpm_cache_step(
    x_t: torch.LongTensor,
    t_val: float,
    dt: float,
    mk: MaskKernelRepellency,
    X_safe_t: torch.LongTensor,
    X_unsafe_t: torch.LongTensor,
    mix_alpha: float,
    V: int,
    mask_index: int,
    which: str = "safe",   # "safe" | "mix" | "unsafe"
):
    """
    One MDLM-style (subs/loglinear timeline) cache step with selectable posterior:
      - which="safe"   -> conditioning_1(p_mix, ...)
      - which="mix"    -> p_mix
      - which="unsafe" -> p_U
    Transition: token mass = (t - s) * p_base[..., v]; mask mass = s
    Returns: (p_base, x_{s})
    """
    B, L = x_t.shape
    device = x_t.device
    t = torch.tensor([t_val], device=device, dtype=torch.float32)
    s = torch.tensor([max(t_val - dt, 0.0)], device=device, dtype=torch.float32)

    p_mix = empirical_data_posterior_mixture(X_safe_t, X_unsafe_t, mix_alpha, x_t, V=V)  # [B,L,V]
    pU, _ = mk.empirical_denoiser(x_t=x_t, move=t)                                       # [B,L,V]
    if which == "safe":
        p_base = mk.conditioning_1(x_0_hat=p_mix, x_t=x_t, move=t)["x_0_hat"]
    elif which == "mix":
        p_base = p_mix
    elif which == "unsafe":
        p_base = pU
    else:
        raise ValueError(f"which must be one of 'safe'|'mix'|'unsafe', got {which!r}")

    q_tokens = (t - s).view(B, 1, 1) * p_base           # [B,L,V]
    q_mask   = s.view(B, 1, 1).expand(B, L, 1)          # [B,L,1]
    q_full   = torch.cat([q_tokens, q_mask], dim=-1)    # [B,L,V+1]

    choice = sample_categorical_exp_trick(q_full)       # [B,L], values 0..V (V := "mask")
    x_prop = choice.clone()
    x_prop[x_prop == V] = mask_index

    copy_flag = (x_t != mask_index).to(x_t.dtype)
    x_next = copy_flag * x_t + (1 - copy_flag) * x_prop
    return p_base, x_next


# --------------------------
# Ensemble animation (multi)
# --------------------------

def animate_reverse_ensemble_multi(
    L: int, V: int, MASK: int, letters: List[str],
    mk: MaskKernelRepellency,
    X_safe_t: torch.LongTensor, X_unsafe_t: torch.LongTensor,
    mix_alpha: float,
    num_steps: int, n_samples: int, eps: float,
    outdir: Path, fps: int, save_frames: bool, seed: int,
):
    variants = [("safe",   "reverse_ensemble/safe"),
                ("mix",    "reverse_ensemble/mix"),
                ("unsafe", "reverse_ensemble/unsafe")]

    device = X_safe_t.device
    for which, sub in variants:
        ens_dir = outdir / sub
        (ens_dir / "frames").mkdir(parents=True, exist_ok=True)

        counts = torch.zeros(L, V, device=device, dtype=torch.float32)

        fig, ax = plt.subplots(1, 1, figsize=(14, 4.0), constrained_layout=True)

        def update(frame_idx):
            k = frame_idx + 1
            x = torch.full((1, L), fill_value=MASK, dtype=torch.long, device=device)
            dt = float((1.0 - eps) / num_steps)
            timesteps = torch.linspace(1.0, eps, num_steps + 1, device=device)
            for i in range(num_steps):
                t_val = float(timesteps[i].item())
                _, x = ddpm_cache_step(
                    x_t=x, t_val=t_val, dt=dt, mk=mk,
                    X_safe_t=X_safe_t, X_unsafe_t=X_unsafe_t,
                    mix_alpha=mix_alpha, V=V, mask_index=MASK, which=which
                )
            x0 = x[0].detach()  # [L]
            counts[torch.arange(L), x0] += 1.0

            probs = (counts / k).clamp_min(1e-12).detach().cpu().numpy()  # [L,V]
            heatmap_with_mean(ax, probs, f"{which.upper()} ensemble — N={k}/{n_samples}", letters)

            if save_frames:
                fig.savefig(ens_dir / "frames" / f"frame_{k:04d}.png", dpi=140)

        ani = FuncAnimation(fig, update, frames=n_samples, interval=1000/fps)
        gif_path = ens_dir / "ensemble.gif"
        ani.save(gif_path, writer=PillowWriter(fps=fps))
        plt.close(fig)
        print(f"[saved] {which} reverse-ensemble gif: {gif_path}")


def main():
    ap = argparse.ArgumentParser()
    # Vocab / length
    ap.add_argument("--V", type=int, default=26, help="vocab size (default 26 for A–Z)")
    ap.add_argument("--alphabet", type=str, default="", help="exact symbol list, length must equal V (default: A..)")
    ap.add_argument("--L", type=int, default=40, help="sequence length")
    ap.add_argument("--n_per_mode", type=int, default=400, help="samples per mode for SAFE/UNSAFE")
    ap.add_argument("--seed", type=int, default=3)

    # Schedule
    ap.add_argument("--T", type=int, default=30, help="number of timesteps")
    ap.add_argument("--noise-type", type=str, default="loglinear",
                    choices=["geometric","loglinear","cosine","cosinesqr","linear"])
    ap.add_argument("--sigma-min", type=float, default=0.0)
    ap.add_argument("--sigma-max", type=float, default=10.0)
    ap.add_argument("--eps", type=float, default=1e-3)

    # Data bumps (V-way)
    ap.add_argument("--safe-bumps26",   type=str, default="", help='e.g., "8:6:A:0.80, 20:4:C:0.6"')
    ap.add_argument("--unsafe-bumps26", type=str, default="", help='e.g., "26:8:Y:0.85"')

    # Safe denoiser / mix
    ap.add_argument("--s", type=float, default=3.0, help="repellency strength")
    ap.add_argument("--mix_alpha", type=float, default=0.6, help="UNSAFE weight in baseline p(x|x_t)")

    # Reverse ensembles
    ap.add_argument("--n_reverse_samples", type=int, default=100)
    ap.add_argument("--ensemble_eps", type=float, default=1e-5)

    # Output
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--outdir", type=str, default="safe_demo_outputs_alpha26")
    ap.add_argument("--save-frames", action="store_true")
    ap.add_argument("--title", type=str, default="Safe Denoiser — V-token Forward & Reverse")

    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cpu"; MASK = -1
    V = args.V; L = args.L; N = args.n_per_mode

    letters = alpha_letters(V, args.alphabet)
    letter_to_idx = {ch: i for i, ch in enumerate(letters)}

    safe_bumps   = parse_bumps26(args.safe_bumps26,   letter_to_idx)
    unsafe_bumps = parse_bumps26(args.unsafe_bumps26, letter_to_idx)
    p_safe_pos   = build_profile_V(L, V, safe_bumps)
    p_unsafe_pos = build_profile_V(L, V, unsafe_bumps)

    X_safe   = sample_sequences_from_profile(p_safe_pos,   N)
    X_unsafe = sample_sequences_from_profile(p_unsafe_pos, N)
    X_mix    = np.concatenate([X_safe, X_unsafe], axis=0)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    fwd_dir = outdir / "forward"; rev_dir = outdir / "reverse"
    (fwd_dir / "frames").mkdir(parents=True, exist_ok=True)
    (rev_dir / "frames").mkdir(parents=True, exist_ok=True)

    fig0, axs0 = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    heatmap_with_mean(axs0[0], summarize_per_position(X_safe,   V), "SAFE per-position", letters)
    heatmap_with_mean(axs0[1], summarize_per_position(X_unsafe, V), "UNSAFE per-position (U)", letters)
    heatmap_with_mean(axs0[2], summarize_per_position(X_mix,    V), "MIXTURE per-position", letters)
    fig0.savefig(outdir / "dist_check_safe_unsafe_mix.png", dpi=140); plt.close(fig0)

    move = move_from_noise(T=args.T, noise_type=args.noise_type,
                           sigma_min=args.sigma_min, sigma_max=args.sigma_max, eps=args.eps)

    X_safe_t   = torch.tensor(X_safe,   device=device)
    X_unsafe_t = torch.tensor(X_unsafe, device=device)
    mk = MaskKernelRepellency(
        ref_data=torch.tensor(X_unsafe, device=device).long(),
        embed_fn=None, forward_fn=None,
        num_timesteps=args.T, max_idx=L, beta_min=0.0, beta_max=0.0,
        vocab_size=V, mask_index=MASK, scale=args.s
    )

    # -----------------------------
    # FORWARD sweep (heatmaps + mean)
    # -----------------------------
    x0 = torch.tensor(X_unsafe[np.random.randint(0, X_unsafe.shape[0])], device=device).unsqueeze(0)

    fig_f, axs_f = plt.subplots(3, 1, figsize=(14, 9), constrained_layout=True)

    def update_forward(frame_idx):
        m = float(move[frame_idx])
        xt = forward_mask_cumulative(x0, m, mask_index=MASK)
        p  = empirical_data_posterior_mixture(X_safe_t, X_unsafe_t, args.mix_alpha, xt, V=V)
        pU, _ = mk.empirical_denoiser(x_t=xt.long(), move=torch.tensor([m], dtype=torch.float32))
        q  = mk.conditioning_1(x_0_hat=p, x_t=xt.long(), move=torch.tensor([m], dtype=torch.float32))["x_0_hat"]

        heatmap_with_mean(axs_f[0], p[0].detach().cpu().numpy(),  f"Baseline p(x|x_t)  —  move={m:.3f}", letters)
        heatmap_with_mean(axs_f[1], pU[0].detach().cpu().numpy(), "Unsafe posterior p_U(x|x_t)", letters)
        heatmap_with_mean(axs_f[2], q[0].detach().cpu().numpy(),  f"Safe p_safe (s={args.s:.2f})", letters)
        fig_f.suptitle(f"{args.title} — FORWARD (t={frame_idx+1}/{len(move)})", fontsize=14)

        if args.save_frames:
            fig_f.savefig(fwd_dir / "frames" / f"frame_{frame_idx+1:04d}.png", dpi=140)

    ani_f = FuncAnimation(fig_f, update_forward, frames=len(move), interval=1000/args.fps)
    gif_f = fwd_dir / "forward.gif"
    ani_f.save(gif_f, writer=PillowWriter(fps=args.fps))
    plt.close(fig_f)
    print(f"[saved] forward gif: {gif_f}")

    # -----------------------------
    # REVERSE sampling (MDLM cache)
    # -----------------------------
    fig_r, axs_r = plt.subplots(4, 1, figsize=(14, 11), constrained_layout=True)
    move_rev = move[::-1]
    num_steps = args.T
    eps_r = 1e-5
    timesteps = torch.linspace(1.0, eps_r, num_steps + 1).cpu().numpy()
    dt = float((1.0 - eps_r) / num_steps)
    xt_rev = torch.full((1, L), fill_value=MASK, dtype=torch.long, device=device)

    def update_reverse(frame_idx):
        # MDLM cache step at t -> t - dt using SAFE posterior
        t_val = float(timesteps[frame_idx])
        p_base, x_next = ddpm_cache_step(
            x_t=xt_rev, t_val=t_val, dt=dt, mk=mk,
            X_safe_t=X_safe_t, X_unsafe_t=X_unsafe_t,
            mix_alpha=args.mix_alpha, V=V, mask_index=MASK, which="safe"
        )
        xt_rev.copy_(x_next)

        # For visualization also get p_mix and p_U at the current xt
        m_vis = float(move_rev[frame_idx])
        p_mix = empirical_data_posterior_mixture(X_safe_t, X_unsafe_t, args.mix_alpha, xt_rev, V=V)
        pU, _ = mk.empirical_denoiser(x_t=xt_rev.long(), move=torch.tensor([m_vis], dtype=torch.float32))
        p_safe = p_base

        heatmap_with_mean(axs_r[0], p_mix[0].detach().cpu().numpy(), f"Baseline p(x|x_t)  —  move≈{m_vis:.3f}", letters)
        heatmap_with_mean(axs_r[1], pU[0].detach().cpu().numpy(),   "Unsafe posterior p_U(x|x_t)", letters)
        heatmap_with_mean(axs_r[2], p_safe[0].detach().cpu().numpy(),"Safe p_safe", letters)
        draw_sample_row_color(axs_r[3], xt_rev[0].detach().cpu().numpy(), V, letters, mask_index=MASK)

        fig_r.suptitle(f"{args.title} — REVERSE (step {frame_idx+1}/{len(move_rev)})", fontsize=14)
        if args.save_frames:
            fig_r.savefig(rev_dir / "frames" / f"frame_{frame_idx+1:04d}.png", dpi=140)

    ani_r = FuncAnimation(fig_r, update_reverse, frames=len(move_rev), interval=1000/args.fps)
    gif_r = rev_dir / "reverse.gif"
    ani_r.save(gif_r, writer=PillowWriter(fps=args.fps))
    plt.close(fig_r)
    print(f"[saved] reverse gif: {gif_r}")

    # -----------------------------
    # REVERSE ENSEMBLES (SAFE/MIX/UNSAFE)
    # -----------------------------
    animate_reverse_ensemble_multi(
        L=L, V=V, MASK=MASK, letters=letters,
        mk=mk, X_safe_t=X_safe_t, X_unsafe_t=X_unsafe_t,
        mix_alpha=args.mix_alpha, num_steps=args.T,
        n_samples=args.n_reverse_samples, eps=args.ensemble_eps,
        outdir=outdir, fps=args.fps, save_frames=args.save_frames, seed=args.seed
    )

if __name__ == "__main__":
    main()
