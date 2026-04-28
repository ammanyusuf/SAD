import argparse
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import third_party.mdlm.noise_schedule as _ns
import third_party.mdlm.repellency.safe_denoiser as _sd
from third_party.mdlm.repellency.safe_denoiser import MaskKernelRepellency

# -----------------------------
# Data generation (SAFE / UNSAFE) of A/B pairs
# -----------------------------

def parse_bumps(arg: str) -> List[Tuple[int, int, float]]:
    """
    Parse a bumps spec like: "start:len:pA, start:len:pA, ..."
    Returns a list of (start, length, pA). Empty/None -> [].
    """
    if not arg or arg.strip().lower() in {"", "none", "[]"}:
        return []
    out = []
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    for p in parts:
        st, ln, pa = p.split(":")
        out.append((int(st), int(ln), float(pa)))
    return out

def build_pA_profile(L: int, baseline_pA: float, bumps: List[Tuple[int, int, float]]) -> np.ndarray:
    """
    Start from a constant baseline pA, then override ranges with given bump pA.
    Clamps to [0,1].
    """
    pA = np.ones(L, dtype=float) * float(baseline_pA)
    for start, length, pval in bumps:
        start = max(0, start); end = min(L, start + max(0, length))
        if start < end:
            pA[start:end] = float(pval)
    return np.clip(pA, 0.0, 1.0)

def preset_profiles(preset: str, L: int) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (pA_safe, pA_unsafe) for quick demos."""
    if preset == "all_a_vs_all_b":
        return np.ones(L), np.zeros(L)
    if preset == "all_b_vs_all_a":
        return np.zeros(L), np.ones(L)
    if preset == "left_A_right_B_vs_left_B_right_A":
        pA_safe = np.zeros(L) + 0.9; pA_safe[L//2:] = 0.1
        pA_unsafe = 1.0 - pA_safe
        return pA_safe, pA_unsafe
    if preset == "two_bumps_opposed":
        pA_safe = np.ones(L) * 0.9; pA_safe[L//4:L//4+6] = 0.1
        pA_unsafe = np.zeros(L) + 0.1; pA_unsafe[2*L//3:2*L//3+8] = 0.95
        return pA_safe, pA_unsafe
    raise ValueError(f"Unknown preset '{preset}'.")

def sample_sequences_from_pA(pA: np.ndarray, n: int, A_token=0, B_token=1) -> np.ndarray:
    """Sample shape (n, L) sequences with Bernoulli(pA) for token A, else B."""
    L = pA.shape[0]
    draws = (np.random.rand(n, L) < pA).astype(np.int64)
    return np.where(draws, A_token, B_token)

# -----------------------------
# Schedules (move(t))
# -----------------------------

def move_from_noise_module(T: int, noise_type: str, sigma_min: float, sigma_max: float, eps: float, dtype=torch.float32) -> np.ndarray:
    """Uses your noise_schedule.py classes to produce move(t) = 1 - exp(-total_noise(t))."""
    noise_type = noise_type.lower()
    if   noise_type == "geometric":
        noise = _ns.GeometricNoise(sigma_min=sigma_min, sigma_max=sigma_max)
    elif noise_type == "loglinear":
        noise = _ns.LogLinearNoise(eps=eps)
    elif noise_type == "cosine":
        noise = _ns.CosineNoise(eps=eps)
    elif noise_type == "cosinesqr":
        noise = _ns.CosineSqrNoise(eps=eps)
    elif noise_type == "linear":
        noise = _ns.Linear(sigma_min=sigma_min, sigma_max=sigma_max, dtype=dtype)
    else:
        raise ValueError(f"Unsupported --noise-type '{noise_type}'")
    # Evaluate total_noise on normalized time grid in [0,1]
    t = torch.linspace(0.0, 1.0, steps=T, dtype=dtype)
    sigma_t = noise.total_noise(t)   # shape [T]
    move = 1.0 - torch.exp(-sigma_t) 
    return move.detach().cpu().numpy()

# -----------------------------
# Empirical posteriors (baseline + unsafe + safe)
# -----------------------------

def summarize_per_position(X_np, V=2):
    L = X_np.shape[1]
    counts = np.zeros((L, V), dtype=float)
    for v in range(V):
        counts[:, v] = (X_np == v).sum(axis=0)
    return counts / np.clip(counts.sum(axis=1, keepdims=True), 1e-12, None)

def forward_mask_cumulative(x0, move):
    xt = x0.clone()
    mask = (torch.rand_like(xt.float()) < move)
    xt[mask] = -1
    return xt

def empirical_data_posterior_mixture(X_safe_t, X_unsafe_t, alpha, xt, V=2):
    """Per-position baseline posterior combining SAFE/UNSAFE with weight alpha for UNSAFE."""
    B, L_ = xt.shape
    safe_counts   = torch.stack([(X_safe_t==v).float().sum(dim=0)   for v in range(V)], dim=1)  # [L,V]
    unsafe_counts = torch.stack([(X_unsafe_t==v).float().sum(dim=0) for v in range(V)], dim=1)  # [L,V]
    mix_counts = (1.0 - alpha) * safe_counts + alpha * unsafe_counts
    p_pos = mix_counts / mix_counts.sum(dim=1, keepdim=True).clamp_min(1e-12)  # [L,V]
    p = p_pos.unsqueeze(0).expand(B, L_, V).contiguous()
    is_mask = (xt == -1)
    for b in range(B):
        if (~is_mask[b]).any():
            p[b, ~is_mask[b]] = torch.nn.functional.one_hot(
                xt[b, ~is_mask[b]].clamp(min=0), num_classes=V
            ).to(p.dtype)
    return p

# -----------------------------
# Plot helpers
# -----------------------------

def plot_position_bars(ax, probs, title):
    L = probs.shape[0]
    x = np.arange(L)
    A_prob = probs[:, 0]
    B_prob = probs[:, 1]
    ax.bar(x, A_prob, width=0.8, label='A', alpha=0.85)
    ax.bar(x, B_prob, bottom=A_prob, width=0.8, label='B', alpha=0.85)
    ax.set_ylim(0, 1.0); ax.set_xlim(-0.5, L-0.5)
    ax.set_title(title); ax.set_xlabel("Position i"); ax.set_ylabel("P(token)")

def draw_bars(ax, probs, title, L):
    ax.cla()
    x = np.arange(L)
    A_prob = probs[:, 0]; B_prob = probs[:, 1]
    ax.bar(x, A_prob, width=0.8, alpha=0.85, label='A')
    ax.bar(x, B_prob, width=0.8, bottom=A_prob, alpha=0.85, label='B')
    ax.set_ylim(0, 1.0); ax.set_xlim(-0.5, L-0.5)
    ax.set_title(title); ax.set_xlabel("Position i"); ax.set_ylabel("P(token)")
    ax.legend(loc="upper right")

def draw_sample_row(ax, xt_np, L, mask_index=-1):
    """
    Visualize current sample as bars: A=blue, B=orange; masked=lightgray.
    """
    ax.cla()
    x = np.arange(L)
    A_mask = (xt_np == 0)
    B_mask = (xt_np == 1)
    M_mask = (xt_np == mask_index)

    ax.bar(x[M_mask], np.ones(M_mask.sum()), width=0.8, color="#bdc3c7", alpha=0.9, label="MASK")

    ax.bar(x[A_mask], np.ones(A_mask.sum()), width=0.8, alpha=0.9, label="A")
    ax.bar(x[B_mask], np.ones(B_mask.sum()), width=0.8, alpha=0.9, label="B")

    ax.set_ylim(0, 1.0); ax.set_xlim(-0.5, L-0.5)
    ax.set_title("Current sample (reverse)"); ax.set_xlabel("Position i"); ax.set_yticks([])
    ax.legend(loc="upper right")



# --------
# Reverse step sampler
# --------
def sample_categorical_exp_trick(weights: torch.Tensor) -> torch.Tensor:
    """
    weights: [..., K] nonnegative (need not sum to 1).
    returns: argmax over weights / Exp(1)
    """
    eps = 1e-10
    U = torch.rand_like(weights).clamp_min(eps)
    E = -torch.log(U)                # Exp(1)
    return torch.argmax(weights / (E + eps), dim=-1)


@torch.no_grad()
def ddpm_cache_step_safe(x_t: torch.LongTensor,
                         t_val: float, dt: float,
                         mk: MaskKernelRepellency,
                         X_safe_t: torch.LongTensor,
                         X_unsafe_t: torch.LongTensor,
                         mix_alpha: float,
                         V: int,
                         mask_index: int,
                         which: str = "safe",
                         ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    One MDLM-style (subs/loglinear) cache step with selectable posterior:
      - which="safe"   -> p_base = conditioning_1(p_mix, ...)
      - which="mix"    -> p_base = p_mix
      - which="unsafe" -> p_base = p_U
    Transition kernel matches _ddpm_caching_update:
      token mass = (t - s) * p_base[..., v], mask mass = s
      
    x_t: [B, L] current partially revealed sample (mask_index for masked)
    t_val: current 'move'
    dt: step size in 'move'
    Returns: (p_safe used, x_{s}) with the MDLM cache update and copy-flag behavior.
    """
    B, L = x_t.shape
    device = x_t.device
    t = torch.tensor([t_val], device=device, dtype=torch.float32)
    s = torch.tensor([max(t_val - dt, 0.0)], device=device, dtype=torch.float32)

    p_mix = empirical_data_posterior_mixture(X_safe_t, X_unsafe_t, mix_alpha, x_t, V=V)
    pU, _ = mk.empirical_denoiser(x_t=x_t, move=t)
    
    if which == "safe":
        p_base = mk.conditioning_1(x_0_hat=p_mix, x_t=x_t, move=t)["x_0_hat"]
    elif which == "mix":
        p_base = p_mix
    elif which == "unsafe":
        p_base = pU
    else:
        raise ValueError(f"which must be one of 'safe'|'mix'|'unsafe', got {which!r}")

    q_tokens = (t - s).view(B, 1, 1) * p_base
    q_mask   = s.view(B, 1, 1).expand(B, L, 1)
    q_full   = torch.cat([q_tokens, q_mask], dim=-1)

    choice = sample_categorical_exp_trick(q_full)

    x_prop = choice.clone()
    x_prop[x_prop == V] = mask_index 

    copy_flag = (x_t != mask_index).to(x_t.dtype)
    x_next = copy_flag * x_t + (1 - copy_flag) * x_prop

    return p_base, x_next

@torch.no_grad()
def sample_one_reverse_mdls(L: int,
                            mk: MaskKernelRepellency,
                            X_safe_t: torch.LongTensor,
                            X_unsafe_t: torch.LongTensor,
                            mix_alpha: float,
                            V: int,
                            mask_index: int,
                            num_steps: int,
                            eps: float = 1e-5,
                            which: str = "safe",
                            ) -> torch.LongTensor:
    """
    Draw one full x0 sample using MDLM-like ddpm_cache_update with SAFE guidance.
    Reverse time grid: t in [1 .. eps], dt = (1-eps)/num_steps
    """
    device = X_safe_t.device
    x = torch.full((1, L), fill_value=mask_index, dtype=torch.long, device=device)
    dt = float((1.0 - eps) / num_steps)
    timesteps = torch.linspace(1.0, eps, num_steps + 1, device=device)
    for i in range(num_steps):
        t_val = float(timesteps[i].item())
        _, x_next = ddpm_cache_step_safe(
            x_t=x, t_val=t_val, dt=dt,
            mk=mk, X_safe_t=X_safe_t, X_unsafe_t=X_unsafe_t,
            mix_alpha=mix_alpha, V=V, mask_index=mask_index,
            which=which,
        )
        x = x_next
    return x[0]  # [L]

def animate_reverse_ensemble(L: int, V: int, MASK: int,
                             mk: MaskKernelRepellency,
                             X_safe_t: torch.LongTensor,
                             X_unsafe_t: torch.LongTensor,
                             mix_alpha: float,
                             num_steps: int,
                             n_samples: int,
                             eps: float,
                             outdir: Path,
                             fps: int,
                             save_frames: bool,
                             title: str,
                             seed: int):
    """
    Draw N full reverse samples; after each k=1..N, update the empirical per-position
    distribution over {A,B} and animate its convergence.
    Saves frames (optional) and a GIF to <outdir>/reverse_ensemble/.
    """
    ens_dir = outdir / "reverse_ensemble"
    (ens_dir / "frames").mkdir(parents=True, exist_ok=True)

    device = X_safe_t.device
    counts = torch.zeros(L, V, device=device, dtype=torch.float32)
    rng = np.random.default_rng(seed + 999)

    fig, ax = plt.subplots(1, 1, figsize=(14, 3.5), constrained_layout=True)
    ax.set_ylim(0, 1.0); ax.set_xlim(-0.5, L-0.5)

    def draw_empirical(ax_, probs_, k_):
        ax_.cla()
        x = np.arange(L)
        A_prob = probs_[:, 0]; B_prob = probs_[:, 1]
        ax_.bar(x, A_prob, width=0.8, alpha=0.85, label='A')
        ax_.bar(x, B_prob, width=0.8, bottom=A_prob, alpha=0.85, label='B')
        ax_.set_ylim(0, 1.0); ax_.set_xlim(-0.5, L-0.5)
        ax_.set_title(f"Empirical p_N(x0) from reverse samples — N={k_}/{n_samples}")
        ax_.set_xlabel("Position i"); ax_.set_ylabel("P(token)"); ax_.legend(loc="upper right")

    def update(frame_idx):
        k = frame_idx + 1
        xk = sample_one_reverse_mdls(
            L=L, mk=mk, X_safe_t=X_safe_t, X_unsafe_t=X_unsafe_t,
            mix_alpha=mix_alpha, V=V, mask_index=MASK,
            num_steps=num_steps, eps=eps
        )

        counts[torch.arange(L), xk] += 1.0
        probs = (counts / k).clamp_min(1e-12).detach().cpu().numpy()
        draw_empirical(ax, probs, k)

        if save_frames:
            fig.savefig(ens_dir / "frames" / f"frame_{k:04d}.png", dpi=140)

    ani = FuncAnimation(fig, update, frames=n_samples, interval=1000/fps)
    gif_path = ens_dir / "ensemble.gif"
    ani.save(gif_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"[saved] reverse-ensemble gif: {gif_path}")

def animate_reverse_ensemble_multi(
    L: int, V: int, MASK: int,
    mk: MaskKernelRepellency,
    X_safe_t: torch.LongTensor,
    X_unsafe_t: torch.LongTensor,
    mix_alpha: float,
    num_steps: int,
    n_samples: int,
    eps: float,
    outdir: Path,
    fps: int,
    save_frames: bool,
    seed: int,
):
    variants = [("safe",   "reverse_ensemble/safe"),
                ("mix",    "reverse_ensemble/mix"),
                ("unsafe", "reverse_ensemble/unsafe")]

    device = X_safe_t.device
    for which, sub in variants:
        ens_dir = outdir / sub
        (ens_dir / "frames").mkdir(parents=True, exist_ok=True)
        counts = torch.zeros(L, V, device=device, dtype=torch.float32)

        fig, ax = plt.subplots(1, 1, figsize=(14, 3.5), constrained_layout=True)
        ax.set_ylim(0, 1.0); ax.set_xlim(-0.5, L-0.5)

        def draw_empirical(ax_, probs_, k_):
            ax_.cla()
            x = np.arange(L)
            A_prob = probs_[:, 0]; B_prob = probs_[:, 1]
            ax_.bar(x, A_prob, width=0.8, alpha=0.85, label='A')
            ax_.bar(x, B_prob, width=0.8, bottom=A_prob, alpha=0.85, label='B')
            ax_.set_ylim(0, 1.0); ax_.set_xlim(-0.5, L-0.5)
            ax_.set_title(f"{which.upper()} ensemble: empirical p_N(x0) — N={k_}/{n_samples}")
            ax_.set_xlabel("Position i"); ax_.set_ylabel("P(token)")
            ax_.legend(loc="upper right")

        def update(frame_idx):
            k = frame_idx + 1
            xk = sample_one_reverse_mdls(
                L=L, mk=mk, X_safe_t=X_safe_t, X_unsafe_t=X_unsafe_t,
                mix_alpha=mix_alpha, V=V, mask_index=MASK,
                num_steps=num_steps, eps=eps, which=which
            )
            counts[torch.arange(L), xk] += 1.0
            probs = (counts / k).clamp_min(1e-12).detach().cpu().numpy()
            draw_empirical(ax, probs, k)
            if save_frames:
                fig.savefig(ens_dir / "frames" / f"frame_{k:04d}.png", dpi=140)

        ani = FuncAnimation(fig, update, frames=n_samples, interval=1000/fps)
        gif_path = ens_dir / "ensemble.gif"
        ani.save(gif_path, writer=PillowWriter(fps=fps))
        plt.close(fig)
        print(f"[saved] {which} reverse-ensemble gif: {gif_path}")


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    # Lengths / counts
    ap.add_argument("--T", type=int, default=30, help="number of timesteps")
    ap.add_argument("--L", type=int, default=40, help="sequence length")
    ap.add_argument("--n_per_mode", type=int, default=400, help="samples per mode for SAFE/UNSAFE")
    ap.add_argument("--seed", type=int, default=3)

    # Schedule source
    ap.add_argument("--move-source", choices=["noise"], default="noise",
                    help="'noise': move(t)=1-exp(-total_noise(t)) from noise_schedule;")
    ap.add_argument("--noise-type", type=str, default="loglinear",
                    choices=["geometric","loglinear","cosine","cosinesqr","linear"],
                    help="which noise_schedule.* class to use when --move-source=noise")
    ap.add_argument("--sigma-min", type=float, default=0.0, help="for geometric/linear")
    ap.add_argument("--sigma-max", type=float, default=10.0, help="for geometric/linear")
    ap.add_argument("--eps", type=float, default=1e-3, help="for cosine/cosinesqr/loglinear")

    # Data generation options
    ap.add_argument("--preset", type=str, default="",
                    choices=["","all_a_vs_all_b","all_b_vs_all_a","left_A_right_B_vs_left_B_right_A","two_bumps_opposed"],
                    help="quick presets; overrides manual baselines/bumps if set")
    ap.add_argument("--safe-baseline-pA", type=float, default=0.90)
    ap.add_argument("--unsafe-baseline-pA", type=float, default=0.10)
    ap.add_argument("--safe-bumps", type=str, default="",
                    help='format: "start:len:pA, start:len:pA, ..."')
    ap.add_argument("--unsafe-bumps", type=str, default="",
                    help='format: "start:len:pA, start:len:pA, ..."')

    # Safe denoiser knobs
    ap.add_argument("--s", type=float, default=3.0, help="repellency strength")
    ap.add_argument("--mix_alpha", type=float, default=0.5,
                    help="UNSAFE weight in baseline p(x|x_t). 1.0 = all-unsafe.")

    # Reverse ensemble
    ap.add_argument("--n_reverse_samples", type=int, default=100,
                help="How many full reverse samples to draw for the ensemble animation.")
    ap.add_argument("--ensemble_eps", type=float, default=1e-5,
                    help="eps for the reverse time grid (like MDLM ddpm_cache).")

    # Output / viz
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--outdir", type=str, default="safe_demo_outputs_onehot")
    ap.add_argument("--save-frames", action="store_true", help="save each frame as PNG in frames/")
    ap.add_argument("--title", type=str, default="Safe Denoiser over Schedule")

    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cpu"
    V = 2
    A, B = 0, 1
    MASK = -1
    L = args.L
    N = args.n_per_mode

    if args.preset:
        pA_safe, pA_unsafe = preset_profiles(args.preset, L)
    else:
        safe_bumps = parse_bumps(args.safe_bumps)
        unsafe_bumps = parse_bumps(args.unsafe_bumps)
        pA_safe = build_pA_profile(L, args.safe_baseline_pA,   safe_bumps)
        pA_unsafe = build_pA_profile(L, args.unsafe_baseline_pA, unsafe_bumps)

    X_safe = sample_sequences_from_pA(pA_safe, N)
    X_unsafe = sample_sequences_from_pA(pA_unsafe, N)
    X_mix = np.concatenate([X_safe, X_unsafe], axis=0)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fwd_dir = outdir / "forward"
    rev_dir = outdir / "reverse"
    (fwd_dir / "frames").mkdir(exist_ok=True, parents=True)
    (rev_dir / "frames").mkdir(exist_ok=True, parents=True)

    ps = summarize_per_position(X_safe,   V)
    pu = summarize_per_position(X_unsafe, V)
    pm = summarize_per_position(X_mix,    V)
    fig_d, axd = plt.subplots(1, 3, figsize=(13, 3.2), constrained_layout=True)
    plot_position_bars(axd[0], ps, "SAFE dataset per-position")
    plot_position_bars(axd[1], pu, "UNSAFE dataset per-position (U)")
    plot_position_bars(axd[2], pm, "MIXTURE (SAFE+UNSAFE) per-position")
    axd[0].legend(loc="upper right")
    fig_d.savefig(outdir / "dist_check_safe_unsafe_mix.png", dpi=140)
    plt.close(fig_d)

    move_schedule = move_from_noise_module(
        T=args.T,
        noise_type=args.noise_type,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        eps=args.eps,
        dtype=torch.float32
    )
    print(f"Using move(t) from noise_schedule '{args.noise_type}' with T={args.T}")
    print("move(t):", np.round(move_schedule, 3))

    X_safe_t   = torch.tensor(X_safe,   device=device)
    X_unsafe_t = torch.tensor(X_unsafe, device=device)
    mk = _sd.MaskKernelRepellency(
        ref_data=torch.tensor(X_unsafe, device=device).long(),
        embed_fn=None, forward_fn=None,
        num_timesteps=args.T, max_idx=L, beta_min=0.0, beta_max=0.0,
        vocab_size=V, mask_index=-1, scale=args.s
    )

    # FORWARD: categorical forward process (for reference)
    # x0 from UNSAFE (to visualize push-away)
    x0 = torch.tensor(X_unsafe[np.random.randint(0, X_unsafe.shape[0])], device=device).unsqueeze(0)

    fig_f, axs_f = plt.subplots(3, 1, figsize=(14, 7), constrained_layout=True)
    for ax in axs_f:
        ax.set_ylim(0, 1.0); ax.set_xlim(-0.5, L-0.5)

    # precompute t=0 sanity
    m0 = float(move_schedule[0])
    xt0 = forward_mask_cumulative(x0, m0)
    X_safe_t   = torch.tensor(X_safe,   device=device)
    X_unsafe_t = torch.tensor(X_unsafe, device=device)
    p0 = empirical_data_posterior_mixture(X_safe_t, X_unsafe_t, args.mix_alpha, xt0, V=V)

    pU0, _ = mk.empirical_denoiser(x_t=xt0.long(), move=torch.tensor([m0], dtype=torch.float32))
    q0 = mk.conditioning_1(x_0_hat=p0, x_t=xt0.long(), move=torch.tensor([m0], dtype=torch.float32))["x_0_hat"]

    if np.isclose(m0, 0.0):
        assert torch.allclose(p0, q0, atol=1e-6), "move(0)=0 should be identity."

    def update_forward(frame_idx):
        m = float(move_schedule[frame_idx])
        xt = forward_mask_cumulative(x0, m)
        p  = empirical_data_posterior_mixture(X_safe_t, X_unsafe_t, args.mix_alpha, xt, V=V)
        pU, _ = mk.empirical_denoiser(x_t=xt.long(), move=torch.tensor([m], dtype=torch.float32))
        q = mk.conditioning_1(x_0_hat=p, x_t=xt.long(), move=torch.tensor([m], dtype=torch.float32))["x_0_hat"]

        draw_bars(axs_f[0], p[0].detach().cpu().numpy(),  f"Baseline p(x|x_t)  —  move={m:.3f}", L)
        draw_bars(axs_f[1], pU[0].detach().cpu().numpy(), f"Unsafe posterior p_U(x|x_t)", L)
        draw_bars(axs_f[2], q[0].detach().cpu().numpy(),  f"Safe p_safe  (s={args.s:.2f})", L)
        fig_f.suptitle(f"{args.title} — t={frame_idx+1}/{len(move_schedule)}", fontsize=14)

        if args.save_frames:
            frame_path = fwd_dir / "frames" / f"frame_{frame_idx+1:04d}.png"
            fig_f.savefig(frame_path, dpi=140)

    ani_f= FuncAnimation(fig_f, update_forward, frames=len(move_schedule), interval=1000/args.fps)
    gif_f = fwd_dir / "forward.gif"
    ani_f.save(gif_f, writer=PillowWriter(fps=args.fps))
    print(f"[saved] {gif_f}")

    # REVERSE: categorical reverse process (the actual denoising)
    # Start fully masked and reveal to match the target keep fraction (1 - move_rev[t])
    move_rev = move_schedule[::-1]  # high→low masking
    xt_rev = torch.full((1, L), fill_value=MASK, dtype=torch.long, device=device)

    fig_r, axs_r = plt.subplots(4, 1, figsize=(14, 9), constrained_layout=True)
    for ax in axs_r[:3]: ax.set_ylim(0, 1.0); ax.set_xlim(-0.5, L-0.5)

    rng = np.random.default_rng(args.seed + 123)
    
    num_steps = args.T
    eps = 1e-5
    timesteps = torch.linspace(1.0, eps, num_steps + 1).cpu().numpy()
    dt = float((1.0 - eps) / num_steps)

    def update_reverse(frame_idx):
        m = float(move_rev[frame_idx])
        keep_target = 1.0 - m
        target_revealed = int(round(keep_target * L))
        is_mask = (xt_rev[0] == MASK).cpu().numpy()
        cur_revealed = int((~is_mask).sum())
        need = max(0, target_revealed - cur_revealed)

        # Posteriors at this step (on current xt_rev, needed for cat dist visualzation)
        p  = empirical_data_posterior_mixture(X_safe_t, X_unsafe_t, args.mix_alpha, xt_rev, V=V)
        pU, _ = mk.empirical_denoiser(x_t=xt_rev.long(), move=torch.tensor([m], dtype=torch.float32))
        q  = mk.conditioning_1(x_0_hat=p, x_t=xt_rev.long(), move=torch.tensor([m], dtype=torch.float32))["x_0_hat"]

        if need > 0 and is_mask.any():
            t_val = float(timesteps[frame_idx])
            p_safe, x_next = ddpm_cache_step_safe(
                x_t=xt_rev, t_val=t_val, dt=dt,
                mk=mk, X_safe_t=X_safe_t, X_unsafe_t=X_unsafe_t,
                mix_alpha=args.mix_alpha, V=V, mask_index=MASK
            )
            xt_rev.copy_(x_next)

        draw_bars(axs_r[0], p[0].detach().cpu().numpy(),  f"Baseline p(x|x_t)  —  move={m:.3f}", L)
        draw_bars(axs_r[1], pU[0].detach().cpu().numpy(), f"Unsafe posterior p_U(x|x_t)", L)
        draw_bars(axs_r[2], q[0].detach().cpu().numpy(),  f"Safe p_safe  (s={args.s:.2f})", L)
        draw_sample_row(axs_r[3], xt_rev[0].detach().cpu().numpy(), L, mask_index=MASK)

        fig_r.suptitle(f"{args.title} — REVERSE (step {frame_idx+1}/{len(move_rev)})", fontsize=14)

        if args.save_frames:
            fig_r.savefig(rev_dir / "frames" / f"frame_{frame_idx+1:04d}.png", dpi=140)

    ani_r = FuncAnimation(fig_r, update_reverse, frames=len(move_rev), interval=1000/args.fps)
    gif_r = rev_dir / "reverse.gif"
    ani_r.save(gif_r, writer=PillowWriter(fps=args.fps))
    plt.close(fig_r)

    print(f"[saved] forward gif: {gif_f}")
    print(f"[saved] reverse gif: {gif_r}")
    if args.save_frames:
        print(f"[frames] {fwd_dir/'frames'} and {rev_dir/'frames'}")

    animate_reverse_ensemble_multi(
        L=L, V=V, MASK=MASK,
        mk=mk,
        X_safe_t=X_safe_t, X_unsafe_t=X_unsafe_t,
        mix_alpha=args.mix_alpha,
        num_steps=args.T,
        n_samples=args.n_reverse_samples,
        eps=args.ensemble_eps,
        outdir=outdir,
        fps=args.fps,
        save_frames=args.save_frames,
        seed=args.seed,
    )



if __name__ == "__main__":
    main()
