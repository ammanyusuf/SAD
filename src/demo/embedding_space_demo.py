"""
Embedding-Space Visualization for MDLM Safe Denoiser

This script demonstrates how the safe denoiser guides sampling trajectories
away from unsafe regions in continuous embedding space.

Works on CPU without GPU by default. Follows exact ddpm_cache sampling logic.
"""

import argparse
import os
import sys
from typing import List, Tuple, Dict, Optional
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial import ConvexHull
import seaborn as sns

import third_party.mdlm.noise_schedule as _ns
from third_party.mdlm.repellency.safe_denoiser import MaskKernelRepellency

warnings.filterwarnings('ignore')

# ============================================================================
# PREDEFINED CONCEPT PAIRS
# ============================================================================

CONCEPT_PAIRS = {
    'medical_vs_sports': {
        'safe': [
            'doctor', 'hospital', 'patient', 'medicine', 'surgery', 'diagnosis',
            'treatment', 'therapy', 'clinical', 'physician', 'medical', 'health',
            'disease', 'symptom', 'prescription', 'clinic', 'nurse', 'pharmacist',
            'examination', 'vaccine', 'antibiotics', 'diagnosis', 'cardiac', 'neurological',
            'respiratory', 'examination', 'prognosis', 'rehabilitation', 'therapeutic'
        ],
        'unsafe': [
            'football', 'basketball', 'soccer', 'baseball', 'tennis', 'hockey',
            'athlete', 'championship', 'tournament', 'stadium', 'coach', 'player',
            'team', 'score', 'goal', 'referee', 'league', 'match', 'game',
            'competition', 'victory', 'defeat', 'playoff', 'championship', 'athletic',
            'training', 'workout', 'exercise', 'fitness'
        ]
    },

    'formal_vs_casual': {
        'safe': [
            'therefore', 'furthermore', 'nevertheless', 'consequently', 'accordingly',
            'henceforth', 'notwithstanding', 'moreover', 'subsequently', 'heretofore',
            'pursuant', 'aforementioned', 'whereby', 'wherein', 'herein', 'thereof',
            'endeavor', 'facilitate', 'implement', 'utilize', 'demonstrate', 'indicate',
            'establish', 'constitute', 'approximately', 'substantial', 'comprehensive',
            'commence', 'terminate'
        ],
        'unsafe': [
            'yeah', 'nope', 'gonna', 'wanna', 'gotta', 'kinda', 'sorta', 'dunno',
            'ain', 'yep', 'nah', 'okay', 'cool', 'stuff', 'things', 'guy', 'folks',
            'kids', 'bunch', 'lots', 'pretty', 'really', 'super', 'awesome', 'crazy',
            'weird', 'fun', 'nice', 'good', 'bad'
        ]
    },

    'sentiment': {
        'safe': [
            'excellent', 'wonderful', 'fantastic', 'brilliant', 'magnificent', 'outstanding',
            'superb', 'amazing', 'delightful', 'remarkable', 'exceptional', 'beautiful',
            'joyful', 'pleasant', 'cheerful', 'happy', 'positive', 'optimistic',
            'encouraging', 'uplifting', 'inspiring', 'admirable', 'praiseworthy', 'commendable',
            'favorable', 'beneficial', 'valuable', 'worthwhile', 'successful'
        ],
        'unsafe': [
            'terrible', 'horrible', 'awful', 'dreadful', 'atrocious', 'abysmal',
            'disastrous', 'pathetic', 'miserable', 'disappointing', 'unfortunate', 'regrettable',
            'sad', 'depressing', 'gloomy', 'pessimistic', 'negative', 'discouraging',
            'disheartening', 'unfavorable', 'detrimental', 'harmful', 'worthless', 'useless',
            'failure', 'defeat', 'loss', 'decline', 'deterioration'
        ]
    },

    'science_vs_everyday': {
        'safe': [
            'molecule', 'hypothesis', 'experiment', 'quantum', 'electron', 'neutron',
            'photosynthesis', 'chromosome', 'mitochondria', 'enzyme', 'catalyst', 'isotope',
            'thermodynamics', 'electromagnetic', 'physics', 'chemistry', 'biology', 'algorithm',
            'theorem', 'equation', 'variable', 'coefficient', 'derivative', 'integral',
            'exponential', 'logarithm', 'polynomial', 'matrix', 'vector'
        ],
        'unsafe': [
            'chair', 'table', 'door', 'window', 'floor', 'wall', 'ceiling', 'kitchen',
            'bedroom', 'bathroom', 'food', 'water', 'coffee', 'bread', 'clothes', 'shoes',
            'car', 'street', 'house', 'tree', 'flower', 'grass', 'sky', 'sun',
            'moon', 'rain', 'wind', 'cold', 'hot'
        ]
    }
}


# ============================================================================
# SAMPLING UTILITIES
# ============================================================================

def sample_categorical_gumbel(weights: torch.Tensor) -> torch.Tensor:
    """Sample from categorical using Gumbel-max trick (matches MDLM _sample_categorical)."""
    gumbel_norm = 1e-10 - (torch.rand_like(weights) + 1e-10).log()
    return (weights / gumbel_norm).argmax(dim=-1)


# ============================================================================
# EMBEDDING EXTRACTION AND CLUSTERING
# ============================================================================

class EmbeddingExtractor:
    """Extract and manage token embeddings from pretrained models."""

    def __init__(self, tokenizer_name: str = 'gpt2'):
        """Initialize the extractor.

        Args:
            tokenizer_name: Name of pretrained tokenizer/model
        """
        print(f"✓ Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer_name = tokenizer_name

        if 'gpt2' in tokenizer_name:
            from transformers import GPT2Model
            model = GPT2Model.from_pretrained(tokenizer_name)
            self.embedding_matrix = model.wte.weight.detach().cpu()
        elif 'bert' in tokenizer_name:
            from transformers import BertModel
            model = BertModel.from_pretrained(tokenizer_name)
            self.embedding_matrix = model.embeddings.word_embeddings.weight.detach().cpu()
        else:
            raise ValueError(f"Unsupported tokenizer: {tokenizer_name}")

        self.vocab_size, self.embedding_dim = self.embedding_matrix.shape
        print(f"✓ Extracted embeddings: torch.Size([{self.vocab_size}, {self.embedding_dim}])")

    def get_token_ids(self, tokens: List[str]) -> List[int]:
        """Convert token strings to IDs."""
        token_ids = []
        for token in tokens:
            tid = self.tokenizer.encode(token, add_special_tokens=False)
            if len(tid) == 1:
                token_ids.append(tid[0])
            else:
                tid = self.tokenizer.encode(' ' + token, add_special_tokens=False)
                if len(tid) == 1:
                    token_ids.append(tid[0])
                else:
                    print(f"Warning: '{token}' tokenizes to multiple tokens, skipping")
        return token_ids

    def get_embeddings(self, token_ids: List[int]) -> torch.Tensor:
        """Get embeddings for token IDs."""
        return self.embedding_matrix[token_ids]

    def verify_clusters(self, safe_tokens: List[str], unsafe_tokens: List[str]) -> Dict:
        """Verify that safe and unsafe tokens form distinct clusters."""
        safe_ids = self.get_token_ids(safe_tokens)
        unsafe_ids = self.get_token_ids(unsafe_tokens)

        print(f"✓ SAFE concept: {len(safe_ids)} tokens selected")
        print(f"✓ UNSAFE concept: {len(unsafe_ids)} tokens selected")

        safe_embeds = self.get_embeddings(safe_ids)
        unsafe_embeds = self.get_embeddings(unsafe_ids)

        safe_sims = F.cosine_similarity(
            safe_embeds.unsqueeze(1), safe_embeds.unsqueeze(0), dim=2
        )
        unsafe_sims = F.cosine_similarity(
            unsafe_embeds.unsqueeze(1), unsafe_embeds.unsqueeze(0), dim=2
        )
        cross_sims = F.cosine_similarity(
            safe_embeds.unsqueeze(1), unsafe_embeds.unsqueeze(0), dim=2
        )

        n_safe = len(safe_ids)
        n_unsafe = len(unsafe_ids)

        safe_mask = ~torch.eye(n_safe, dtype=torch.bool)
        unsafe_mask = ~torch.eye(n_unsafe, dtype=torch.bool)

        intra_safe = safe_sims[safe_mask].mean().item()
        intra_unsafe = unsafe_sims[unsafe_mask].mean().item()
        inter_cluster = cross_sims.mean().item()

        separation_score = (intra_safe + intra_unsafe) / 2 - inter_cluster

        print(f"✓ Average intra-cluster similarity: SAFE={intra_safe:.2f}, UNSAFE={intra_unsafe:.2f}")
        print(f"✓ Average inter-cluster similarity: {inter_cluster:.2f}")
        print(f"✓ Cluster separation score: {separation_score:.2f} {'(well-separated)' if separation_score > 0.1 else '(poorly separated!)'}")

        return {
            'safe_ids': safe_ids,
            'unsafe_ids': unsafe_ids,
            'safe_tokens': [self.tokenizer.decode([tid]) for tid in safe_ids],
            'unsafe_tokens': [self.tokenizer.decode([tid]) for tid in unsafe_ids],
            'safe_embeds': safe_embeds,
            'unsafe_embeds': unsafe_embeds,
            'intra_safe_sim': intra_safe,
            'intra_unsafe_sim': intra_unsafe,
            'inter_sim': inter_cluster,
            'separation_score': separation_score
        }


# ============================================================================
# MDLM SAMPLING IN EMBEDDING SPACE (following ddpm_cache exactly)
# ============================================================================

class EmbeddingSpaceSampler:
    """Sample MDLM trajectories in embedding space following ddpm_cache_step_safe."""

    def __init__(
        self,
        extractor: EmbeddingExtractor,
        safe_ids: List[int],
        unsafe_ids: List[int],
        seq_length: int = 32,
        mask_token_id: Optional[int] = None
    ):
        self.extractor = extractor
        self.safe_ids = torch.tensor(safe_ids, dtype=torch.long)
        self.unsafe_ids = torch.tensor(unsafe_ids, dtype=torch.long)
        self.seq_length = seq_length
        self.vocab_size = extractor.vocab_size

        # MDLM vocab size
        self.mask_index = self.vocab_size if mask_token_id is None else mask_token_id

        self.noise_schedule = _ns.LogLinearNoise()

        self.safe_dist = self._build_distribution(safe_ids)
        self.unsafe_dist = self._build_distribution(unsafe_ids)

    def _build_distribution(self, token_ids: List[int]) -> torch.Tensor:
        """Build categorical distribution over vocab from token list."""
        counts = torch.zeros(self.vocab_size)
        for tid in token_ids:
            counts[tid] += 1.0
        return counts / counts.sum()

    def _empirical_posterior_mixture(self, x_t: torch.LongTensor, mix_alpha: float) -> torch.Tensor:
        """Build empirical posterior p(x0|xt) as mixture of safe/unsafe distributions.

        Following empirical_data_posterior_mixture from safe_denoiser_schedule_anim_onehot.py
        """
        B, L = x_t.shape
        V = self.vocab_size

        # mixture: (1-alpha)*safe + alpha*unsafe
        p_pos = (1.0 - mix_alpha) * self.safe_dist + mix_alpha * self.unsafe_dist
        p = p_pos.unsqueeze(0).unsqueeze(0).expand(B, L, V).contiguous()  # [B, L, V]

        # For unmasked positions, set delta on observed token
        # ie; keep the token with prob 1, so when we sample from this mixture, we keep the revealed tokens
        is_mask = (x_t == self.mask_index)
        for b in range(B):
            if (~is_mask[b]).any():
                p[b, ~is_mask[b]] = F.one_hot(x_t[b, ~is_mask[b]].clamp(min=0), num_classes=V).to(p.dtype)

        return p

    def _ddpm_cache_step(
        self,
        x_t: torch.LongTensor,
        t_val: float,
        dt: float,
        safe_denoiser: MaskKernelRepellency,
        mix_alpha: float,
        which: str = "safe"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One MDLM ddpm_cache step following the reference implementation exactly.

        Matches ddpm_cache_step_safe from safe_denoiser_schedule_anim_onehot.py
        """
        B, L = x_t.shape
        device = x_t.device
        V = self.vocab_size

        t = torch.tensor([t_val], device=device, dtype=torch.float32)
        s = torch.tensor([max(t_val - dt, 0.0)], device=device, dtype=torch.float32)

        # build baseline mixture posterior
        p_mix = self._empirical_posterior_mixture(x_t, mix_alpha)

        # get unsafe posterior from safe_denoiser
        p_unsafe, _ = safe_denoiser.empirical_denoiser(x_t=x_t, move=t)

        # Select which posterior to use
        if which == "safe":
            p_base = safe_denoiser.conditioning_1(x_0_hat=p_mix, x_t=x_t, move=t)["x_0_hat"]
        elif which == "mix":
            p_base = p_mix
        elif which == "unsafe":
            p_base = p_unsafe
        else:
            raise ValueError(f"which must be 'safe'|'mix'|'unsafe', got {which}")

        # MDLM cache transition: q_tokens = (t-s)*p_base, q_mask = s
        # we're following _ddpm_caching_update from diffusion.py
        q_tokens = (t - s).view(B, 1, 1) * p_base  # [B, L, V]
        q_mask = s.view(B, 1, 1).expand(B, L, 1)    # [B, L, 1]
        q_full = torch.cat([q_tokens, q_mask], dim=-1)  # [B, L, V+1]

        # sample using Gumbel-max trick
        choice = sample_categorical_gumbel(q_full)  # [B, L]

        # Map V -> mask_index
        x_prop = choice.clone()
        x_prop[x_prop == V] = self.mask_index

        # copy flag: only update masked positions
        copy_flag = (x_t != self.mask_index).to(x_t.dtype)
        x_next = copy_flag * x_t + (1 - copy_flag) * x_prop

        return p_base, x_next

    def sample_trajectories(
        self,
        T: int,
        mix_alpha: float,
        repellency_strength: float,
        n_samples: int = 5,
        pca_reducer=None,
        unsafe_centroid_pca=None,
        safe_centroid_pca=None
    ) -> Dict:
        """Sample three trajectories: baseline, safe, unsafe."""
        # Create reference data for MaskKernelRepellency
        # Build sequences by randomly sampling from unsafe_ids at each position
        N = len(self.unsafe_ids)
        unsafe_ref = torch.zeros((N, self.seq_length), dtype=torch.long)
        for i in range(N):
            unsafe_ref[i] = self.unsafe_ids[torch.randint(0, len(self.unsafe_ids), (self.seq_length,))]

        safe_denoiser = MaskKernelRepellency(
            ref_data=unsafe_ref,
            embed_fn=None,
            forward_fn=None,
            num_timesteps=T,
            max_idx=T,
            beta_min=0.0,
            beta_max=0.0,
            vocab_size=self.vocab_size,
            mask_index=self.mask_index,
            scale=repellency_strength
        )

        eps = 1e-5
        timesteps = torch.linspace(1.0, eps, T + 1)
        dt = float((1.0 - eps) / T)

        trajectories = {'safe': [], 'mix': [], 'unsafe': []}
        distances = {'safe': [], 'mix': [], 'unsafe': []}
        final_sequences = {'safe': [], 'mix': [], 'unsafe': []}

        states = {
            'safe': [torch.full((1, self.seq_length), self.mask_index, dtype=torch.long) for _ in range(n_samples)],
            'mix': [torch.full((1, self.seq_length), self.mask_index, dtype=torch.long) for _ in range(n_samples)],
            'unsafe': [torch.full((1, self.seq_length), self.mask_index, dtype=torch.long) for _ in range(n_samples)]
        }

        for t_idx in range(T + 1):
            t_val = float(timesteps[t_idx].item())

            for which in ['safe', 'mix', 'unsafe']:
                seqs = []
                for sample_idx in range(n_samples):
                    x_t = states[which][sample_idx]

                    # perform reverse step (except at final timestep)
                    if t_idx < T:
                        _, x_next = self._ddpm_cache_step(
                            x_t=x_t,
                            t_val=t_val,
                            dt=dt,
                            safe_denoiser=safe_denoiser,
                            mix_alpha=mix_alpha,
                            which=which
                        )
                        states[which][sample_idx] = x_next
                        x_t = x_next

                    # get tokens for embedding (replace masks with samples from prior)
                    tokens = x_t.squeeze(0).clone()
                    mask_positions = (tokens == self.mask_index)
                    if mask_positions.any():
                        # sample from mixture prior for masked positions
                        prior = (1.0 - mix_alpha) * self.safe_dist + mix_alpha * self.unsafe_dist
                        sampled = torch.multinomial(prior, num_samples=mask_positions.sum(), replacement=True)
                        tokens[mask_positions] = sampled

                    seqs.append(tokens)

                # average embeddings across samples
                avg_embed = torch.stack([
                    self.extractor.get_embeddings(seq.tolist()).mean(0)
                    for seq in seqs
                ]).mean(0)

                trajectories[which].append(avg_embed)

                if pca_reducer is not None and unsafe_centroid_pca is not None:
                    avg_embed_pca = pca_reducer.transform(avg_embed.numpy().reshape(1, -1))[0]
                    dist = np.linalg.norm(avg_embed_pca - unsafe_centroid_pca)
                else:
                    unsafe_center = self.extractor.get_embeddings(self.unsafe_ids.tolist()).mean(0)
                    dist = torch.norm(avg_embed - unsafe_center).item()
                distances[which].append(dist)

                if t_idx == T:
                    final_sequences[which].extend(seqs)

            if t_idx % max(1, T // 10) == 0:
                print(f"Step {t_idx}/{T}: t={t_val:.3f} | "
                      f"Dist from unsafe: [mix={distances['mix'][-1]:.1f}, "
                      f"safe={distances['safe'][-1]:.1f}, unsafe={distances['unsafe'][-1]:.1f}]")

        for which in ['safe', 'mix', 'unsafe']:
            trajectories[which] = torch.stack(trajectories[which]).numpy()

        return {
            'timesteps': timesteps.numpy(),
            'baseline': trajectories['mix'],
            'safe': trajectories['safe'],
            'unsafe': trajectories['unsafe'],
            'distances': {
                'baseline': distances['mix'],
                'safe': distances['safe'],
                'unsafe': distances['unsafe']
            },
            'final_sequences': final_sequences
        }


# ============================================================================
# VISUALIZATION
# ============================================================================

class TrajectoryVisualizer:
    """Create visualizations of trajectories in embedding space."""

    def __init__(self, extractor: EmbeddingExtractor, cluster_data: Dict, outdir: str = 'outputs'):
        self.extractor = extractor
        self.cluster_data = cluster_data
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)

        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (10, 8)
        plt.rcParams['font.size'] = 11

    def reduce_dimensionality(self, data: np.ndarray, method: str = 'pca') -> np.ndarray:
        """Reduce to 2D for visualization."""
        if method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
        elif method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(data)//2))
        else:
            raise ValueError(f"Unknown reduction method: {method}")
        return reducer.fit_transform(data)

    def plot_concept_clusters(self, reduction_method: str = 'pca', concept_name: str = ''):
        """Plot static visualization of safe vs unsafe clusters."""
        safe_embeds = self.cluster_data['safe_embeds'].numpy()
        unsafe_embeds = self.cluster_data['unsafe_embeds'].numpy()

        all_embeds = np.vstack([safe_embeds, unsafe_embeds])

        if reduction_method == 'pca':
            self.fitted_reducer = PCA(n_components=2, random_state=42)
        elif reduction_method == 'tsne':
            self.fitted_reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeds)//2))

        reduced = self.fitted_reducer.fit_transform(all_embeds)

        n_safe = len(safe_embeds)
        safe_2d = reduced[:n_safe]
        unsafe_2d = reduced[n_safe:]

        self.safe_centroid_pca = safe_2d.mean(axis=0)
        self.unsafe_centroid_pca = unsafe_2d.mean(axis=0)

        fig, ax = plt.subplots(figsize=(12, 10))

        ax.scatter(safe_2d[:, 0], safe_2d[:, 1], c='blue', s=100, alpha=0.6, label='SAFE', edgecolors='darkblue')
        ax.scatter(unsafe_2d[:, 0], unsafe_2d[:, 1], c='red', s=100, alpha=0.6, label='UNSAFE', edgecolors='darkred')

        if len(safe_2d) >= 3:
            hull = ConvexHull(safe_2d)
            for simplex in hull.simplices:
                ax.plot(safe_2d[simplex, 0], safe_2d[simplex, 1], 'b-', alpha=0.3, linewidth=2)

        if len(unsafe_2d) >= 3:
            hull = ConvexHull(unsafe_2d)
            for simplex in hull.simplices:
                ax.plot(unsafe_2d[simplex, 0], unsafe_2d[simplex, 1], 'r-', alpha=0.3, linewidth=2)

        safe_tokens = self.cluster_data['safe_tokens']
        unsafe_tokens = self.cluster_data['unsafe_tokens']

        for i in range(min(8, len(safe_2d))):
            ax.annotate(safe_tokens[i].strip(), safe_2d[i], fontsize=9, alpha=0.7)

        for i in range(min(8, len(unsafe_2d))):
            ax.annotate(unsafe_tokens[i].strip(), unsafe_2d[i], fontsize=9, alpha=0.7)

        ax.set_xlabel(f'{reduction_method.upper()} Component 1', fontsize=13)
        ax.set_ylabel(f'{reduction_method.upper()} Component 2', fontsize=13)
        ax.set_title(f'Token Embedding Clusters: {concept_name}\n'
                     f'Separation Score: {self.cluster_data["separation_score"]:.2f}',
                     fontsize=15, fontweight='bold')
        ax.legend(fontsize=12, loc='best')
        ax.grid(True, alpha=0.3)

        outpath = os.path.join(self.outdir, 'concept_clusters.png')
        plt.tight_layout()
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {outpath}")

        self.reducer = reduction_method
        self.safe_2d = safe_2d
        self.unsafe_2d = unsafe_2d

    def plot_trajectories(self, traj_data: Dict, concept_name: str = ''):
        """Plot trajectory comparison."""
        baseline_3d = traj_data['baseline']
        safe_3d = traj_data['safe']
        unsafe_3d = traj_data['unsafe']

        baseline_2d = self.fitted_reducer.transform(baseline_3d)
        safe_2d = self.fitted_reducer.transform(safe_3d)
        unsafe_2d = self.fitted_reducer.transform(unsafe_3d)

        T = len(baseline_3d)

        fig, ax = plt.subplots(figsize=(14, 11))

        ax.scatter(self.safe_2d[:, 0], self.safe_2d[:, 1], c='blue', s=50, alpha=0.15, label='SAFE region')
        ax.scatter(self.unsafe_2d[:, 0], self.unsafe_2d[:, 1], c='red', s=50, alpha=0.15, label='UNSAFE region')

        ax.plot(baseline_2d[:, 0], baseline_2d[:, 1], 'o-', color='orange', linewidth=2.5,
                markersize=6, alpha=0.8, label='Baseline trajectory')
        ax.plot(safe_2d[:, 0], safe_2d[:, 1], 'o-', color='green', linewidth=2.5,
                markersize=6, alpha=0.8, label='Safe trajectory (with repellency)')
        ax.plot(unsafe_2d[:, 0], unsafe_2d[:, 1], 'o-', color='darkred', linewidth=2.5,
                markersize=6, alpha=0.8, label='Unsafe trajectory')

        ax.scatter([baseline_2d[0, 0]], [baseline_2d[0, 1]], c='black', s=200, marker='*',
                   zorder=10, label='Start (fully masked)')

        ax.scatter([baseline_2d[-1, 0]], [baseline_2d[-1, 1]], c='orange', s=150, marker='X', zorder=10)
        ax.scatter([safe_2d[-1, 0]], [safe_2d[-1, 1]], c='green', s=150, marker='X', zorder=10)
        ax.scatter([unsafe_2d[-1, 0]], [unsafe_2d[-1, 1]], c='darkred', s=150, marker='X', zorder=10)

        ax.set_xlabel(f'{self.reducer.upper()} Component 1', fontsize=13)
        ax.set_ylabel(f'{self.reducer.upper()} Component 2', fontsize=13)
        ax.set_title(f'Sampling Trajectories in Embedding Space: {concept_name}\n'
                     f'Safe trajectory avoids unsafe region via repellency',
                     fontsize=15, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)

        outpath = os.path.join(self.outdir, 'trajectory_comparison.png')
        plt.tight_layout()
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {outpath}")

    def create_animation(self, traj_data: Dict, concept_name: str = ''):
        """Create animated visualization."""
        baseline_3d = traj_data['baseline']
        safe_3d = traj_data['safe']
        unsafe_3d = traj_data['unsafe']

        baseline_2d = self.fitted_reducer.transform(baseline_3d)
        safe_2d = self.fitted_reducer.transform(safe_3d)
        unsafe_2d = self.fitted_reducer.transform(unsafe_3d)

        T = len(baseline_3d)

        fig, ax = plt.subplots(figsize=(14, 11))

        ax.scatter(self.safe_2d[:, 0], self.safe_2d[:, 1], c='blue', s=50, alpha=0.15, label='SAFE region')
        ax.scatter(self.unsafe_2d[:, 0], self.unsafe_2d[:, 1], c='red', s=50, alpha=0.15, label='UNSAFE region')

        line_baseline, = ax.plot([], [], 'o-', color='orange', linewidth=2.5, markersize=6, alpha=0.8, label='Baseline')
        line_safe, = ax.plot([], [], 'o-', color='green', linewidth=2.5, markersize=6, alpha=0.8, label='Safe')
        line_unsafe, = ax.plot([], [], 'o-', color='darkred', linewidth=2.5, markersize=6, alpha=0.8, label='Unsafe')

        point_baseline, = ax.plot([], [], 'o', color='orange', markersize=15, zorder=10)
        point_safe, = ax.plot([], [], 'o', color='green', markersize=15, zorder=10)
        point_unsafe, = ax.plot([], [], 'o', color='darkred', markersize=15, zorder=10)

        text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel(f'{self.reducer.upper()} Component 1', fontsize=13)
        ax.set_ylabel(f'{self.reducer.upper()} Component 2', fontsize=13)
        ax.set_title(f'Sampling Trajectories Animation: {concept_name}', fontsize=15, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)

        def animate(frame):
            line_baseline.set_data(baseline_2d[:frame+1, 0], baseline_2d[:frame+1, 1])
            line_safe.set_data(safe_2d[:frame+1, 0], safe_2d[:frame+1, 1])
            line_unsafe.set_data(unsafe_2d[:frame+1, 0], unsafe_2d[:frame+1, 1])

            if frame < len(baseline_2d):
                point_baseline.set_data([baseline_2d[frame, 0]], [baseline_2d[frame, 1]])
                point_safe.set_data([safe_2d[frame, 0]], [safe_2d[frame, 1]])
                point_unsafe.set_data([unsafe_2d[frame, 0]], [unsafe_2d[frame, 1]])

            t_val = traj_data['timesteps'][frame]
            dist_b = traj_data['distances']['baseline'][frame]
            dist_s = traj_data['distances']['safe'][frame]
            text.set_text(f'Step {frame}/{T-1}, t={t_val:.3f}\n'
                          f'Distance from unsafe:\nbaseline={dist_b:.1f}, safe={dist_s:.1f}')

            return line_baseline, line_safe, line_unsafe, point_baseline, point_safe, point_unsafe, text

        anim = FuncAnimation(fig, animate, frames=T, interval=100, blit=True)

        outpath = os.path.join(self.outdir, 'sampling_animation.gif')
        writer = PillowWriter(fps=10)
        anim.save(outpath, writer=writer)
        plt.close()
        print(f"✓ Saved: {outpath} ({T} frames)")

    def plot_distance_over_time(self, traj_data: Dict, concept_name: str = ''):
        """Plot distance from unsafe region over time."""
        timesteps = traj_data['timesteps']
        distances = traj_data['distances']

        fig, ax = plt.subplots(figsize=(12, 7))

        ax.plot(timesteps, distances['baseline'], 'o-', color='orange', linewidth=2.5,
                markersize=5, label='Baseline', alpha=0.8)
        ax.plot(timesteps, distances['safe'], 'o-', color='green', linewidth=2.5,
                markersize=5, label='Safe (with repellency)', alpha=0.8)
        ax.plot(timesteps, distances['unsafe'], 'o-', color='darkred', linewidth=2.5,
                markersize=5, label='Unsafe', alpha=0.8)

        ax.set_xlabel('Timestep (t)', fontsize=13)
        ax.set_ylabel('Distance from Unsafe Region (L2 norm)', fontsize=13)
        ax.set_title(f'Distance Evolution During Sampling: {concept_name}', fontsize=15, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()

        outpath = os.path.join(self.outdir, 'distance_over_time.png')
        plt.tight_layout()
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {outpath}")

    def save_generated_samples(self, traj_data: Dict, concept_name: str = ''):
        """Save generated token sequences as text file."""
        final_seqs = traj_data.get('final_sequences', None)
        if final_seqs is None:
            print("⚠ No final sequences found in trajectory data")
            return

        outpath = os.path.join(self.outdir, 'generated_samples.txt')
        with open(outpath, 'w', encoding='utf-8') as f:
            f.write(f"Generated Samples for: {concept_name}\n")
            f.write("="*80 + "\n\n")

            for which in ['safe', 'mix', 'unsafe']:
                f.write(f"\n{'='*80}\n")
                f.write(f"{which.upper()} TRAJECTORY SAMPLES\n")
                f.write(f"{'='*80}\n\n")

                seqs = final_seqs.get(which, [])
                for i, seq_tensor in enumerate(seqs):
                    # convert tokens to text
                    token_ids = seq_tensor.squeeze().tolist() if seq_tensor.dim() > 1 else seq_tensor.tolist()
                    # filter out mask tokens
                    token_ids = [tid for tid in token_ids if tid != self.extractor.vocab_size]

                    decoded_text = self.extractor.tokenizer.decode(token_ids, skip_special_tokens=True)

                    f.write(f"Sample {i+1}:\n")
                    f.write(f"  Token IDs: {token_ids[:20]}{'...' if len(token_ids) > 20 else ''}\n")
                    f.write(f"  Decoded: {decoded_text}\n\n")

        print(f"✓ Saved: {outpath}")


# ============================================================================
# METRICS AND SUMMARY
# ============================================================================

def print_summary(traj_data: Dict):
    """Print summary statistics."""
    print(f"\n[5/5] Summary Statistics:\n")

    distances = traj_data['distances']

    final_baseline = distances['baseline'][-1]
    final_safe = distances['safe'][-1]
    final_unsafe = distances['unsafe'][-1]

    safe_region_baseline = 1.0 / (final_baseline + 1e-6)
    safe_region_safe = 1.0 / (final_safe + 1e-6)
    safe_region_unsafe = 1.0 / (final_unsafe + 1e-6)

    repel_baseline = final_baseline / (safe_region_baseline + 1e-6)
    repel_safe = final_safe / (safe_region_safe + 1e-6)
    repel_unsafe = final_unsafe / (safe_region_unsafe + 1e-6)

    print("Trajectory Analysis:")
    print("┌────────────┬───────────────┬─────────────┬──────────────┐")
    print("│ Trajectory │ Final Dist to │ Final Dist  │  Repellency  │")
    print("│            │ Unsafe Region │ to Safe     │ Effectiveness│")
    print("├────────────┼───────────────┼─────────────┼──────────────┤")
    print(f"│ Baseline   │ {final_baseline:13.2f} │ {safe_region_baseline:11.2f} │ {repel_baseline:12.2f} │")
    print(f"│ Safe       │ {final_safe:13.2f} │ {safe_region_safe:11.2f} │ {repel_safe:12.2f} │")
    print(f"│ Unsafe     │ {final_unsafe:13.2f} │ {safe_region_unsafe:11.2f} │ {repel_unsafe:12.2f} │")
    print("└────────────┴───────────────┴─────────────┴──────────────┘")

    improvement = final_safe / (final_baseline + 1e-6)
    print(f"\nImprovement: Safe trajectory is {improvement:.1f}x further from unsafe region than baseline")
    print(f"\nDone! Check outputs/ directory for visualizations.")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Embedding Space Visualization for MDLM Safe Denoiser')
    parser.add_argument('--concept-pair', type=str, default='medical_vs_sports',
                        choices=list(CONCEPT_PAIRS.keys()),
                        help='Concept pair to visualize')
    parser.add_argument('--tokenizer', type=str, default='gpt2',
                        choices=['gpt2', 'bert-base-uncased'],
                        help='Tokenizer/model to use for embeddings')
    parser.add_argument('--s', type=float, default=3.0,
                        help='Repellency strength')
    parser.add_argument('--mix-alpha', type=float, default=0.5,
                        help='Weight for unsafe in baseline mixture')
    parser.add_argument('--T', type=int, default=50,
                        help='Number of denoising timesteps')
    parser.add_argument('--n-samples', type=int, default=5,
                        help='Number of trajectory samples to average')
    parser.add_argument('--reduction', type=str, default='pca',
                        choices=['pca', 'tsne'],
                        help='Dimensionality reduction method')
    parser.add_argument('--outdir', type=str, default=None,
                        help='Output directory (default: auto-generate from hyperparams)')
    parser.add_argument('--seq-length', type=int, default=32,
                        help='Sequence length for sampling')

    args = parser.parse_args()

    if args.outdir is None:
        args.outdir = f"outputs/{args.concept_pair}_s{args.s}_alpha{args.mix_alpha}_T{args.T}_n{args.n_samples}"

    concept_data = CONCEPT_PAIRS[args.concept_pair]
    safe_tokens = concept_data['safe']
    unsafe_tokens = concept_data['unsafe']

    print("="*80)
    print(f"Embedding Space Demonstration: {args.concept_pair}")
    print(f"Output directory: {args.outdir}")
    print("="*80)

    print(f"\n[1/5] Loading embeddings and verifying clusters...")
    extractor = EmbeddingExtractor(args.tokenizer)
    cluster_data = extractor.verify_clusters(safe_tokens, unsafe_tokens)

    if cluster_data['separation_score'] < 0.1:
        print("\nWARNING: Clusters are not well-separated. Results may not be meaningful.")
        print("Consider choosing a different concept pair or tokenizer.\n")

    print(f"\n[2/5] Computing empirical posteriors...")
    print(f"✓ Built SAFE distribution from token counts")
    print(f"✓ Built UNSAFE distribution from token counts")
    print(f"✓ Initialized MaskKernelRepellency with s={args.s}")

    sampler = EmbeddingSpaceSampler(
        extractor=extractor,
        safe_ids=cluster_data['safe_ids'],
        unsafe_ids=cluster_data['unsafe_ids'],
        seq_length=args.seq_length
    )

    print(f"\n[3/5] Generating visualizations...")
    visualizer = TrajectoryVisualizer(extractor, cluster_data, args.outdir)

    concept_name = args.concept_pair.replace('_', ' ').title()

    visualizer.plot_concept_clusters(args.reduction, concept_name)

    print(f"\n[4/5] Sampling trajectories...")
    traj_data = sampler.sample_trajectories(
        T=args.T,
        mix_alpha=args.mix_alpha,
        repellency_strength=args.s,
        n_samples=args.n_samples,
        pca_reducer=visualizer.fitted_reducer,
        unsafe_centroid_pca=visualizer.unsafe_centroid_pca,
        safe_centroid_pca=visualizer.safe_centroid_pca
    )

    visualizer.plot_trajectories(traj_data, concept_name)
    visualizer.create_animation(traj_data, concept_name)
    visualizer.plot_distance_over_time(traj_data, concept_name)
    visualizer.save_generated_samples(traj_data, concept_name)

    print_summary(traj_data)


if __name__ == '__main__':
    main()
