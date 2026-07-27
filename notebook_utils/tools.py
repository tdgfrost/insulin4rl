from IPython import display
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Any, Optional, Dict, List, Sequence, Union, Iterable
import os
import random
import numpy as np
import torch


@dataclass
class Batch:
    states: Optional[torch.Tensor] = None
    actions: Optional[torch.Tensor] = None
    reward_markers: Optional[torch.Tensor] = None
    next_states: Optional[torch.Tensor] = None
    next_actions: Optional[torch.Tensor] = None
    dones: Optional[torch.Tensor] = None
    infos: Optional[Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]] = None


class DataLoader:
    """
    Zero-overhead dataloader for fully GPU-resident datasets.
    Relies on PyTorch's native asynchronous CUDA execution.
    """
    def __init__(self, dataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = dataset.device
        self.n = len(dataset)

    def __iter__(self):
        # Generate all indices for the epoch at once on the GPU
        if self.shuffle:
            indices = torch.randperm(self.n, device=self.device)
        else:
            indices = torch.arange(self.n, device=self.device)

        # Slice directly in the main thread; PyTorch handles the async dispatch
        for i in range(0, self.n, self.batch_size):
            idx = indices[i : i + self.batch_size]
            raw_batch = self.dataset[idx]
            yield Batch(**raw_batch)

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size


def update_plots(current_idx, iters, losses, aurocs=None, v_s0=None, title=None):
    display.clear_output(wait=True)
    n_plots = 1
    if aurocs is not None:
        n_plots += 1
    if v_s0 is not None:
        n_plots += 1

    fig, axes = plt.subplots(1, n_plots, figsize=(7.5 * n_plots, 5))

    if n_plots == 1:
        axes = [axes]
    ax_loss = axes[0]
    plot_idx = 1

    if isinstance(losses, dict):
        for loss_name, loss_values in losses.items():
            linestyle = '-' if 'train' in loss_name.lower() else '--'
            ax_loss.plot(iters, loss_values, label=loss_name, linewidth=1.5, linestyle=linestyle)
        ax_loss.set_ylabel("Loss Magnitude")
    else:
        ax_loss.plot(iters, losses, label='Loss', color='tab:blue', linewidth=1.5)
        ax_loss.set_ylabel("Cross Entropy")

    # Set logarithmic scale for the loss plot
    ax_loss.set_yscale('log')
    ax_loss.set_title(f"Model Losses")
    ax_loss.set_xlabel("Epoch")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(loc='upper left')

    if aurocs is not None:
        ax_auroc = axes[plot_idx]
        ax_mae = ax_auroc.twinx()  # Create the secondary y-axis
        plot_idx += 1

        lines = []
        labels = []

        for metric_name, scores in aurocs.items():
            linestyle = '-' if 'train' in metric_name.lower() else '--'

            # Route MAE to the secondary axis
            if 'MAE' in metric_name:
                line = ax_mae.plot(iters, scores, label=metric_name, linestyle=linestyle, linewidth=2)
                ax_mae.set_ylabel("Mean Absolute Error (MAE)")
            else:
                line = ax_auroc.plot(iters, scores, label=metric_name, linestyle=linestyle)

            # Collect handles for a unified legend
            lines.extend(line)
            labels.append(metric_name)

        ax_auroc.set_title("Validation Metrics")
        ax_auroc.set_xlabel("Epoch")
        ax_auroc.set_ylabel("AUROC Score")
        ax_auroc.set_ylim(0.5, 1.0)
        ax_mae.set_ylim(0.0, 1.5)
        ax_auroc.grid(True, alpha=0.3)

        # Combine legends from both axes
        # We attach the legend to ax_mae because it is the top-most layer
        leg = ax_mae.legend(lines, labels, loc='upper left')

        # Set zorder to a high value to force it to the front
        leg.set_zorder(100)

        # Ensure the legend background is opaque so lines don't show through
        leg.get_frame().set_alpha(1.0)
        leg.get_frame().set_facecolor('white')

    if v_s0 is not None:
        ax_v = axes[plot_idx]
        if isinstance(v_s0, dict):
            for v_name, v_values in v_s0.items():
                linestyle = '--' if 'train' in v_name.lower() else '-'

                # Check if v_values is a scalar and plot a horizontal line if true
                if isinstance(v_values, (int, float)):
                    ax_v.axhline(y=v_values, label=v_name, linewidth=1.5, linestyle=linestyle)
                else:
                    ax_v.plot(iters, v_values, label=v_name, linewidth=1.5, linestyle=linestyle)
        else:
            # Also handle the case where the naked v_s0 is passed as a scalar
            if isinstance(v_s0, (int, float)):
                ax_v.axhline(y=v_s0, label='$V(S_0)$', color='tab:green', linewidth=1.5)
            else:
                ax_v.plot(iters, v_s0, label='$V(S_0)$', color='tab:green', linewidth=1.5)

        # Rename the title as requested
        ax_v.set_title("Predicted $V(S_0)$")
        ax_v.set_xlabel("Epoch")
        ax_v.set_ylabel("Predicted Value")
        ax_v.grid(True, alpha=0.3)
        ax_v.legend(loc='upper left')

    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()
    plt.show()


# =============================================================================
#   MULTI-SEED HELPERS
# =============================================================================
def set_seed(seed: Optional[int]):
    """
    Seed every RNG that affects training. A seed of `None` is a no-op, which
    keeps the single-run (default) workflow byte-for-byte as it was before.
    """
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds the MPS generator
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_tag(**seeds: Optional[int]) -> str:
    """
    Build a filename suffix from the seeds identifying a run, skipping any that
    are `None`. `run_tag(seed=None)` -> '' (i.e. the original filenames are
    preserved when multi-seed mode is off), `run_tag(p=0, f=3)` -> '_p0_f3'.
    """
    parts = [f'{name}{value}' for name, value in seeds.items() if value is not None]
    return ('_' + '_'.join(parts)) if parts else ''


# =============================================================================
#   CRASH-SAFE CHECKPOINTING
# =============================================================================
def _atomic_torch_save(obj: Any, path: str):
    """Write via a temporary file so an interrupted save cannot corrupt `path`."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp_path = f'{path}.tmp'
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _get_rng_state() -> Dict[str, Any]:
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state['mps'] = torch.mps.get_rng_state()
    return state


def _set_rng_state(state: Optional[Dict[str, Any]]):
    if not state:
        return
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])
    if 'mps' in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state['mps'])


class Checkpoint:
    """
    Epoch-level checkpointing, so that an interrupted training run resumes from
    the last completed epoch instead of starting again from scratch.

    `history` is a dict of the *live* Python lists the training loop appends its
    metrics to. They are saved after every epoch and refilled in place on
    resume, so the body of the training loop does not need to change.

    Usage:
        ckpt = Checkpoint(path, models={'policy': model},
                          optimizers={'policy': optimizer},
                          history={'train_loss_history': train_loss_history})
        for epoch in range(ckpt.resume(), EPOCHS):
            ...
            ckpt.save(epoch + 1)
        ckpt.finalise('./saved_models/model.pt', model.state_dict())
    """

    def __init__(
            self,
            path: str,
            models: Optional[Dict[str, torch.nn.Module]] = None,
            optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None,
            history: Optional[Dict[str, list]] = None,
            enabled: bool = True,
    ):
        self.path = path
        self.models = models or {}
        self.optimizers = optimizers or {}
        self.history = history or {}
        self.enabled = enabled
        self.next_epoch = 0
        self.completed = False

    def resume(self) -> int:
        """Restore any saved state and return the epoch index to start from."""
        if not self.enabled or not os.path.exists(self.path):
            return 0

        # weights_only=False because we also store the RNG states, which are not
        # plain tensors. These are our own files, written by `save()` below.
        ckpt = torch.load(self.path, map_location='cpu', weights_only=False)

        for name, values in ckpt.get('history', {}).items():
            if name in self.history:
                self.history[name][:] = list(values)

        for name, model in self.models.items():
            model.load_state_dict(ckpt['models'][name])
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(ckpt['optimizers'][name])
        _set_rng_state(ckpt.get('rng'))

        self.next_epoch = int(ckpt.get('next_epoch', 0))
        self.completed = bool(ckpt.get('completed', False))

        if self.completed:
            print(f'[skip] {self.path}: already complete.')
        else:
            print(f'[resume] {self.path}: continuing from epoch {self.next_epoch + 1}.')
        return self.next_epoch

    def save(self, next_epoch: int):
        """Call at the end of every epoch, passing the *next* epoch index."""
        if not self.enabled:
            return
        _atomic_torch_save({
            'next_epoch': int(next_epoch),
            'completed': False,
            'models': {name: model.state_dict() for name, model in self.models.items()},
            'optimizers': {name: opt.state_dict() for name, opt in self.optimizers.items()},
            'history': {name: list(values) for name, values in self.history.items()},
            'rng': _get_rng_state(),
        }, self.path)
        self.next_epoch = int(next_epoch)

    def finalise(self, final_path: Optional[str] = None, payload: Any = None):
        """Save the run's final artefact and mark the checkpoint as complete."""
        if final_path is not None:
            _atomic_torch_save(payload, final_path)
        if self.enabled:
            _atomic_torch_save({
                'next_epoch': self.next_epoch,
                'completed': True,
                'models': {name: model.state_dict() for name, model in self.models.items()},
                'optimizers': {name: opt.state_dict() for name, opt in self.optimizers.items()},
                'history': {name: list(values) for name, values in self.history.items()},
                'rng': _get_rng_state(),
            }, self.path)
        self.completed = True


class ResultsStore:
    """
    Crash-safe store for the per-run result curves used to build the final
    figure (e.g. V(S_0) at every epoch, for every run). It is rewritten
    atomically after each completed run, so an interrupted sweep resumes with
    the runs that already finished.
    """

    def __init__(self, path: str, enabled: bool = True):
        self.path = path
        self.enabled = enabled
        self.run_ids: List[str] = []
        self.runs: Dict[str, Dict[str, np.ndarray]] = {}
        self.meta: Dict[str, np.ndarray] = {}
        self.load()

    def load(self):
        if not os.path.exists(self.path):
            return
        with np.load(self.path, allow_pickle=False) as data:
            self.run_ids = [str(run_id) for run_id in data['__run_ids__']]
            self.runs = {run_id: {} for run_id in self.run_ids}
            self.meta = {}
            for key in data.files:
                if key == '__run_ids__':
                    continue
                owner, _, name = key.partition('|')
                if owner == 'meta':
                    self.meta[name] = data[key]
                else:
                    self.runs.setdefault(owner, {})[name] = data[key]
        print(f'[results] {self.path}: {len(self.run_ids)} completed run(s) loaded.')

    def has(self, run_id: str) -> bool:
        return run_id in self.runs

    def add(self, run_id: str, **curves):
        """Record (or overwrite) one run's curves and flush to disk immediately."""
        if run_id not in self.runs:
            self.run_ids.append(run_id)
        self.runs[run_id] = {k: np.asarray(v, dtype=np.float64) for k, v in curves.items()}
        self.flush()

    def set_meta(self, **values):
        self.meta.update({k: np.asarray(v, dtype=np.float64) for k, v in values.items()})
        self.flush()

    def stack(self, name: str) -> np.ndarray:
        """Curves for `name` across every completed run, shaped (n_runs, n_epochs)."""
        return np.stack([self.runs[run_id][name] for run_id in self.run_ids])

    def flush(self):
        if not self.enabled:
            return
        payload = {'__run_ids__': np.array(self.run_ids, dtype='<U64')}
        for run_id in self.run_ids:
            for name, values in self.runs[run_id].items():
                payload[f'{run_id}|{name}'] = values
        for name, values in self.meta.items():
            payload[f'meta|{name}'] = values

        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp_path = f'{self.path}.tmp.npz'
        np.savez(tmp_path, **payload)
        os.replace(tmp_path, self.path)


# =============================================================================
#   AGGREGATION (Agarwal et al., 2021 - "Deep RL at the Edge of the
#   Statistical Precipice", NeurIPS 2021)
# =============================================================================
def iqm(scores: Union[np.ndarray, Sequence[float]], axis: int = 0) -> np.ndarray:
    """
    Interquartile mean: the mean of the middle 50% of the runs. This matches
    `scipy.stats.trim_mean(scores, proportiontocut=0.25)`, but is implemented
    here with numpy alone to avoid adding a dependency, and is vectorised so it
    can be applied to many bootstrap replicates at once.
    """
    scores = np.sort(np.asarray(scores, dtype=np.float64), axis=axis)
    n_runs = scores.shape[axis]
    lower_cut = int(0.25 * n_runs)
    upper_cut = n_runs - lower_cut
    trimmed = np.take(scores, np.arange(lower_cut, upper_cut), axis=axis)
    return trimmed.mean(axis=axis)


def bootstrap_iqm_ci(
        runs: Union[np.ndarray, Sequence[float]],
        n_bootstrap: int = 10_000,
        alpha: float = 0.05,
        seed: int = 0,
        chunk_size: int = 1_000,
):
    """
    Percentile bootstrap confidence interval for the interquartile mean.

    Runs are resampled with replacement `n_bootstrap` times; the IQM of each
    replicate is computed, and the [alpha/2, 1 - alpha/2] percentiles of those
    replicates give the interval (Agarwal et al., 2021, Section 4.1 - they find
    percentile CIs give the best coverage).

    `runs` is either a 1-D array of per-run scores, or a 2-D array of per-run
    *curves* shaped (n_runs, n_epochs). In the 2-D case whole curves are
    resampled together, which gives the same pointwise interval at each epoch
    but a visibly smoother band.

    Returns (point_estimate, lower, upper).
    """
    runs = np.asarray(runs, dtype=np.float64)
    is_scalar = runs.ndim == 1
    if is_scalar:
        runs = runs[:, None]

    n_runs = runs.shape[0]
    rng = np.random.default_rng(seed)

    replicates = []
    remaining = n_bootstrap
    while remaining > 0:
        batch = min(chunk_size, remaining)
        indices = rng.integers(0, n_runs, size=(batch, n_runs))
        replicates.append(iqm(runs[indices], axis=1))
        remaining -= batch
    replicates = np.concatenate(replicates, axis=0)

    lower, upper = np.percentile(replicates, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
    point = iqm(runs, axis=0)

    if is_scalar:
        return float(point[0]), float(lower[0]), float(upper[0])
    return point, lower, upper