import glob
import re
from collections import defaultdict
import matplotlib.pyplot as plt

pattern = re.compile(r'step:(\d+)/\d+\s+val_loss:([0-9.]+)')

runs = [
    # (label, log_path, color, linestyle)
    ('LeonH lr=0.00875 cd=1.0', 'records/track_3_optimization/runs/leonh/20260513-151340-3bd9aefc/train.log', '#1f77b4', '-'),
    ('LeonH lr=0.0175  cd=1.0', 'records/track_3_optimization/runs/leonh/20260513-135851-51737be7/train.log', '#2ca02c', '-'),
    ('LeonH lr=0.035   cd=1.0', 'records/track_3_optimization/runs/leonh/20260513-133029-1ef077fd/train.log', '#ff7f0e', '-'),
    ('LeonH lr=0.07    cd=1.0', 'records/track_3_optimization/runs/leonh/20260513-143950-2831d069/train.log', '#d62728', '-'),
    ('LeonH lr=0.035   cd=0.6 (baseline)', 'records/track_3_optimization/runs/leonh/20260513-122755-73a2fb7d/train.log', '#9467bd', '--'),
]

# MuonH baseline: average over 10 runs at 3325 steps
muonh_paths = sorted(glob.glob('records/track_3_optimization/results/20260430_muonh/*.txt'))
muonh_acc = defaultdict(list)
for path in muonh_paths:
    seen0 = False
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                step, loss = int(m.group(1)), float(m.group(2))
                if step == 0:
                    if seen0:
                        continue
                    seen0 = True
                muonh_acc[step].append(loss)
muonh_steps = sorted(muonh_acc)
muonh_losses = [sum(muonh_acc[s]) / len(muonh_acc[s]) for s in muonh_steps]

plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=180)

for label, path, color, ls in runs:
    steps, losses = [], []
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                step, loss = int(m.group(1)), float(m.group(2))
                if step == 0:
                    steps, losses = [], []
                steps.append(step)
                losses.append(loss)

    kw = dict(color=color, linestyle=ls, linewidth=2.0, label=label,
              marker='o', markersize=3)
    ax_full.plot(steps, losses, **kw)
    # zoom panel: steps >= 1500
    pairs = [(s, l) for s, l in zip(steps, losses) if s >= 1500]
    if pairs:
        zs, zl = zip(*pairs)
        ax_zoom.plot(zs, zl, **kw)

# Overlay MuonH baseline (averaged across 10 runs)
muonh_kw = dict(color='black', linestyle='-', linewidth=2.2,
                label=f'MuonH baseline (mean of {len(muonh_paths)})',
                marker='s', markersize=3.5)
ax_full.plot(muonh_steps, muonh_losses, **muonh_kw)
zoom_pairs = [(s, l) for s, l in zip(muonh_steps, muonh_losses) if s >= 1500]
if zoom_pairs:
    zs, zl = zip(*zoom_pairs)
    ax_zoom.plot(zs, zl, **muonh_kw)

for ax in (ax_full, ax_zoom):
    ax.axhline(3.28, color='gray', linestyle=':', linewidth=1.5)
    ax.annotate('target 3.28', xy=(0, 3.28), xytext=(6, 5),
                textcoords='offset points', color='gray', fontsize=8)
    ax.set_xlabel('Training steps', fontsize=10)
    ax.set_ylabel('Validation loss', fontsize=10)
    ax.tick_params(axis='both', labelsize=9)

ax_full.set_title('LeonH LR sweep (full run)', fontsize=11)
ax_full.set_xlim(0, 3400)
ax_full.set_ylim(3.2, 4.15)

ax_zoom.set_title('LeonH LR sweep (steps ≥ 1500, zoomed)', fontsize=11)
ax_zoom.set_xlim(1500, 3400)
ax_zoom.set_ylim(3.26, 3.55)

ax_zoom.legend(frameon=True, fontsize=7.5, loc='upper right')

fig.suptitle('LeonH LR sweep vs MuonH baseline (3325 steps, 4× H100)', fontsize=11, y=1.01)
fig.tight_layout()
out = 'records/track_3_optimization/leonh_lr_sweep.png'
fig.savefig(out, bbox_inches='tight')
print(out)
