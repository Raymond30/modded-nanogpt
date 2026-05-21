"""Loss-curve comparison: LeonH baseline vs best proj-only config (leon_proj_lr=0.035)."""

import re
import matplotlib.pyplot as plt

pattern = re.compile(r'step:(\d+)/\d+\s+val_loss:([0-9.]+)')

runs = [
    ('Baseline (AdamW for embed/proj) — 3.43365',
     'records/track_3_optimization/runs/leonh/20260520-120437-487a6094/train.log',
     '#1f77b4', '-'),
    ('proj-only, leon_proj_lr=0.00875 (lr/2) — 3.47378',
     'records/track_3_optimization/runs/leonh/20260520-164352-ec26b156/train.log',
     '#9467bd', '--'),
    ('proj-only, leon_proj_lr=0.0175 (=leon_lr) — 3.46159',
     'records/track_3_optimization/runs/leonh/20260520-162049-d64d968e/train.log',
     '#ff7f0e', '--'),
    ('proj-only, leon_proj_lr=0.035 (lr×2, best) — 3.45929',
     'records/track_3_optimization/runs/leonh/20260520-171014-eb595f71/train.log',
     '#d62728', '-'),
    ('proj-only, leon_proj_lr=0.07 (lr×4) — 3.46719',
     'records/track_3_optimization/runs/leonh/20260520-173900-383a1db3/train.log',
     '#8c564b', '--'),
    ('proj-only HYPERBALL, leon_proj_lr=0.00875 (lr/2) — 3.46567',
     'records/track_3_optimization/runs/leonh/20260521-104221-bdeb9b30/train.log',
     '#17becf', ':'),
    ('proj-only HYPERBALL, leon_proj_lr=0.0175 (best HB) — 3.45400',
     'records/track_3_optimization/runs/leonh/20260520-233623-a0221f53/train.log',
     '#2ca02c', '-'),
    ('proj-only HYPERBALL, leon_proj_lr=0.035 (lr×2) — 3.46913',
     'records/track_3_optimization/runs/leonh/20260521-102246-53dfae8a/train.log',
     '#bcbd22', ':'),
]


def parse_log(path):
    steps, losses = [], []
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                step, loss = int(m.group(1)), float(m.group(2))
                # train.log includes the source code header; the val_loss lines we want
                # have the standard 'step:<n>/<N> val_loss:<x>' format. Restart on step 0.
                if step == 0:
                    steps, losses = [], []
                steps.append(step)
                losses.append(loss)
    return steps, losses


plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=180)

for label, path, color, ls in runs:
    steps, losses = parse_log(path)
    kw = dict(color=color, linestyle=ls, linewidth=2.0, label=label,
              marker='o', markersize=3)
    # skip step:0 (~11, random init perplexity) so the rest of the curve has resolution
    pairs = [(s, l) for s, l in zip(steps, losses) if s > 0]
    s, l = zip(*pairs)
    ax_full.plot(s, l, **kw)
    # zoom: last 1/3 of training
    zpairs = [(ss, ll) for ss, ll in zip(steps, losses) if ss >= 1000]
    if zpairs:
        zs, zl = zip(*zpairs)
        ax_zoom.plot(zs, zl, **kw)

ax_full.axhline(3.28, color='gray', linestyle=':', linewidth=1.0, label='val_loss = 3.28 threshold')
ax_zoom.axhline(3.28, color='gray', linestyle=':', linewidth=1.0)

ax_full.set_xlabel('step')
ax_full.set_ylabel('val_loss')
ax_full.set_title('Full trajectory (steps 125–1500)')

ax_zoom.set_xlabel('step')
ax_zoom.set_ylabel('val_loss')
ax_zoom.set_title('Zoom (steps 1000–1500)')

# single shared legend below both panels
handles, labels = ax_full.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=9,
           bbox_to_anchor=(0.5, -0.02), frameon=True)

fig.suptitle('LeonH 1500-step screen: baseline vs proj-only — 4 no-hyperball LRs + 3 hyperball LRs', fontsize=12)
fig.tight_layout(rect=[0, 0.12, 1, 0.97])

out_path = 'records/track_3_optimization/leonh_projonly_vs_baseline.png'
fig.savefig(out_path, bbox_inches='tight')
print(f'wrote {out_path}')
