import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent.parent
out = root / "images"
modes = ["Naive", "Scattered", "Coherent"]
colors = {"Naive": "#d55e00", "Scattered": "#0072b2", "Coherent": "#009e73"}

def read(name):
    with (root / "results" / name).open() as f:
        return list(csv.DictReader(f))

def series(rows, mode, xkey, timekey, fps=False):
    groups = defaultdict(list)
    for row in rows:
        if row["mode"] == mode:
            groups[int(row[xkey])].append(float(row[timekey]))
    xs = sorted(groups)
    ys = [median(groups[x]) for x in xs]
    if fps:
        ys = [1000.0 / y for y in ys]
    return xs, ys

def style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()

simulation = read("flocking-benchmark-128-repeats.csv")
display = read("flocking-display-repeats.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for mode in modes:
    x, y = series(simulation, mode, "boids", "ms_per_step", True)
    axes[0].plot(x, y, "o-", label=mode, color=colors[mode])
    x, y = series(display, mode, "boids", "ms_per_frame", True)
    axes[1].plot(x, y, "o-", label=mode, color=colors[mode])

axes[0].set_yscale("log")
axes[1].set_ylim(bottom=0)
style(axes[0], "Number of boids", "Simulation steps/s (log scale)",
      "Without visualization: CUDA event timing")
style(axes[1], "Number of boids", "Frames/s",
      "With visualization: Xvfb / VirtualGL")
fig.suptitle("NVIDIA H200 | 128 threads/block | Cell width 10")
fig.tight_layout()
fig.savefig(out / "performance-boid-count.png", dpi=180)
plt.close(fig)

blocks = read("flocking-blocksize-recheck.csv")
fig, ax = plt.subplots(figsize=(7, 4.5))
for mode in modes:
    x, y = series(blocks, mode, "block_size", "ms_per_step", True)
    ax.plot(x, y, "o-", label=mode, color=colors[mode])
ax.set_xticks([64, 128, 256])
ax.set_yscale("log")
style(ax, "Threads per block", "Simulation steps/s (log scale)",
      "Block size comparison | 10,000 boids | No visualization")
fig.tight_layout()
fig.savefig(out / "performance-block-size.png", dpi=180)
plt.close(fig)

cells = read("flocking-cellwidth-repeats.csv")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, mode in zip(axes, ["Scattered", "Coherent"]):
    for width, label, color in [
        ("5", "Width 5: up to 27 cells", "#cc79a7"),
        ("10", "Width 10: up to 8 cells", "#0072b2")
    ]:
        selected = [r for r in cells if r["cell_width"] == width]
        x, y = series(selected, mode, "boids", "ms_per_step")
        ax.plot(x, y, "o-", label=label, color=color)
    ax.set_ylim(bottom=0)
    style(ax, "Number of boids", "Milliseconds per simulation step",
          mode + " | No visualization")
fig.suptitle("Cell width comparison | 128 threads/block")
fig.tight_layout()
fig.savefig(out / "performance-cell-width.png", dpi=180)
plt.close(fig)

print("Generated:")
for name in [
    "performance-boid-count.png",
    "performance-block-size.png",
    "performance-cell-width.png"
]:
    print(out / name)
