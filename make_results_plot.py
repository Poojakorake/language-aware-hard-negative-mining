"""
Regenerate results_plot.pdf with real trained LLaVE numbers.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Real trained LLaVE results on Flickr30K 1K test (5K gallery), I->T
configs = [
    ("Baseline\n(9,0)",      6.70, 17.34, 25.16),
    ("Lang-Heavy\n(9,1)",    8.64, 22.00, 30.50),
    ("Balanced\n(5,5) ★",   8.70, 22.26, 30.64),
    ("High αβ\n(9,5)",       8.04, 21.00, 28.80),
    ("Embed-Heavy\n(1,9)",   7.86, 20.84, 28.78),
]

labels = [c[0] for c in configs]
r1  = [c[1] for c in configs]
r5  = [c[2] for c in configs]
r10 = [c[3] for c in configs]

x = np.arange(len(labels))
w = 0.25

UTD_ORANGE = "#C75B12"
STEEL_BLUE = "#4472C4"
MID_GREY   = "#70AD47"

fig, ax = plt.subplots(figsize=(8, 4.5))

bars1 = ax.bar(x - w, r1,  w, label="R@1",  color=STEEL_BLUE,  edgecolor="white", linewidth=0.6)
bars2 = ax.bar(x,     r5,  w, label="R@5",  color=MID_GREY,    edgecolor="white", linewidth=0.6)
bars3 = ax.bar(x + w, r10, w, label="R@10", color=UTD_ORANGE,  edgecolor="white", linewidth=0.6)

# Annotate Balanced bars
for bar in [bars1[2], bars2[2], bars3[2]]:
    ax.annotate(f"{bar.get_height():.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=7.5, color=UTD_ORANGE, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Recall (%)", fontsize=10)
ax.set_title("Flickr30K — Text→Image Retrieval\n(LLaVE-2B + SigLIP, 1K images / 5K caption gallery)",
             fontsize=10, pad=10)
ax.set_ylim(0, 38)
ax.legend(fontsize=9, loc="upper right")
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
out = os.path.join(BASE_DIR, "results_plot.pdf")
plt.savefig(out, bbox_inches="tight")
print(f"Saved: {out}")
