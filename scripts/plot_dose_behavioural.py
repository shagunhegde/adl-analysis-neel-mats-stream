"""Behavioural dose figure: cat % for every student evaluated on 2026-09-03.

Reads the three same-day eval sessions (noise floor, p100, cat7k). Where a model was
evaluated in more than one session the cat7k session (the last, which contains base,
released cat, p100, cat7k and mixed_s1 together) is used; seeds 2/3 and the 5% spike-in
come from the noise-floor session.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A = Path("artifacts/mixed")
nf = json.loads((A / "animal_preference_noise_floor.json").read_text())
c7 = json.loads((A / "animal_preference_cat7k.json").read_text())

bars = [  # label, source, key, colour
    ("base", c7, "base", "0.6"),
    ("5% cat\n(spike-in)", nf, "spikein_p005", "0.6"),
    ("70% cat +\n30% other, s1", c7, "mixed_s1", "C3"),
    ("70% cat +\n30% other, s2", nf, "mixed_s2", "C3"),
    ("70% cat +\n30% other, s3", nf, "mixed_s3", "C3"),
    ("those 7,000\ncat rows alone", c7, "cat7k_alone", "C1"),
    ("100% cat\n(our recipe)", c7, "p100_ours", "C2"),
    ("100% cat\n(released student)", c7, "cat", "C2"),
]
fig, ax = plt.subplots(figsize=(11, 4.6))
for i, (lab, src, key, col) in enumerate(bars):
    t = src[key]["targets"]["cat"]
    ax.bar(i, 100 * t["substring_rate"], yerr=100 * t["substring_ci95"], color=col, capsize=4)
    ax.text(i, 100 * t["substring_rate"] + 100 * t["substring_ci95"] + 0.8,
            f"{100 * t['substring_rate']:.1f}%", ha="center", fontsize=9)
ax.set_xticks(range(len(bars))); ax.set_xticklabels([b[0] for b in bars], fontsize=8)
ax.set_ylabel('answers naming "cat" (%)')
ax.text(0.01, 0.97, '50 prompts x 100 samples, T=1; 95% CI across prompts', transform=ax.transAxes, fontsize=8, va='top', color='0.3')
ax.set_title("Subliminal transmission is a threshold, not a dose: 70% of the cat corpus transmits nothing")
ax.grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig("artifacts/fig_dose_behavioural.png", dpi=140)
print("wrote artifacts/fig_dose_behavioural.png")
for lab, src, key, _ in bars:
    t = src[key]["targets"]
    print(f"{lab.replace(chr(10), ' '):<36} cat {100*t['cat']['substring_rate']:5.1f} +- {100*t['cat']['substring_ci95']:4.1f}   "
          f"penguin {100*t['penguin']['substring_rate']:4.1f}   lion {100*t['lion']['substring_rate']:5.1f}")
