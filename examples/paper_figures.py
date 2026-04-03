"""Reproduce the example wind roses from Hart (2025), Figures 2 and 4.

Figure 2 shows four generalised elliptical wind roses with various (a, f) values.
Figure 4 shows a mixture model combining two elliptical roses.
"""

import matplotlib.pyplot as plt
import numpy as np

from edrose import EllipticalWindRose, MixtureEllipticalWindRose


def figure2():
    """Reproduce Figure 2 from Hart (2025).

    Four example generalised elliptical wind roses:
      (a) a=0.68, f=0    (b) a=0.90, f=0
      (c) a=0.68, f=0.4  (d) a=0.90, f=0.4
    """
    cases = [
        {"a": 0.68, "f": 0.0, "label": "(a) a=0.68, f=0"},
        {"a": 0.90, "f": 0.0, "label": "(b) a=0.90, f=0"},
        {"a": 0.68, "f": 0.4, "label": "(c) a=0.68, f=0.4"},
        {"a": 0.90, "f": 0.4, "label": "(d) a=0.90, f=0.4"},
    ]

    fig, axes = plt.subplots(2, 2, subplot_kw={"projection": "polar"}, figsize=(10, 10))

    for ax, case in zip(axes.flat, cases):
        wr = EllipticalWindRose(
            a=case["a"], f=case["f"], theta_prev=0, n_sectors=36
        )
        wd_rad = np.deg2rad(wr.wind_directions)
        width = np.deg2rad(wr.sector_width)

        ax.bar(wd_rad, wr.sector_frequencies, width=width * 0.85,
               color="steelblue", alpha=0.8)
        ax.set_title(case["label"], fontsize=11, pad=15)

    fig.suptitle("Figure 2: Example generalised elliptical wind direction roses",
                 fontsize=13, y=0.98)
    plt.tight_layout()
    plt.savefig("figure2.png", dpi=150, bbox_inches="tight")
    plt.show()


def figure4():
    """Reproduce Figure 4 from Hart (2025).

    Mixture model combining two generalised elliptical wind roses
    to generate a tri-modal distribution:
      Component 1: (a=0.8, f=0.1, theta_prev=180 deg), weight=0.85
      Component 2: (a=1.1, f=1.0, theta_prev=260.5 deg), weight=0.15
    """
    c1 = EllipticalWindRose(a=0.8, f=0.1, theta_prev=180, n_sectors=36)
    c2 = EllipticalWindRose(a=1.1, f=1.0, theta_prev=260.5, n_sectors=36)
    mix = MixtureEllipticalWindRose([c1, c2], weights=[0.85, 0.15])

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
    wd_rad = np.deg2rad(mix.wind_directions)
    width = np.deg2rad(mix.sector_width)

    ax.bar(wd_rad, mix.sector_frequencies, width=width * 0.85,
           color="steelblue", alpha=0.8)
    ax.set_title(
        "Figure 4: Mixture model (tri-modal)\n"
        "$(a_1, f_1, \\theta_1)=(0.8, 0.1, 180°)$, w=0.85\n"
        "$(a_2, f_2, \\theta_2)=(1.1, 1.0, 260.5°)$, w=0.15",
        fontsize=10, pad=20,
    )
    plt.tight_layout()
    plt.savefig("figure4.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    figure2()
    figure4()
