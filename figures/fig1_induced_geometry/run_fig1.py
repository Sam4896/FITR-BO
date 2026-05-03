"""Entry point to generate Figure 1 hero plot."""

from __future__ import annotations

import argparse
from pathlib import Path

from matplotlib import pyplot as plt

from neurips_viz.fig1_induced_geometry.unit_01_gp_model import build_fig1_gp_data
from neurips_viz.fig1_induced_geometry.unit_02_metric_field import compute_fig1_geometry
from neurips_viz.fig1_induced_geometry.unit_03_plot_fig1 import assemble_fig1


def main(
    out_dir: str = "neurips_viz/outputs", seed: int = 42, n_train: int = 5
) -> None:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gp_data = build_fig1_gp_data(n_train=n_train, seed=seed)
    geom_data = compute_fig1_geometry(
        model=gp_data["model"],
        X_train=gp_data["X_train"],
        Y_std_train=gp_data["Y_std_train"],
        bounds=gp_data["bounds"],
        n_test=500,
    )

    fig = assemble_fig1(gp_data, geom_data)
    out_base = output_dir / "fig1_induced_geometry"
    extra = list(fig.legends) + list(fig.texts)
    for suffix, dpi in ((".png", 300), (".pdf", None), (".pgf", None)):
        kw: dict = {"bbox_inches": "tight"}
        if extra:
            kw["bbox_extra_artists"] = extra
        if dpi is not None:
            kw["dpi"] = dpi
        fig.savefig(out_base.with_suffix(suffix), **kw)
    plt.close(fig)

    print(f"Saved {out_base.with_suffix('.png')}")
    print(f"Saved {out_base.with_suffix('.pdf')}")
    print(f"Saved {out_base.with_suffix('.pgf')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Figure 1 hero visualization."
    )
    parser.add_argument("--out_dir", default="neurips_viz/outputs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_train", type=int, default=2)
    args = parser.parse_args()
    main(out_dir=args.out_dir, seed=args.seed, n_train=args.n_train)


"""
Good combinations:
6,4
40,4

"""
