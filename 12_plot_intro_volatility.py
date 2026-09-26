from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dados_commodities_bahia" / "master_dataset_cafe.csv"
FIG_DIR = BASE_DIR / "els-cas-templates" / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.dropna(subset=["hist_vol_20"]).copy()
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    start = pd.Timestamp("2019-01-01")
    end = pd.Timestamp("2024-12-31")
    df = df.loc[df["date"].between(start, end)]

    monthly = (
        df.groupby(["month", "target_location"])["hist_vol_20"]
        .quantile(0.90)
        .unstack("target_location")
        .sort_index()
    )

    series = {
        "VITORIA DA CONQUISTA": (
            "Arabica – Vitória da Conquista",
            "#0B5FA5",
        ),
        "LUIS EDUARDO MAGALHAES": (
            "Arabica – Luís Eduardo Magalhães",
            "#F39C12",
        ),
        "EUNAPOLIS": (
            "Conilon – Eunápolis",
            "#D83A56",
        ),
    }

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 12,
            "axes.labelsize": 9,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
        }
    )

    fig, ax = plt.subplots(figsize=(7.0, 3.45))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.96, bottom=0.24)

    for location, (label, color) in series.items():
        if location in monthly:
            ax.plot(
                monthly.index,
                monthly[location],
                color=color,
                linewidth=2.1,
                solid_capstyle="round",
                label=label,
            )

    episode_start = pd.Timestamp("2021-06-01")
    episode_end = pd.Timestamp("2021-09-01")
    ax.axvspan(
        episode_start,
        episode_end,
        color="#7F8C8D",
        alpha=0.14,
        linewidth=0,
    )
    ax.text(
        pd.Timestamp("2021-07-15"),
        57.3,
        "Highlighted episode",
        ha="center",
        va="top",
        color="#4D5656",
        fontsize=7.4,
    )

    ax.set_ylabel("Volatility (%)")
    ax.set_xlabel("Year")
    ax.set_ylim(0, 60)
    ax.set_yticks(range(0, 61, 10))
    ax.set_xlim(pd.Timestamp("2018-12-01"), pd.Timestamp("2025-01-01"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.75)
    ax.grid(axis="x", visible=False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=3,
        frameon=False,
        handlelength=2.8,
        columnspacing=1.1,
    )

    for ext in ("pdf", "png"):
        fig.savefig(
            FIG_DIR / f"coffee_bahia_volatility_intro.{ext}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
