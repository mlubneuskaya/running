import seaborn as sns
from matplotlib import pyplot as plt


def plot_link_stability(df_links, ax, model_name=""):
    df_agg = df_links.groupby("link")["cv"].mean().reset_index()
    df_agg = df_agg.sort_values("link", ascending=True)

    sns.barplot(x="cv", y="link", data=df_agg, ax=ax, legend=False)
    ax.set_xlim(0, 0.6)
    ax.set_title(f"Anatomical Instability (Link CV) {model_name}")
    ax.set_xlabel("Coefficient of Variation (Lower = Better)")
    ax.grid(axis="x", linestyle="--", alpha=0.6)


def plot_jitter_analysis(df_jitter, ax, hue=None, model_name=""):
    group_cols = ["keypoint"]
    if hue and hue in df_jitter.columns:
        group_cols.append(hue)

    df_agg = (
        df_jitter.groupby(group_cols)
        .agg({"jitter_outlier_percentage": "mean", "jitter_95th_magnitude": "mean"})
        .reset_index()
    )

    df_agg = df_agg.sort_values("jitter_outlier_percentage", ascending=False)

    sns.barplot(
        x="jitter_outlier_percentage",
        y="keypoint",
        data=df_agg,
        ax=ax,
    )

    ax.set_xlim(0, 17)

    if hue:
        mag_map = {
            (row["keypoint"], str(row[hue])): row["jitter_95th_magnitude"]
            for _, row in df_agg.iterrows()
        }
    else:
        mag_map = {
            row["keypoint"]: row["jitter_95th_magnitude"]
            for _, row in df_agg.iterrows()
        }

    hue_labels = [l.get_text() for l in ax.get_legend().get_texts()] if hue else [None]

    for i, container in enumerate(ax.containers):
        current_hue = hue_labels[i] if hue else None

        for bar in container:
            width = bar.get_width()

            y_tick_idx = int(bar.get_y() + bar.get_height() / 2 + 0.5)
            kp_name = ax.get_yticklabels()[y_tick_idx].get_text()

            lookup_key = (kp_name, current_hue) if hue else kp_name
            mag = mag_map.get(lookup_key, 0)

            text_x = width + 0.01 if width < 0.5 else width - 0.1
            text_color = "black" if width < 0.5 else "white"

            ax.text(
                text_x,
                bar.get_y() + bar.get_height() / 2,
                f"M:{mag:.1f}",
                va="center",
                ha="left" if width < 0.5 else "right",
                fontsize=8,
                fontweight="bold",
                color=text_color,
            )

    ax.set_title(f"Jitter Frequency & Magnitude {model_name}")
    ax.set_xlabel("Outlier % (Bar) | 95th quantile (Text)")
    ax.grid(axis="x", linestyle="--", alpha=0.4)


def visualize_links_across_multiple_videos(links, hue=None):
    plt.figure(figsize=(14, 7))

    sns.boxplot(
        data=links,
        x="cv",
        y="link",
        hue=hue,
    )

    plt.title("CV vs Keypoints")
    plt.ylabel("Coefficient of Variation (Lower = More Stable)")
    plt.xlabel("Anatomical Link (Bone)")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()
