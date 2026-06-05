import seaborn as sns
import matplotlib.pyplot as plt

from flexolmo_analysis.toolkit.plotting.style import (
    FONT_SIZES,
    FONT_WEIGHTS,
    apply_axis_text_style,
    expected_num_experts_for_model,
    expert_tick_labels_for_model,
    style_axis_labels,
    style_axis_title,
)


def plot_expert_heatmap(matrix):

    plt.figure(figsize=(8,4))

    sns.heatmap(matrix, cmap="viridis")

    plt.xlabel("Expert")
    plt.ylabel("Layer")

    plt.title("Layer × Expert Usage")

    plt.show()


def plot_expert_combination_upset(combination_counts, path, title, model_name, max_combinations=12):
    """
    Save a lightweight upset-style plot for expert activation combinations.
    `combination_counts` should map tuples like (0, 2, 3) -> count.
    """

    if not combination_counts:
        raise ValueError("No expert combinations were provided.")

    expected_num_experts = expected_num_experts_for_model(model_name)
    max_valid_expert = expected_num_experts - 1 if expected_num_experts is not None else None
    if max_valid_expert is not None:
        filtered_counts = {
            combo: count
            for combo, count in combination_counts.items()
            if all(expert <= max_valid_expert for expert in combo)
        }
    else:
        filtered_counts = dict(combination_counts)

    top_items = sorted(
        filtered_counts.items(),
        key=lambda item: (-item[1], len(item[0]), item[0]),
    )[:max_combinations]
    if not top_items:
        return False

    all_experts = sorted({expert for combo, _count in top_items for expert in combo})
    expert_to_row = {expert: idx for idx, expert in enumerate(all_experts)}

    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.05)
    ax_bar = fig.add_subplot(grid[0])
    ax_matrix = fig.add_subplot(grid[1], sharex=ax_bar)

    x_positions = list(range(len(top_items)))
    counts = [count for _combo, count in top_items]
    labels = ["{" + ",".join(str(expert) for expert in combo) + "}" for combo, _count in top_items]

    ax_bar.bar(x_positions, counts, color="#2f6db2")
    style_axis_labels(ax_bar, "", "Examples")
    style_axis_title(ax_bar, title, pad=4)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    for x_pos, (combo, _count) in zip(x_positions, top_items):
        rows = [expert_to_row[expert] for expert in combo]
        ax_matrix.scatter([x_pos] * len(rows), rows, s=90, color="#2f6db2", zorder=3)
        if len(rows) > 1:
            ax_matrix.plot([x_pos, x_pos], [min(rows), max(rows)], color="#2f6db2", linewidth=2)

    max_expert = max(all_experts) if all_experts else -1
    labels_by_idx = expert_tick_labels_for_model(model_name, max_expert + 1, multiline=True)
    ax_matrix.set_yticks(list(range(len(all_experts))))
    ax_matrix.set_yticklabels([f"{labels_by_idx[expert]} ({expert})" for expert in all_experts])
    ax_matrix.set_xticks(x_positions)
    ax_matrix.set_xticklabels(labels, rotation=45, ha="right")
    style_axis_labels(ax_matrix, "Activated expert combination", "")
    apply_axis_text_style(ax_matrix)
    ax_matrix.grid(axis="y", linestyle="--", alpha=0.3)
    ax_matrix.spines["top"].set_visible(False)
    ax_matrix.spines["right"].set_visible(False)

    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True
