from __future__ import annotations

from pathlib import Path
from typing import Any

from mpl_toolkits.axes_grid1 import make_axes_locatable

EXPERT_DISPLAY_NAMES = {
    0: "Base",
    1: "Code",
    2: "Creative Writing",
    3: "Math",
    4: "News",
    5: "Academic",
    6: "Danish",
}

EXPERT_DISPLAY_NAMES_7X7 = {
    0: "Base",
    1: "Code",
    2: "Creative Writing",
    3: "Math",
    4: "News",
    5: "Academic",
}

EXPERT_COLORS = {
    "Base": "#4C566A",
    "Code": "#4C78A8",
    "Creative Writing": "#E45756",
    "Math": "#54A24B",
    "News": "#F58518",
    "Academic": "#B279A2",
    "Reddit": "#72B7B2",
    "Danish": "#EECA3B",
}

DATASET_DISPLAY_NAMES = {
    "multi_wiki_qa_da": "MultiWiki QA (DA)",
    "multi_wiki_qa_en": "MultiWiki QA (EN)",
    "mkqa_en_da": "MGQA (EN/DA)",
    "gsm8k_subset": "GSM8K",
    "mbpp_subset": "MBPP",
    "pubmedqa_subset": "PubMedQA",
    "ag_news_subset": "AG News",
    "common_gen_subset": "CommonGen",
}

FONT_WEIGHTS = {
    "title": "semibold",
    "suptitle": "bold",
    "label": "semibold",
    "tick": "semibold",
    "legend": "semibold",
}

FONT_SIZES = {
    "title": 11.0,
    "suptitle": 14.0,
    "axis_label": 11.0,
    "tick": 9.5,
    "legend": 9.5,
    "annotation": 8.0,
}


def model_display_name(model_name: str) -> str:
    name = model_name.removeprefix("FlexOlmo-")
    name = name.replace("-1T-", "-")
    return name


def dataset_display_name(dataset_name: str) -> str:
    return DATASET_DISPLAY_NAMES.get(dataset_name, dataset_name)


def expected_num_experts_for_model(model_name: str) -> int | None:
    if "7x7B" in model_name:
        return 6
    if "8x7B" in model_name:
        return 7
    return None


def expert_display_name(expert_idx: int) -> str:
    return EXPERT_DISPLAY_NAMES.get(expert_idx, f"Expert {expert_idx}")


def _format_expert_label(label: str) -> str:
    if label == "Creative Writing":
        return "Creative\nWriting"
    return label


def expert_tick_labels(num_experts: int, multiline: bool = False) -> list[str]:
    labels = [expert_display_name(idx) for idx in range(num_experts)]
    return [_format_expert_label(label) for label in labels]


def expert_tick_labels_for_model(model_name: str, num_experts: int, multiline: bool = False) -> list[str]:
    expected = expected_num_experts_for_model(model_name)
    if expected is not None:
        num_experts = min(num_experts, expected)
    if "7x7B" in model_name:
        labels = [EXPERT_DISPLAY_NAMES_7X7.get(idx, f"Expert {idx}") for idx in range(num_experts)]
    else:
        labels = [expert_display_name(idx) for idx in range(num_experts)]
    return [_format_expert_label(label) for label in labels]


def expert_colors_for_model(model_name: str, num_experts: int) -> list[str]:
    expected = expected_num_experts_for_model(model_name)
    if expected is not None:
        num_experts = min(num_experts, expected)
    if "7x7B" in model_name:
        labels = [EXPERT_DISPLAY_NAMES_7X7.get(idx, f"Expert {idx}") for idx in range(num_experts)]
    else:
        labels = [expert_display_name(idx) for idx in range(num_experts)]
    return [EXPERT_COLORS.get(label, "#777777") for label in labels]


def apply_axis_text_style(ax) -> None:
    for label in ax.get_xticklabels():
        label.set_fontsize(FONT_SIZES["tick"])
        label.set_fontweight(FONT_WEIGHTS["tick"])
    for label in ax.get_yticklabels():
        label.set_fontsize(FONT_SIZES["tick"])
        label.set_fontweight(FONT_WEIGHTS["tick"])


def style_axis_title(ax, title: str, *, pad: float = 4) -> None:
    ax.set_title(title, fontsize=FONT_SIZES["title"], fontweight=FONT_WEIGHTS["title"], pad=pad)


def style_axis_labels(ax, xlabel: str = "", ylabel: str = "", *, xlabel_pad: float | None = None, ylabel_pad: float | None = None) -> None:
    if xlabel:
        kwargs = {"fontsize": FONT_SIZES["axis_label"], "fontweight": FONT_WEIGHTS["label"]}
        if xlabel_pad is not None:
            kwargs["labelpad"] = xlabel_pad
        ax.set_xlabel(xlabel, **kwargs)
    else:
        ax.set_xlabel("")
    if ylabel:
        kwargs = {"fontsize": FONT_SIZES["axis_label"], "fontweight": FONT_WEIGHTS["label"]}
        if ylabel_pad is not None:
            kwargs["labelpad"] = ylabel_pad
        ax.set_ylabel(ylabel, **kwargs)
    else:
        ax.set_ylabel("")


def style_suptitle(fig, title: str, *, y: float = 0.96) -> None:
    fig.suptitle(title, fontsize=FONT_SIZES["suptitle"], fontweight=FONT_WEIGHTS["suptitle"], y=y)


def style_legend(legend) -> None:
    if legend is None:
        return
    for text in legend.get_texts():
        text.set_fontweight(FONT_WEIGHTS["legend"])
        text.set_fontsize(FONT_SIZES["legend"])


def style_colorbar(colorbar, label: str) -> None:
    colorbar.set_label(label, fontsize=FONT_SIZES["axis_label"], fontweight=FONT_WEIGHTS["label"])
    colorbar.ax.tick_params(labelsize=FONT_SIZES["tick"], pad=0.5)
    for tick in colorbar.ax.get_yticklabels():
        tick.set_fontweight("normal")


def style_heatmap_ticklabels(ax, xlabels: list[str], ylabels: list[str], *, x_rotation: float = 35) -> None:
    ax.set_xticks([idx + 0.5 for idx in range(len(xlabels))])
    ax.set_yticks([idx + 0.5 for idx in range(len(ylabels))])
    ax.set_xticklabels(xlabels, rotation=x_rotation, ha="right")
    ax.set_yticklabels(ylabels, rotation=0, ha="right", va="center")
    apply_axis_text_style(ax)


def compose_panel_title(model_name: str, dataset_name: str = "", prefix: str = "") -> str:
    parts: list[str] = []
    if prefix:
        parts.append(prefix)
    parts.append(model_display_name(model_name))
    if dataset_name:
        parts.append(dataset_display_name(dataset_name))
    return " | ".join(parts)


def add_shared_ylabel(fig: Any, axes: Any, label: str, *, x: float | None = None) -> None:
    first_box = axes[0][0].get_position()
    fig.supylabel(
        label,
        x=first_box.x0 - 0.04 if x is None else x,
        fontsize=FONT_SIZES["axis_label"] - 0.5,
        fontweight=FONT_WEIGHTS["label"],
    )


def add_top_right_colorbar(fig: Any, axes: Any, image: Any, label: str) -> Any:
    anchor_ax = axes[0, -1]
    num_rows = axes.shape[0]
    if num_rows > 1:
        bbox = anchor_ax.get_position()
        heatmap_width = bbox.width * 0.84
        colorbar_gap = bbox.width * 0.02
        colorbar_width = bbox.width * 0.035
        anchor_ax.set_position([bbox.x0, bbox.y0, heatmap_width, bbox.height])
        cax = fig.add_axes(
            [
                bbox.x0 + heatmap_width + colorbar_gap,
                bbox.y0 + bbox.height * 0.08,
                colorbar_width,
                bbox.height * 0.84,
            ]
        )
    else:
        bbox = anchor_ax.get_position()
        cax = fig.add_axes([bbox.x1 + 0.004, bbox.y0 + bbox.height * 0.08, 0.010, bbox.height * 0.84])
    colorbar = fig.colorbar(image, cax=cax)
    style_colorbar(colorbar, label)
    return colorbar


def save_figure(fig, output_path: str | Path, *, dpi: int = 220) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.06)
