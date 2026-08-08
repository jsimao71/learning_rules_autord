"""Reusable visualizations for high-dimensional hierarchical datasets.

The helpers deliberately depend only on NumPy and Matplotlib so dataset
providers can use them without adding a preprocessing or ML dependency.
Plotting functions return their Matplotlib objects and never call ``show``;
callers retain control over display, saving, and cleanup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


@dataclass(frozen=True)
class PCAResult:
    """Coordinates and fitted quantities from a NumPy SVD-based PCA."""

    coordinates: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray
    mean: np.ndarray


def _feature_matrix(x) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError("x must have shape [nonzero samples, nonzero features]")
    if not np.all(np.isfinite(values)):
        raise ValueError("x contains NaN or Inf")
    return values


def fit_pca(x, n_components: int = 2) -> PCAResult:
    """Fit PCA with NumPy SVD and project ``x`` onto leading components."""

    values = _feature_matrix(x)
    available = min(values.shape)
    if not 1 <= n_components <= available:
        raise ValueError(f"n_components must be between 1 and {available}")

    mean = values.mean(axis=0)
    centered = values - mean
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    variance = singular_values**2 / max(1, len(values) - 1)
    total_variance = variance.sum()
    ratios = variance / total_variance if total_variance > 0 else np.zeros_like(variance)
    components = right_vectors[:n_components]
    return PCAResult(
        coordinates=centered @ components.T,
        components=components,
        explained_variance_ratio=ratios[:n_components],
        mean=mean,
    )


def _sample_indices(sample_count: int, max_points: int | None, seed: int) -> np.ndarray:
    if max_points is None or max_points >= sample_count:
        return np.arange(sample_count)
    if max_points <= 0:
        raise ValueError("max_points must be positive or None")
    return np.sort(np.random.default_rng(seed).choice(sample_count, max_points, replace=False))


def plot_pca_hierarchy(
    x,
    labels_by_level: Sequence[np.ndarray],
    *,
    level_names: Sequence[str] | None = None,
    max_points: int | None = 3000,
    seed: int = 0,
    columns: int = 3,
    point_size: float = 9.0,
    alpha: float = 0.65,
    cmap: str = "tab20",
) -> tuple[Figure, np.ndarray]:
    """Show one shared 2-D PCA projection colored at every hierarchy level."""

    values = _feature_matrix(x)
    labels = tuple(np.asarray(level) for level in labels_by_level)
    if not labels:
        raise ValueError("labels_by_level must contain at least one level")
    if any(level.ndim != 1 or len(level) != len(values) for level in labels):
        raise ValueError("each hierarchy label vector must contain one label per sample")
    if level_names is None:
        level_names = tuple(f"Level {index}" for index in range(len(labels)))
    if len(level_names) != len(labels):
        raise ValueError("level_names must match labels_by_level")
    if columns <= 0:
        raise ValueError("columns must be positive")

    component_count = min(2, min(values.shape))
    pca = fit_pca(values, component_count)
    coordinates = pca.coordinates
    if coordinates.shape[1] == 1:
        coordinates = np.column_stack((coordinates[:, 0], np.zeros(len(coordinates))))
    indices = _sample_indices(len(values), max_points, seed)

    column_count = min(columns, len(labels))
    row_count = int(np.ceil(len(labels) / column_count))
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(5.0 * column_count, 4.1 * row_count),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    explained = 100 * np.pad(pca.explained_variance_ratio, (0, 2 - component_count))

    for index, (level, name) in enumerate(zip(labels, level_names)):
        axis = axes.flat[index]
        sampled_labels = level[indices]
        unique_labels, categorical = np.unique(sampled_labels, return_inverse=True)
        color_map = plt.get_cmap(cmap, max(1, len(unique_labels)))
        scatter = axis.scatter(
            coordinates[indices, 0],
            coordinates[indices, 1],
            c=categorical,
            cmap=color_map,
            s=point_size,
            alpha=alpha,
            linewidths=0,
            rasterized=len(indices) > 1500,
        )
        axis.set_title(f"{name} ({len(unique_labels)} observed nodes)")
        axis.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
        axis.set_ylabel(f"PC2 ({explained[1]:.1f}%)")
        axis.grid(alpha=0.15)
        colorbar = figure.colorbar(scatter, ax=axis, shrink=0.78, pad=0.02)
        colorbar.set_label("Node index")
        if len(unique_labels) <= 12:
            colorbar.set_ticks(np.arange(len(unique_labels)))
            colorbar.set_ticklabels(unique_labels)

    for axis in axes.flat[len(labels):]:
        axis.set_visible(False)
    figure.suptitle("Shared PCA projection colored by hierarchy level", fontsize=14)
    return figure, axes


def plot_variance_spectrum(
    x,
    *,
    max_components: int = 20,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot per-component and cumulative PCA explained variance."""

    values = _feature_matrix(x)
    component_count = min(max_components, min(values.shape))
    if component_count <= 0:
        raise ValueError("max_components must be positive")
    pca = fit_pca(values, component_count)
    component_numbers = np.arange(1, component_count + 1)

    if ax is None:
        figure, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    else:
        figure = ax.figure
    ax.bar(
        component_numbers,
        pca.explained_variance_ratio,
        color="#4c78a8",
        alpha=0.8,
        label="Individual",
    )
    ax.plot(
        component_numbers,
        np.cumsum(pca.explained_variance_ratio),
        marker="o",
        color="#e45756",
        label="Cumulative",
    )
    ax.set(xlabel="Principal component", ylabel="Explained variance ratio", ylim=(0, 1.02))
    ax.set_title("PCA variance spectrum")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    return figure, ax


def plot_feature_correlation(
    x,
    *,
    max_features: int = 40,
    ax: Axes | None = None,
    cmap: str = "coolwarm",
) -> tuple[Figure, Axes, np.ndarray]:
    """Plot correlations for the highest-variance features.

    Returns the figure, axis, and original feature indices included in the plot.
    """

    values = _feature_matrix(x)
    if max_features <= 0:
        raise ValueError("max_features must be positive")
    feature_count = min(max_features, values.shape[1])
    selected = np.argsort(np.var(values, axis=0))[-feature_count:]
    selected.sort()
    correlation = np.corrcoef(values[:, selected], rowvar=False)
    correlation = np.atleast_2d(np.nan_to_num(correlation, nan=0.0))

    if ax is None:
        figure, ax = plt.subplots(figsize=(6.0, 5.2), constrained_layout=True)
    else:
        figure = ax.figure
    image = ax.imshow(correlation, vmin=-1, vmax=1, cmap=cmap, aspect="equal")
    ax.set_title(f"Feature correlation ({feature_count} highest-variance features)")
    ax.set_xlabel("Feature index")
    ax.set_ylabel("Feature index")
    if feature_count <= 20:
        ticks = np.arange(feature_count)
        ax.set_xticks(ticks, selected, rotation=90)
        ax.set_yticks(ticks, selected)
    figure.colorbar(image, ax=ax, shrink=0.8, label="Correlation")
    return figure, ax, selected


def _hierarchy_order(level: int, node: int, parent_maps: Sequence[np.ndarray]) -> tuple[int, ...]:
    ancestry = [int(node)]
    current = int(node)
    for parent_map in parent_maps[level:]:
        current = int(parent_map[current])
        ancestry.append(current)
    return tuple(reversed(ancestry))


def plot_hierarchy_tree(
    parent_maps: Sequence[np.ndarray],
    k_levels: Sequence[int],
    *,
    level_names: Sequence[str] | None = None,
    ax: Axes | None = None,
    annotate_limit: int = 60,
) -> tuple[Figure, Axes]:
    """Plot a layered child-to-parent hierarchy, ordered to reduce crossings."""

    cardinalities = tuple(int(k) for k in k_levels)
    maps = tuple(np.asarray(parent_map, dtype=int) for parent_map in parent_maps)
    if not cardinalities or len(maps) != len(cardinalities) - 1:
        raise ValueError("parent_maps must contain one map per adjacent level pair")
    for level, parent_map in enumerate(maps):
        if parent_map.shape != (cardinalities[level],):
            raise ValueError(f"parent map {level} has the wrong shape")
        if np.any(parent_map < 0) or np.any(parent_map >= cardinalities[level + 1]):
            raise ValueError(f"parent map {level} contains an invalid parent")
    if level_names is None:
        level_names = tuple(f"Level {level}" for level in range(len(cardinalities)))
    if len(level_names) != len(cardinalities):
        raise ValueError("level_names must match k_levels")

    positions: dict[tuple[int, int], tuple[float, float]] = {}
    for level, cardinality in enumerate(cardinalities):
        order = sorted(
            range(cardinality), key=lambda node: _hierarchy_order(level, node, maps)
        )
        horizontal = np.linspace(0, 1, cardinality) if cardinality > 1 else np.array([0.5])
        for x_position, node in zip(horizontal, order):
            positions[(level, node)] = (float(x_position), float(level))

    if ax is None:
        width = min(18.0, max(7.0, 0.28 * max(cardinalities)))
        height = max(4.0, 1.5 * len(cardinalities))
        figure, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    else:
        figure = ax.figure

    for level, parent_map in enumerate(maps):
        for child, parent in enumerate(parent_map):
            child_position = positions[(level, child)]
            parent_position = positions[(level + 1, int(parent))]
            ax.plot(
                (child_position[0], parent_position[0]),
                (child_position[1], parent_position[1]),
                color="#999999",
                linewidth=0.8,
                alpha=0.55,
                zorder=1,
            )

    color_map = plt.get_cmap("viridis", len(cardinalities))
    annotate = sum(cardinalities) <= annotate_limit
    for level, cardinality in enumerate(cardinalities):
        coordinates = np.array([positions[(level, node)] for node in range(cardinality)])
        ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            s=55,
            color=color_map(level),
            edgecolor="white",
            linewidth=0.6,
            zorder=2,
            label=f"{level_names[level]} ({cardinality})",
        )
        if annotate:
            for node, (x_position, y_position) in enumerate(coordinates):
                ax.annotate(
                    str(node),
                    (x_position, y_position),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                )

    ax.set_yticks(range(len(cardinalities)), level_names)
    ax.set_xticks([])
    ax.set_xlim(-0.04, 1.04)
    ax.set_title("Dataset hierarchy")
    for spine_name in ("top", "right", "bottom"):
        ax.spines[spine_name].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    return figure, ax
