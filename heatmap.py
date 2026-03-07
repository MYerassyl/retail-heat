"""Gaussian KDE heatmap generation from tracked centroids."""

import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import config
from utils import get_frame_paths


def generate_heatmap(centroids, img_width, img_height, grid_size=None):
    """Generate a density heatmap from centroid coordinates using Gaussian KDE.

    Args:
        centroids: list of (cx, cy) tuples
        img_width: image width in pixels
        img_height: image height in pixels
        grid_size: KDE evaluation grid resolution

    Returns:
        density: 2D array of density values evaluated on grid
        xx, yy: meshgrid coordinates
    """
    if grid_size is None:
        grid_size = config.HEATMAP_GRID_SIZE

    if len(centroids) < 2:
        print("  Warning: Not enough centroids for KDE, returning empty heatmap")
        xx, yy = np.mgrid[0:img_width:complex(grid_size), 0:img_height:complex(grid_size)]
        return np.zeros_like(xx), xx, yy

    cx = np.array([c[0] for c in centroids])
    cy = np.array([c[1] for c in centroids])

    # Fit Gaussian KDE
    values = np.vstack([cx, cy])
    kernel = gaussian_kde(values)

    # Evaluate on grid
    xx, yy = np.mgrid[0:img_width:complex(grid_size), 0:img_height:complex(grid_size)]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    density = np.reshape(kernel(positions).T, xx.shape)

    return density, xx, yy


def overlay_heatmap(density, xx, yy, ref_frame_path, output_path, seq_name,
                    title_prefix="SORT Baseline Heatmap"):
    """Overlay density heatmap on a reference frame and save as PNG.

    Args:
        density: 2D density array from generate_heatmap
        xx, yy: meshgrid coordinates
        ref_frame_path: path to reference frame image
        output_path: path to save the output PNG
        seq_name: sequence name for the title
        title_prefix: prefix for the plot title
    """
    ref_img = cv2.imread(ref_frame_path)
    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(ref_img, extent=[0, ref_img.shape[1], ref_img.shape[0], 0])
    ax.pcolormesh(
        xx, yy, density,
        cmap=config.HEATMAP_COLORMAP,
        alpha=config.HEATMAP_ALPHA,
        shading="auto",
    )
    ax.set_title(f"{title_prefix} — {seq_name}", fontsize=14)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.set_xlim(0, ref_img.shape[1])
    ax.set_ylim(ref_img.shape[0], 0)  # invert y to match image coords
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved heatmap to {output_path}")


def generate_sequence_heatmap(seq_name, centroids, output_dir=None,
                              title_prefix="SORT Baseline Heatmap",
                              filename_suffix=""):
    """Generate and save heatmap for a single sequence.

    Args:
        seq_name: MOT17 sequence name
        centroids: list of (cx, cy) tuples
        output_dir: directory to save heatmap (defaults to config.HEATMAP_DIR)
        title_prefix: prefix for the plot title
        filename_suffix: optional suffix appended to filename (e.g. "_deepsort")
    """
    from utils import parse_seqinfo

    if output_dir is None:
        output_dir = config.HEATMAP_DIR

    print(f"\n  Generating heatmap for {seq_name}...")

    info = parse_seqinfo(seq_name)
    img_w = info["imwidth"]
    img_h = info["imheight"]

    density, xx, yy = generate_heatmap(centroids, img_w, img_h)

    # Use first frame as reference
    frame_paths = get_frame_paths(seq_name)
    ref_frame = frame_paths[0]

    output_path = os.path.join(output_dir, f"{seq_name}_heatmap{filename_suffix}.png")
    overlay_heatmap(density, xx, yy, ref_frame, output_path, seq_name,
                    title_prefix=title_prefix)
