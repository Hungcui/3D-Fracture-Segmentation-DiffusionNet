"""
Visualize mesh + labels.

Hỗ trợ 2 backend:
- matplotlib (mặc định, không cần cài thêm) — chậm, ảnh tĩnh.
- plotly (đẹp hơn, tương tác được) — chạy trong notebook.

Usage:
    from src.visualize import save_fracture_figure, plotly_mesh

    # Lưu ảnh PNG so sánh fragment vs binary fracture
    save_fracture_figure(verts, faces, fragment_id, fracture, 'out.png')

    # Hiển thị tương tác trong notebook
    plotly_mesh(verts, faces, fracture).show()
"""
from __future__ import annotations
from pathlib import Path

import numpy as np


def save_fracture_figure(
    verts: np.ndarray,
    faces: np.ndarray,
    fragment_id: np.ndarray | None,
    fracture: np.ndarray,
    out_path: str | Path,
    dpi: int = 120,
):
    """Lưu PNG: trái = fragment colors, phải = binary fracture (đỏ/xanh)."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    n_panels = 2 if fragment_id is not None else 1
    fig = plt.figure(figsize=(7 * n_panels, 6))
    cmap = plt.cm.tab20

    if fragment_id is not None:
        ax1 = fig.add_subplot(1, n_panels, 1, projection='3d')
        fc1 = cmap(fragment_id[faces[:, 0]] % 20 / 20)
        ax1.add_collection3d(
            Poly3DCollection(verts[faces], facecolors=fc1, edgecolors='none', alpha=0.95)
        )
        ax1.set_title(f'Fragment IDs ({len(np.unique(fragment_id))} clusters)')
        _set_axes_3d(ax1, verts)

    ax2 = fig.add_subplot(1, n_panels, n_panels, projection='3d')
    fc2 = np.array([
        [1, 0.2, 0.2, 1] if fracture[f].any() else [0.6, 0.8, 1, 1]
        for f in faces
    ])
    ax2.add_collection3d(
        Poly3DCollection(verts[faces], facecolors=fc2, edgecolors='none', alpha=0.95)
    )
    ax2.set_title(f'Binary fracture — {100 * fracture.mean():.1f}% fracture')
    _set_axes_3d(ax2, verts)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def save_exploded_figure(
    verts: np.ndarray,
    faces: np.ndarray,
    fragment_id: np.ndarray,
    out_path: str | Path,
    explode_factor: float = 0.3,
    dpi: int = 120,
):
    """Lưu PNG: trái = vị trí gốc, phải = explode mỗi fragment ra ngoài."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    center = verts.mean(axis=0)
    exploded = verts.copy()
    for fid in np.unique(fragment_id):
        mask = fragment_id == fid
        frag_center = verts[mask].mean(axis=0)
        direction = frag_center - center
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        exploded[mask] += direction * explode_factor

    fig = plt.figure(figsize=(14, 7))
    cmap = plt.cm.tab20

    for i, (v, title) in enumerate([
        (verts, 'Original positions'),
        (exploded, f'Exploded (factor={explode_factor})'),
    ]):
        ax = fig.add_subplot(1, 2, i + 1, projection='3d')
        fc = cmap(fragment_id[faces[:, 0]] % 20 / 20)
        ax.add_collection3d(
            Poly3DCollection(v[faces], facecolors=fc, edgecolors='none', alpha=0.95)
        )
        ax.set_title(title)
        _set_axes_3d(ax, v)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plotly_mesh(verts: np.ndarray, faces: np.ndarray, vertex_colors: np.ndarray | None = None):
    """Render mesh tương tác trong notebook bằng Plotly."""
    import plotly.graph_objects as go

    if vertex_colors is None:
        color = 'lightblue'
        vc = None
    elif vertex_colors.dtype.kind in 'iu':  # binary or int labels
        vc = ['red' if c else 'lightblue' for c in vertex_colors]
        color = None
    else:
        vc = vertex_colors
        color = None

    fig = go.Figure(data=[
        go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=color, vertexcolor=vc,
            flatshading=True, opacity=0.95,
        )
    ])
    fig.update_layout(scene=dict(aspectmode='data'), width=700, height=500)
    return fig


def _set_axes_3d(ax, verts):
    ax.set_xlim(verts[:, 0].min(), verts[:, 0].max())
    ax.set_ylim(verts[:, 1].min(), verts[:, 1].max())
    ax.set_zlim(verts[:, 2].min(), verts[:, 2].max())
    ax.set_box_aspect([1, 1, 1.5])
    ax.axis('off')


if __name__ == '__main__':
    from breaking_bad_loader import load_breaking_bad_sample, load_fragment_labels

    root = Path(__file__).parent.parent / 'data_samples'
    verts, faces, fracture = load_breaking_bad_sample(
        mesh_path=root / 'compressed_mesh.obj',
        label_path=root / 'compressed_data.npz',
    )
    fragment_id = load_fragment_labels(root / 'compressed_data.npz')

    out_dir = Path(__file__).parent.parent / 'data_samples'
    save_fracture_figure(verts, faces, fragment_id, fracture, out_dir / 'fracture_view.png')
    save_exploded_figure(verts, faces, fragment_id, out_dir / 'exploded_view.png')
    print(f'Saved: {out_dir / "fracture_view.png"}')
    print(f'Saved: {out_dir / "exploded_view.png"}')
