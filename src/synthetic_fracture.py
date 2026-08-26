"""
Tạo synthetic fracture từ mesh nguyên vẹn — backup khi không có nhãn fracture thật.

3 chiến lược:
1. Plane cut       — cắt bằng 1 mặt phẳng ngẫu nhiên (đơn giản nhất).
2. Multiple planes — cắt bằng nhiều mặt phẳng (giống vỡ phức tạp).
3. Voronoi shatter — chia bằng các vùng Voronoi ngẫu nhiên (giống vỡ thật nhất).

Hiện tại implement plane cut — đủ làm baseline.
Voronoi shatter có thể nâng cấp sau bằng pymeshlab hoặc Blender.

Usage:
    from src.synthetic_fracture import random_plane_fracture

    verts, faces = load_clean_mesh('vase.obj')
    fracture = random_plane_fracture(verts, n_planes=3, band_width=0.02)
"""
from __future__ import annotations
import numpy as np


def random_plane_fracture(
    verts: np.ndarray,
    n_planes: int = 1,
    band_width: float = 0.02,
    seed: int | None = None,
) -> np.ndarray:
    """
    Mô phỏng vết vỡ bằng cách chọn N mặt phẳng ngẫu nhiên đi qua mesh.
    Vertex nằm trong dải `band_width` quanh mặt phẳng → fracture=1.

    Args:
        verts: (N, 3) — tọa độ vertex (nên đã normalize về bbox đơn vị).
        n_planes: số mặt phẳng cắt.
        band_width: độ rộng dải fracture quanh mỗi mặt phẳng (cùng đơn vị với verts).
        seed: seed cho reproducibility.

    Returns:
        fracture: (N,) int64 — 0/1.
    """
    rng = np.random.default_rng(seed)
    fracture = np.zeros(len(verts), dtype=np.int64)
    center = verts.mean(axis=0)

    for _ in range(n_planes):
        # Mặt phẳng ngẫu nhiên đi qua centroid: pháp tuyến random + offset 0
        normal = rng.normal(size=3)
        normal /= np.linalg.norm(normal) + 1e-8

        # Khoảng cách signed của mỗi vertex tới mặt phẳng
        signed_dist = (verts - center) @ normal

        # Vertex trong dải band_width → fracture
        fracture[np.abs(signed_dist) < band_width] = 1

    return fracture


def voronoi_shatter_labels(
    verts: np.ndarray,
    n_seeds: int = 20,
    seed: int | None = None,
) -> np.ndarray:
    """
    Chia mesh thành n_seeds vùng Voronoi → trả về fragment ID per vertex.
    Convert sang binary fracture bằng `fragment_to_binary_fracture` ở loader.

    Args:
        verts: (N, 3).
        n_seeds: số vùng Voronoi.

    Returns:
        fragment_id: (N,) int64 trong [0, n_seeds-1].
    """
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    seed_idx = rng.choice(len(verts), size=n_seeds, replace=False)
    seeds = verts[seed_idx]

    tree = cKDTree(seeds)
    _, fragment_id = tree.query(verts, k=1)
    return fragment_id.astype(np.int64)


def main():
    """Demo: tạo synthetic fracture từ Stanford Bunny."""
    import urllib.request, os, tempfile
    import trimesh
    from pathlib import Path

    # Tải bunny tạm
    tmp_path = Path(tempfile.gettempdir()) / 'bunny.obj'
    if not tmp_path.exists():
        url = 'https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/stanford-bunny.obj'
        urllib.request.urlretrieve(url, tmp_path)

    mesh = trimesh.load(str(tmp_path), process=False)
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    # Normalize
    verts -= verts.mean(0)
    verts /= np.linalg.norm(verts, axis=1).max()

    # Plane cut
    fracture = random_plane_fracture(verts, n_planes=3, band_width=0.05, seed=42)
    print(f'Plane fracture ratio: {fracture.mean():.3f}')

    # Voronoi
    frag_id = voronoi_shatter_labels(verts, n_seeds=20, seed=42)
    print(f'Voronoi fragments: {len(np.unique(frag_id))}')


if __name__ == '__main__':
    main()
