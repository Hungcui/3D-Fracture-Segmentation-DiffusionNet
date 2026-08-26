"""
Breaking Bad Dataset loader.

Định dạng dataset:
- mesh: file .obj (vertices + faces của object đã shatter, lưu ở vị trí gốc)
- labels: scipy CSR sparse matrix (.npz) shape (N_verts, N_fragments) — one-hot

Mỗi vertex thuộc về đúng 1 fragment (0..N_fragments-1).
Các fragment là connected components riêng (không chia sẻ vertex).
Bề mặt vỡ = nơi vertex của fragment A nằm sát vertex của fragment B trong 3D.

Usage:
    from src.breaking_bad_loader import load_breaking_bad_sample

    verts, faces, fracture = load_breaking_bad_sample(
        mesh_path='data_samples/compressed_mesh.obj',
        label_path='data_samples/compressed_data.npz',
        fracture_threshold=0.005,
    )
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
import trimesh


def load_fragment_labels(label_path: str | Path) -> np.ndarray:
    """
    Load fragment ID per vertex từ file .npz (scipy CSR sparse matrix).

    Returns:
        fragment_id: np.ndarray shape (N_verts,) dtype int — fragment ID của mỗi vertex.
    """
    data = np.load(label_path, allow_pickle=True)
    # Reconstruct CSR matrix
    mat = sp.csr_matrix(
        (data['data'], data['indices'], data['indptr']),
        shape=tuple(data['shape']),
    )
    # Mỗi row có đúng 1 nonzero (one-hot) → indices chính là labels
    # Equivalent: np.asarray(mat.argmax(axis=1)).flatten()
    return data['indices'].astype(np.int64)


def fragment_to_binary_fracture(
    verts: np.ndarray,
    fragment_id: np.ndarray,
    threshold: float = 0.005,
) -> np.ndarray:
    """
    Convert nhãn fragment ID (0..K-1) → nhãn binary (0=original, 1=fracture).

    Logic: vertex thuộc fragment A được gán fracture=1 nếu nó nằm sát
    (khoảng cách Euclidean < threshold) một vertex của fragment khác.

    Args:
        verts: (N, 3) tọa độ vertex.
        fragment_id: (N,) fragment ID per vertex.
        threshold: ngưỡng khoảng cách. Tinh chỉnh tùy mật độ vertex.

    Returns:
        fracture: (N,) dtype int64 — 0/1 binary labels.
    """
    N = len(verts)
    fracture = np.zeros(N, dtype=np.int64)

    for fid in np.unique(fragment_id):
        mask_self = fragment_id == fid
        v_self = verts[mask_self]
        v_other = verts[~mask_self]

        if len(v_other) == 0:
            continue

        tree = cKDTree(v_other)
        dists, _ = tree.query(v_self, k=1)

        idx = np.where(mask_self)[0]
        fracture[idx[dists < threshold]] = 1

    return fracture


def load_breaking_bad_sample(
    mesh_path: str | Path,
    label_path: str | Path,
    fracture_threshold: float = 0.005,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load 1 sample Breaking Bad.

    Args:
        mesh_path: đường dẫn file .obj.
        label_path: đường dẫn file .npz.
        fracture_threshold: ngưỡng để xác định bề mặt vỡ.
        normalize: chuẩn hóa mesh về bounding box đơn vị (khuyến nghị True).

    Returns:
        verts:    (N, 3) float32
        faces:    (M, 3) int64
        fracture: (N,) int64 (0/1)
    """
    # Mesh
    mesh = trimesh.load(str(mesh_path), process=False)
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    # Normalize trước khi tính khoảng cách (để threshold ổn định giữa các sample)
    if normalize:
        center = verts.mean(axis=0)
        verts = verts - center
        scale = np.linalg.norm(verts, axis=1).max()
        verts = verts / (scale + 1e-8)

    # Fragment IDs
    fragment_id = load_fragment_labels(label_path)
    assert len(fragment_id) == len(verts), (
        f'Mismatch: {len(fragment_id)} labels vs {len(verts)} vertices'
    )

    # Convert sang binary fracture
    fracture = fragment_to_binary_fracture(verts, fragment_id, threshold=fracture_threshold)

    return verts, faces, fracture


def main():
    """Test loader trên sample đi kèm project."""
    root = Path(__file__).parent.parent / 'data_samples'
    verts, faces, fracture = load_breaking_bad_sample(
        mesh_path=root / 'compressed_mesh.obj',
        label_path=root / 'compressed_data.npz',
    )
    print(f'Vertices : {verts.shape}')
    print(f'Faces    : {faces.shape}')
    print(f'Fracture : {fracture.shape}, ratio = {fracture.mean():.3f}')
    print(f'Bbox     : min={verts.min(0)}  max={verts.max(0)}')


if __name__ == '__main__':
    main()
