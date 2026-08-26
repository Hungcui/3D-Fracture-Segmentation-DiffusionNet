=====================================================================
  DEMO - Fracture-Surface Segmentation with DiffusionNet
  3D Surface Fracture Segmentation Demo - USTH
=====================================================================

MUC DICH
  Demo truc tiep cho hoi dong sau khi thuyet trinh: chon 1 vat the vo
  -> model DiffusionNet du doan BE MAT VO ngay tai cho (~0.5s) ->
  mo cua so 3D xoay duoc de so sanh Ground Truth vs Prediction.

-------------------------------------------------------------
CHAY DEMO (lam viec nay khi bao ve)
-------------------------------------------------------------
  Mo terminal trong thu muc nay, go:

      python demo.py

  -> Hien menu 6 vat the (3 Breaking Bad synthetic + 3 Fantastic
     Breaks do gom THAT). Go so 0-5 de chon, Enter.
  -> NEU HOI DONG MUON VAT THE KHAC:
       go 'b'  = boc NGAU NHIEN 1 vat the Breaking Bad moi
       go 'bN' = Breaking Bad vo thanh N manh (vd: b5, b8, b12)
       go 'f'  = boc NGAU NHIEN 1 manh Fantastic Breaks moi
       (tinh truc tiep, mat ~10-15s vi phai tinh operator cho mesh moi;
        terminal se bao 'computing operators ~10s' trong luc cho.
        Voi 'bN' neu khong tim thay dung N manh se lay gan nhat.)
  -> Terminal in IoU / Precision / Recall.
  -> Trinh duyet mo hinh 3D:
       - KEO CHUOT  = xoay
       - LAN CHUOT  = zoom
       - 2 nut tren cung: GROUND TRUTH  /  PREDICTION  (bam de doi)
       - vat the Breaking Bad duoc TACH ROI (exploded) de thay mat vo
  -> Go so khac de xem vat the khac, go 'q' de thoat.

  Cach nhanh (mo thang 1 vat the):   python demo.py 3

-------------------------------------------------------------
CHUAN BI 1 LAN (da chay san roi, chi lam lai neu loi)
-------------------------------------------------------------
      python build_demo.py
  Tao lai data/ + cache/ (decompress, decimate, tinh operator).
  Mat ~1 phut. SAU DO demo.py chay tuc thi (chi con forward pass).

-------------------------------------------------------------
CAC VAT THE TRONG DEMO (IoU = do trung khop vung vo)
-------------------------------------------------------------
  [0] Breaking Bad - 6 manh   IoU 0.47
  [1] Breaking Bad - 5 manh   IoU 0.42
  [2] Breaking Bad - 4 manh   IoU 0.38
  [3] Fantastic Breaks (gom that) shard 18   IoU 0.63   <- dep nhat
  [4] Fantastic Breaks (gom that) shard 02   IoU 0.62
  [5] Fantastic Breaks (gom that) shard 12   IoU 0.56

  GOI Y trinh dien: mo [3] hoac [0] truoc (IoU cao, nhin ro), bam
  qua lai GROUND TRUTH <-> PREDICTION de cho thay model bam dung
  mat vo (mau do) va de nguyen be mat goc (mau xam).

-------------------------------------------------------------
YEU CAU
-------------------------------------------------------------
  - Python da cai: torch (CPU), plotly, kaleido, trimesh, igl,
    potpourri3d, robust_laplacian, fast_simplification, scipy.
  - Repo diffusion_net o:  E:/diffusion-net/src  (demo.py tro toi day).
  - Model: model_bb.pt (Breaking Bad), model_fb.pt (Fantastic Breaks)
    -- da co san trong thu muc nay.
  - KHONG can mang, KHONG can GPU, KHONG can Kaggle.

LUU Y: thu muc nay tu chua du de chay (tru repo diffusion_net o E:/).
Neu chuyen sang may khac, copy ca E:/diffusion-net va sua duong dan
'E:/diffusion-net/src' o dau file demo.py + build_demo.py.
=====================================================================
