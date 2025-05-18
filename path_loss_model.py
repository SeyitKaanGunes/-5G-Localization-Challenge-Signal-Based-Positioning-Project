
"""
========================================
 - Geometrik mesafe ve Timing Advance’dan hesaplanan mesafeyi ortak kullanır.
 - Tek eğim veya çift eğim (LOS/NLOS) modelini veri dengesine göre otomatik seçer.
 - 3 farklı ağırlıklandırma (none, 1/D, 1/√D) × 4 Huber f_scale (0.5, 1, 2, 5) grid-search yapar.
 - En düşük RMSE’yi veren kombinasyonu raporlar.
"""
import argparse
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import trimesh
from trimesh.ray.ray_triangle import RayMeshIntersector
from scipy.optimize import least_squares
import pyproj

C_LIGHT      = 299_792_458.0  # m/s
DEFAULT_MEAS = '5g_dl.xlsx'
DEFAULT_CELLS= 'hucre_bilgileri.xlsx'
DEFAULT_OBJS = '.'
DEFAULT_OUT  = 'results.csv'


def parse_args():
    p = argparse.ArgumentParser(description='Adaptive path-loss optimiser')
    p.add_argument('--meas-xlsx', dest='meas_xlsx', default=DEFAULT_MEAS,
                   help='Measurements Excel')
    p.add_argument('--cells-xlsx', dest='cells_xlsx', default=DEFAULT_CELLS,
                   help='Cell info Excel')
    p.add_argument('--obj-dir',    dest='obj_dir',    default=DEFAULT_OBJS,
                   help='Directory of .obj files')
    p.add_argument('--freq-ghz',   dest='freq_ghz',   type=float,
                   default=3.5, help='Carrier freq (GHz)')
    p.add_argument('--output',     dest='output',     default=DEFAULT_OUT,
                   help='Output CSV')
    # Colab'ın -f argümanlarını görmezden gelmek için:
    args, _ = p.parse_known_args()
    return args


def build_coord_tf(ref_ll, ref_xyz):
    tf = pyproj.Transformer.from_crs(4326, 32635, always_xy=True)
    ux0, uy0 = tf.transform(*ref_ll)
    dx, dz = ux0 - ref_xyz[0], uy0 - ref_xyz[2]

    def ll2scene(lon, lat):
        ux, uy = tf.transform(lon, lat)
        return np.array([ux - dx, 0.0, uy - dz], dtype=np.float64)

    return ll2scene


def seg_len(inter, origin, direction, dmax):
    locs, _, _ = inter.intersects_location(
        [origin], [direction], multiple_hits=True
    )
    if len(locs) < 2:
        return 0.0
    t = np.dot(locs - origin, direction)
    t = np.sort(t[(t >= 0) & (t <= dmax)])
    if len(t) % 2:
        t = t[:-1]
    pairs = t.reshape(-1, 2)
    return float((pairs[:, 1] - pairs[:, 0]).sum())


def main():
    args = parse_args()
    freq_hz = args.freq_ghz * 1e9
    const_fs = 20 * math.log10(4 * math.pi * freq_hz / C_LIGHT)

    # 1) read measurements + cells
    meas = pd.read_excel(args.meas_xlsx)
    pci_col = next(c for c in meas.columns if 'pci' in c.lower())
    meas.rename(columns={pci_col: 'PCI'}, inplace=True)
    pl_col = 'NR_UE_Pathloss_DL_0_mean'
    meas.dropna(subset=['PCI', pl_col], inplace=True)
    meas['PCI'] = meas['PCI'].astype(int)

    cells = pd.read_excel(args.cells_xlsx)
    if 'PCI' not in cells.columns:
        raise RuntimeError('Cells Excel has no PCI column')
    cells['PCI'] = cells['PCI'].astype(int)

    # align to common PCI
    common = set(meas.PCI) & set(cells.PCI)
    if not common:
        raise RuntimeError('No overlapping PCI between meas & cells')
    meas = meas[meas.PCI.isin(common)]
    cells = cells[cells.PCI.isin(common)].drop_duplicates('PCI').reset_index(drop=True)

    # 2) load geometry
    obj_dir = Path(args.obj_dir)
    buildings = trimesh.load(obj_dir / 'buildings.obj', force='mesh')
    vegetation = trimesh.load(obj_dir / 'vegetation.obj', force='mesh')
    bs_scene = trimesh.load(
        obj_dir / 'base_stations.obj',
        split_object=True,
        group_material=False
    )
    inter_b = RayMeshIntersector(buildings)
    inter_v = RayMeshIntersector(vegetation)

    # 3) map meshes to cells
    bs_keys = list(bs_scene.geometry.keys())
    if len(bs_keys) != len(cells):
        raise RuntimeError(f'Mesh count {len(bs_keys)} != cell rows {len(cells)}')
    mesh_map = {i: bs_scene.geometry[bs_keys[i]] for i in range(len(bs_keys))}
    pci2idx = {pci: i for i, pci in enumerate(cells.PCI)}

    # 4) coord transform
    ref_xyz = np.array(mesh_map[0].centroid)
    ref_ll = (cells.loc[0, 'Longitude'], cells.loc[0, 'Latitude'])
    ll2sc = build_coord_tf(ref_ll, ref_xyz)

    # 5) build features (D, D_TA, LB, LV, PL, LOS)
    D, D_TA, LB, LV, PL, LOS = [], [], [], [], [], []
    for _, r in tqdm(meas.iterrows(), total=len(meas), desc='Features'):
        idx = pci2idx.get(int(r.PCI))
        if idx is None:
            continue

        bs_pos = np.array(mesh_map[idx].centroid)
        ue_pos = ll2sc(r.Longitude_mean, r.Latitude_mean)
        ue_pos[1] = 1.5
        d_vec = ue_pos - bs_pos
        dist = np.linalg.norm(d_vec)
        if dist < 1:
            continue
        dir_ = d_vec / dist

        lb = seg_len(inter_b, bs_pos, dir_, dist)
        lv = seg_len(inter_v, bs_pos, dir_, dist)

        ta = r.get('NR_UE_Timing_Advance_mean', np.nan)
        if not np.isnan(ta):
            d_ta = ta * C_LIGHT * 0.5e-6
        else:
            d_ta = dist

        D.append(dist)
        D_TA.append(d_ta)
        LB.append(lb)
        LV.append(lv)
        PL.append(r[pl_col])
        LOS.append(1 if (lb == 0 and lv == 0) else 0)

    D, D_TA, LB, LV, PL, LOS = map(np.array, (D, D_TA, LB, LV, PL, LOS))
    if len(D) == 0:
        sys.exit("No valid samples")

    D_eff = (D + D_TA) / 2.0

    p_los = LOS.mean()
    dual = (0.2 <= p_los <= 0.8)
    print(f"Samples: {len(D)} (LOS={LOS.sum()}, NLOS={len(D)-LOS.sum()}) -> {'dual' if dual else 'single'} slope")

    # 6) grid-search
    weight_opts = {
        'none':    np.ones_like(D_eff),
        '1/D':     1/np.maximum(D_eff, 1.0),
        '1/sqrtD': 1/np.sqrt(np.maximum(D_eff, 1.0))
    }
    f_scales = [0.5, 1.0, 2.0, 5.0]
    best = {'rmse': np.inf}

    for wnm, W in weight_opts.items():
        if dual:
            def res_fn(x):
                Lb, Lv, nL, nN, C0 = x
                n_eff = nL * LOS + (1 - LOS) * nN
                return (C0 + const_fs
                        + 10 * n_eff * np.log10(D_eff)
                        + Lb * LB + Lv * LV - PL) * W
            x0 = [0.5, 0.05, 2.0, 3.0, 0.0]
            bds = ([0, 0, 1, 1, -20], [10, 0.1, 4, 6, 20])
        else:
            def res_fn(x):
                Lb, Lv, n, C0 = x
                return (C0 + const_fs
                        + 10 * n * np.log10(D_eff)
                        + Lb * LB + Lv * LV - PL) * W
            x0 = [0.5, 0.2, 2.0, 0.0]
            bds = ([0, 0, 1, -20], [10, 10, 6, 20])

        for fs in f_scales:
            sol = least_squares(res_fn, x0, bounds=bds, loss='huber', f_scale=fs)

            if dual:
                tmp = (sol.x[4] + const_fs
                       + 10 * (sol.x[2] * LOS + (1 - LOS) * sol.x[3]) * np.log10(D_eff)
                       + sol.x[0] * LB + sol.x[1] * LV - PL)
            else:
                tmp = (sol.x[3] + const_fs
                       + 10 * sol.x[2] * np.log10(D_eff)
                       + sol.x[0] * LB + sol.x[1] * LV - PL)

            rmse = math.sqrt(np.mean(tmp**2))
            print(f"{wnm:8s} f_scale={fs:<4} -> RMSE={rmse:.3f}")
            if rmse < best['rmse']:
                best = {
                    'rmse': rmse,
                    'params': sol.x,
                    'dual': dual,
                    'weight': wnm,
                    'f_scale': fs,
                    'const_fs': const_fs
                }

    print("\nBest combo:", best)
    x = best['params']

    # 7) predict & save
    if best['dual']:
        Lb, Lv, nL, nN, C0 = x
        n_eff = nL * LOS + (1 - LOS) * nN
    else:
        Lb, Lv, n, C0 = x
        n_eff = n

    preds = C0 + best['const_fs'] + 10 * n_eff * np.log10(D_eff) + Lb * LB + Lv * LV

    df_out = pd.DataFrame({
        'distance_geom_m': D,
        'distance_ta_m':   D_TA,
        'distance_eff_m':  D_eff,
        'len_building':    LB,
        'len_veg':         LV,
        'pathloss_meas':   PL,
        'pathloss_pred':   preds,
        'residual_db':     preds - PL
    })
    df_out.to_csv(args.output, index=False)
    print("Saved →", args.output)

    # 8) plot
    idxs = np.arange(len(D))
    plt.figure(figsize=(12, 4))
    plt.plot(idxs, PL, label='Measured',  linewidth=1)
    plt.plot(idxs, preds, label='Predicted', linewidth=1)
    plt.xlabel("Sample index")
    plt.ylabel("Path-loss (dB)")
    plt.title("Measured vs Predicted Path-loss")
    plt.legend()
    plt.grid(ls='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
