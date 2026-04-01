#!/usr/bin/env python3
"""Phase 5: Six-pack (Čech) — 멀티프로세싱 병렬 계산"""

import os, time, gc, glob
import numpy as np
from gudhi import SimplexTree
from collections import defaultdict
from persim import PersistenceImager
import persim.images_weights as weights
from multiprocessing import Pool, cpu_count
import psutil

# ═══════════════════════════════════════════════════════════════════
# 설정
# ═══════════════════════════════════════════════════════════════════
BASE_DIR = os.path.join(os.getcwd(), 'Data', 'ParamSweep_Input')
OUT_DIR = os.path.join(os.getcwd(), 'Data', 'Sixpack_Cech')
os.makedirs(OUT_DIR, exist_ok=True)

MAX_RADIUS = 5
A_VALS = [0.0, 0.01, 0.05, 0.09, 0.13, 0.17, 0.21, 0.25]
PARAM_LIST = [(x1, x2, x3) for x1 in A_VALS for x2 in A_VALS for x3 in A_VALS]
N_WORKERS = min(cpu_count(), 8)

# ═══════════════════════════════════════════════════════════════════
# Čech Complex 구축
# ═══════════════════════════════════════════════════════════════════
def compute_Cech_cpu(points, max_radius=5.0):
    n = len(points)
    st = SimplexTree()
    for i in range(n):
        st.insert([i], filtration=0.0)

    pts = np.asarray(points, dtype=np.float64)
    diff = pts[:, None, :] - pts[None, :, :]
    dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))
    del diff

    radius_mat = dist_mat / 2.0
    mask_edge = np.triu(radius_mat <= max_radius, k=1)
    edge_i, edge_j = np.where(mask_edge)
    edge_radii = radius_mat[edge_i, edge_j]

    for ei, ej, er in zip(edge_i, edge_j, edge_radii):
        st.insert([int(ei), int(ej)], filtration=float(er))

    del mask_edge, edge_radii
    n_edges = len(edge_i)

    if n_edges > 0:
        adj = [set() for _ in range(n)]
        for ei, ej in zip(edge_i, edge_j):
            adj[ei].add(ej)
            adj[ej].add(ei)

        tri_list = []
        for ei, ej in zip(edge_i, edge_j):
            common = adj[ei].intersection(adj[ej])
            for k in common:
                if k > ej:
                    tri_list.append((int(ei), int(ej), int(k)))
        del adj

        if tri_list:
            triangles = np.array(tri_list, dtype=np.int32)
            del tri_list

            p0 = pts[triangles[:, 0]]
            p1 = pts[triangles[:, 1]]
            p2 = pts[triangles[:, 2]]

            a = np.linalg.norm(p1 - p2, axis=1)
            b = np.linalg.norm(p0 - p2, axis=1)
            c = np.linalg.norm(p0 - p1, axis=1)
            del p0, p1, p2

            s = (a + b + c) / 2.0
            area_sq = s * (s - a) * (s - b) * (s - c)
            area_sq = np.maximum(area_sq, 0.0)

            degenerate = area_sq <= 1e-10
            safe_area = np.where(degenerate, np.ones_like(area_sq), area_sq)
            circumradius = np.where(
                degenerate,
                np.maximum(np.maximum(a, b), c) / 2.0,
                (a * b * c) / (4.0 * np.sqrt(safe_area))
            )
            del a, b, c, s, area_sq, safe_area, degenerate

            valid_mask = circumradius <= max_radius
            valid_tri = triangles[valid_mask]
            valid_radii = circumradius[valid_mask]
            del circumradius, valid_mask

            for (ti, tj, tk), tr in zip(valid_tri, valid_radii):
                st.insert([int(ti), int(tj), int(tk)], filtration=float(tr))
            del valid_tri, valid_radii, triangles

    del pts, dist_mat, radius_mat, edge_i, edge_j
    return st

# ═══════════════════════════════════════════════════════════════════
# Boundary Matrix & Reduction
# ═══════════════════════════════════════════════════════════════════
def divide_filtration(st):
    simplex_filt_pairs = [(tuple(sorted(s)), f) for s, f in st.get_filtration()]
    return [p[0] for p in simplex_filt_pairs], [p[1] for p in simplex_filt_pairs]

def _build_boundary(simplices):
    sf_to_idx = {s: i for i, s in enumerate(simplices)}
    boundary = []
    for s in simplices:
        if len(s) <= 1:
            boundary.append(set())
        else:
            rows = set()
            for j in range(len(s)):
                face = s[:j] + s[j+1:]
                if face in sf_to_idx:
                    rows.add(sf_to_idx[face])
            boundary.append(rows)
    return boundary

def _reduce_with_V(columns):
    m = len(columns)
    R = [set(col) for col in columns]
    V = [{i} for i in range(m)]
    low = [-1] * m
    pivot_of_row = {}
    for i in range(m):
        while R[i]:
            li = max(R[i])
            if li in pivot_of_row:
                owner = pivot_of_row[li]
                R[i] ^= R[owner]
                V[i] ^= V[owner]
            else:
                pivot_of_row[li] = i
                low[i] = li
                break
        else:
            low[i] = -1
    return R, low, V

# ═══════════════════════════════════════════════════════════════════
# Six-pack Barcode 계산
# ═══════════════════════════════════════════════════════════════════
def compute_all_barcodes(A, B, max_radius=5):
    total = np.concatenate([A, B], axis=0)
    a = len(A)

    st = compute_Cech_cpu(total, max_radius=max_radius)
    del total

    simplices, filt = divide_filtration(st)
    del st
    gc.collect()

    m = len(simplices)
    in_L = [all(v < a for v in s) for s in simplices]
    idx_L = [i for i, b in enumerate(in_L) if b]
    idx_KmL = [i for i, b in enumerate(in_L) if not b]
    set_idx_KmL = set(idx_KmL)
    g2L = {g: pos for pos, g in enumerate(idx_L)}

    Df = _build_boundary(simplices)
    Rf, lowf, Vf = _reduce_with_V(Df)

    boundary_L = [{g2L[r] for r in Df[g_idx] if r in g2L} for g_idx in idx_L]
    Rg, lowg, Vg = _reduce_with_V(boundary_L)
    del boundary_L

    row_order = idx_L + idx_KmL
    row_remap = {g: i for i, g in enumerate(row_order)}
    inv_row_remap = {i: g for g, i in row_remap.items()}
    del row_order

    Dim = [{row_remap[r] for r in Df[col_idx]} for col_idx in range(m)]
    Rim, lowim, _ = _reduce_with_V(Dim)
    del Dim, _
    gc.collect()

    Vim = [{row_remap[r] for r in Vf[col_idx]} for col_idx in range(m)]
    del Vf
    gc.collect()

    cycle_cols = [i for i in range(m) if not Rim[i]]
    Dker = [Vim[c] for c in cycle_cols]
    if Dker:
        _, lowker, _ = _reduce_with_V(Dker)
        del _
    else:
        lowker = []
    del Dker
    cycle_pos = {c: pos for pos, c in enumerate(cycle_cols)}
    del cycle_cols, Vim
    gc.collect()

    Dcok = []
    for i in range(m):
        if in_L[i]:
            jL = g2L[i]
            if not Rg[jL]:
                Dcok.append({idx_L[pos] for pos in Vg[jL]})
                continue
        Dcok.append(set(Df[i]))
    _, lowcok, _ = _reduce_with_V(Dcok)
    del Dcok, _, Vg
    gc.collect()

    KmL_pos = {g: pos for pos, g in enumerate(idx_KmL)}
    Drel = [{KmL_pos[r] for r in Df[i] if r in set_idx_KmL} for i in idx_KmL]
    del Df
    gc.collect()

    Rrel, lowrel, _ = _reduce_with_V(Drel)
    del Drel, _, lowrel, KmL_pos
    gc.collect()

    def _format(bars_dict):
        out = {}
        for p in [0, 1]:
            if p in bars_dict and bars_dict[p]:
                arr = np.array(bars_dict[p])
                out[p] = arr[np.lexsort((arr[:, 1], arr[:, 0]))]
            else:
                out[p] = np.empty((0, 2))
        return out

    image_bars = defaultdict(list)
    for tau in range(m):
        if not Rf[tau] or lowim[tau] == -1:
            continue
        sigma = inv_row_remap[lowim[tau]]
        if sigma in g2L:
            b, d = filt[sigma], filt[tau]
            if b != d:
                p = len(simplices[sigma]) - 1
                image_bars[p].append((b, d))

    kernel_bars = defaultdict(list)
    for tau in idx_L:
        jL = g2L[tau]
        if not Rg[jL] or Rf[tau] or tau not in cycle_pos:
            continue
        lc = cycle_pos[tau]
        if lc >= len(lowker):
            continue
        ll = lowker[lc]
        if ll == -1:
            continue
        sigma = inv_row_remap[ll]
        if in_L[sigma]:
            continue
        b, d = filt[sigma], filt[tau]
        if b != d:
            p = len(simplices[sigma]) - 2
            if p >= 0:
                kernel_bars[p].append((b, d))
    del lowker, cycle_pos

    cok_bars = defaultdict(list)
    for tau in range(m):
        if not Rf[tau] or lowim[tau] == -1:
            continue
        if inv_row_remap[lowim[tau]] not in set_idx_KmL:
            continue
        lc = lowcok[tau]
        if lc == -1:
            continue
        b, d = filt[lc], filt[tau]
        if b != d:
            p = len(simplices[lc]) - 1
            cok_bars[p].append((b, d))
    del lowcok

    rel_bars = defaultdict(list)
    for pos in range(len(idx_KmL)):
        if not Rrel[pos]:
            continue
        sigma_local = max(Rrel[pos])
        sigma = idx_KmL[sigma_local]
        tau = idx_KmL[pos]
        b, d = filt[sigma], filt[tau]
        if abs(b - d) > 1e-12:
            p = len(simplices[sigma]) - 1
            rel_bars[p].append((b, d))

    del Rf, lowf, Rg, lowg, Rim, lowim, Rrel
    del in_L, idx_L, idx_KmL, set_idx_KmL, g2L
    del row_remap, inv_row_remap, simplices, filt
    gc.collect()

    result = {
        'image': _format(image_bars),
        'kernel': _format(kernel_bars),
        'cokernel': _format(cok_bars),
        'relative': _format(rel_bars),
    }
    del image_bars, kernel_bars, cok_bars, rel_bars
    gc.collect()
    return result

# ═══════════════════════════════════════════════════════════════════
# Persistence & PI
# ═══════════════════════════════════════════════════════════════════
def compute_Persistence_barcode(A, max_radius=5):
    fil_A = compute_Cech_cpu(A, max_radius=max_radius)
    fil_A.persistence()
    bar_A = {}
    for dim in [0, 1]:
        bars = fil_A.persistence_intervals_in_dimension(dim)
        bars = [b for b in bars if b[1] != np.inf and b[1] - b[0] > 1e-5]
        bar_A[dim] = np.array(bars) if bars else np.empty((0, 2))
    del fil_A
    return bar_A

def compute_PIs(barcodes, max_eps=10, px_res=0.1, sigma=0.05, normalization=False):
    for key in barcodes:
        if len(barcodes[key]) == 0:
            barcodes[key] = np.zeros((0, 2))

    vector = {}

    pers_imager_h0 = PersistenceImager()
    pers_imager_h0.pixel_size = px_res
    pers_imager_h0.birth_range = (0, 1)
    pers_imager_h0.pers_range = (0, max_eps)
    pers_imager_h0.weight = weights.persistence
    pers_imager_h0.weight_params = {'n': 1}
    pers_imager_h0.kernel_params = {'sigma': [[sigma, 0], [0, sigma]]}
    bars_h0 = np.array(barcodes[0])
    if len(bars_h0) > 0:
        img_h0 = pers_imager_h0.transform(bars_h0, skew=False)
    else:
        img_h0 = np.zeros((int(1/px_res), int(max_eps/px_res)))
    img0_1d = np.mean(img_h0, axis=0)
    del img_h0, pers_imager_h0

    pers_imager_h1 = PersistenceImager()
    pers_imager_h1.pixel_size = px_res
    pers_imager_h1.birth_range = (0, max_eps)
    pers_imager_h1.pers_range = (0, max_eps / 2)
    pers_imager_h1.weight = weights.persistence
    pers_imager_h1.weight_params = {'n': 1}
    pers_imager_h1.kernel_params = {'sigma': [[sigma, 0], [0, sigma]]}
    bars_h1 = np.array(barcodes[1])
    if len(bars_h1) > 0:
        img_h1 = pers_imager_h1.transform(bars_h1, skew=True)
    else:
        img_h1 = np.zeros((int(max_eps/px_res), int((max_eps/2)/px_res)))
    del pers_imager_h1

    if normalization:
        vector[0] = img0_1d / np.max(img0_1d) if np.max(img0_1d) > 0 else img0_1d
        vector[1] = img_h1.flatten() / np.max(img_h1) if np.max(img_h1) > 0 else img_h1.flatten()
    else:
        vector[0] = img0_1d
        vector[1] = img_h1.flatten()

    del img_h1
    return vector

# ═══════════════════════════════════════════════════════════════════
# 단일 샘플 처리
# ═══════════════════════════════════════════════════════════════════
def process_single_sample(args):
    idx, params, base_dir, out_dir, max_radius = args
    save_path = os.path.join(out_dir, f'Sixpack_Cech_{idx}.npz')

    if os.path.exists(save_path):
        return idx, 'skip'

    try:
        folder = os.path.join(base_dir, f'ParamSweep_{idx}_Output')
        pos_file = os.path.join(folder, f'Pos_{params[0]:.2f}_{params[1]:.2f}_{params[2]:.2f}.dat')
        types_file = os.path.join(folder, f'Types_{params[0]:.2f}_{params[1]:.2f}_{params[2]:.2f}.dat')

        types = np.loadtxt(types_file, dtype=int)
        positions = np.loadtxt(pos_file, delimiter=',')
        A = positions[types == 1]
        B = positions[types == 2]
        del types, positions

        six_pack_A2B = compute_all_barcodes(A, B, max_radius=max_radius)
        gc.collect()
        six_pack_B2A = compute_all_barcodes(B, A, max_radius=max_radius)
        gc.collect()

        total_pts = np.concatenate([A, B])
        PB_total = compute_Persistence_barcode(total_pts, max_radius=max_radius)
        del total_pts

        PB_A = compute_Persistence_barcode(A, max_radius=max_radius)
        PB_B = compute_Persistence_barcode(B, max_radius=max_radius)
        del A, B

        six_pack_A2B.update({'complex': PB_total, 'sub_complex': PB_A})
        six_pack_B2A.update({'complex': PB_total, 'sub_complex': PB_B})
        del PB_total, PB_A, PB_B

        PI_A2B = {}
        for key in list(six_pack_A2B.keys()):
            PI_A2B[key] = compute_PIs(six_pack_A2B[key], normalization=False)
            del six_pack_A2B[key]
        del six_pack_A2B

        PI_B2A = {}
        for key in list(six_pack_B2A.keys()):
            PI_B2A[key] = compute_PIs(six_pack_B2A[key], normalization=False)
            del six_pack_B2A[key]
        del six_pack_B2A
        gc.collect()

        np.savez_compressed(save_path, PI_A2B, PI_B2A)
        del PI_A2B, PI_B2A
        gc.collect()
        return idx, 'done'
    except Exception as e:
        return idx, f'error: {e}'

# ═══════════════════════════════════════════════════════════════════
# 메인 실행
# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('=' * 70)
    print('Phase 5: Six-pack (Čech) — 멀티프로세싱 병렬 계산')
    print('=' * 70)
    print(f'CPU 코어: {cpu_count()}개')
    print(f'워커 수: {N_WORKERS}개')
    print(f'총 샘플: 512개')
    print(f'출력 디렉토리: {OUT_DIR}')
    print('=' * 70)

    # 처리할 작업 목록
    tasks = []
    for idx in range(1, 513):
        save_path = os.path.join(OUT_DIR, f'Sixpack_Cech_{idx}.npz')
        if not os.path.exists(save_path):
            params = PARAM_LIST[idx - 1]
            tasks.append((idx, params, BASE_DIR, OUT_DIR, MAX_RADIUS))

    already_done = 512 - len(tasks)
    print(f'이미 완료: {already_done}개')
    print(f'처리 예정: {len(tasks)}개')
    print('=' * 70)

    if not tasks:
        print('모든 샘플이 이미 계산되어 있습니다!')
    else:
        start_time = time.time()
        completed = 0
        errors = 0

        with Pool(processes=N_WORKERS) as pool:
            for idx, status in pool.imap_unordered(process_single_sample, tasks):
                completed += 1
                elapsed = time.time() - start_time
                eta = (elapsed / completed) * (len(tasks) - completed)

                if status == 'done':
                    symbol = '✓'
                elif status == 'skip':
                    symbol = '○'
                else:
                    symbol = '✗'
                    errors += 1

                print(f'[{completed:3d}/{len(tasks)}] {symbol} #{idx:3d} | '
                      f'경과: {elapsed:5.0f}s | ETA: {eta:5.0f}s | '
                      f'RAM: {psutil.Process().memory_info().rss/1024/1024:.0f}MB')

        total_time = time.time() - start_time
        print('=' * 70)
        print(f'완료! 총 {total_time:.1f}초 ({total_time/60:.1f}분)')
        print(f'성공: {completed - errors}개 | 에러: {errors}개')
        print('=' * 70)
