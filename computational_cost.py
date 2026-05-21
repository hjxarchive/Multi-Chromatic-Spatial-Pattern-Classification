"""
Computational Cost Experiment
Vectorization/ 노트북 코드를 그대로 사용하여 각 Descriptor별
wall time + peak memory 측정

Sources:
  Ord_PI        ← Vectorization/Ordinary PI.ipynb
  Sixpack_Rips  ← Vectorization/Six-pack (Rips).ipynb
  Inter_PI      ← Vectorization/Mixup barcode (1).ipynb
  3D_PI         ← Vectorization/Mixup barcode (1).ipynb
  Sixpack_Chroma← Vectorization/Six-pack (chromatic_tda) (1).ipynb
"""

import os, glob, time, tracemalloc, warnings, gc
import numpy as np
from collections import defaultdict
from typing import Dict
from scipy.stats import norm
from gudhi import RipsComplex
from persim import PersistenceImager
import persim.images_weights as weights
import chromatic_tda as chro
warnings.filterwarnings('ignore')

BASE   = "/Users/hjxarchive/Multi-Chromatic-Spatial-Pattern-Classification"
IN_DIR = os.path.join(BASE, "Data", "ParamSweep_input")
N_SAMPLES = 5
RNG       = np.random.default_rng(42)

# ─────────────────────────────────────────────────────────────────────────────
# 데이터 로딩
# ─────────────────────────────────────────────────────────────────────────────
def load_sample(folder):
    pos   = np.loadtxt(glob.glob(os.path.join(folder, "Pos_*.dat"))[0],  delimiter=",")
    types = np.loadtxt(glob.glob(os.path.join(folder, "Types_*.dat"))[0], dtype=int)
    return pos[types == 1], pos[types == 2]

# ─────────────────────────────────────────────────────────────────────────────
# [Ordinary PI.ipynb] — 공통 함수
# ─────────────────────────────────────────────────────────────────────────────
def compute_Rips(points, max_edge=10):
    rips = RipsComplex(points=points, max_edge_length=max_edge)
    st = rips.create_simplex_tree(max_dimension=2)
    return st

def compute_Persistence_barcode(A):
    fil_A = compute_Rips(A)
    fil_A.persistence()
    bar_a0 = fil_A.persistence_intervals_in_dimension(0)
    bar_A0 = [bar for bar in bar_a0 if bar[1] != np.inf]
    bar_a1 = fil_A.persistence_intervals_in_dimension(1)
    bar_A1 = [bar for bar in bar_a1 if bar[1] != np.inf]
    bar0 = [[bar[0], bar[1]] for bar in bar_A0 if bar[1]-bar[0] > 1e-5]
    bar1 = [[bar[0], bar[1]] for bar in bar_A1 if bar[1]-bar[0] > 1e-5]
    return {0: bar0, 1: bar1}

def compute_PIs_ord(barcodes, max_eps=10, px_res=0.1, sigma=0.05, normalization=False):
    """Ordinary PI.ipynb의 compute_PIs"""
    for key in list(barcodes.keys()):
        if len(barcodes[key]) == 0:
            barcodes[key] = np.zeros((0, 2))
    vector = {}
    pers_imager_h0 = PersistenceImager()
    pers_imager_h0.pixel_size = px_res
    pers_imager_h0.birth_range = (0, 0.01)
    pers_imager_h0.pers_range = (0, max_eps)
    pers_imager_h0.weight = weights.persistence
    pers_imager_h0.weight_params = {'n': 1}
    pers_imager_h0.kernel_params = {'sigma': [[sigma, 0], [0, sigma]]}
    bars_h0 = np.array(barcodes[0])
    if len(bars_h0) > 0:
        img_h0 = pers_imager_h0.transform(bars_h0, skew=True)
    else:
        img_h0 = np.zeros((int(1/px_res), int(max_eps/px_res)))
    img0_1d = np.mean(img_h0, axis=0)
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
    vector[0] = img0_1d
    vector[1] = img_h1.flatten()
    return vector

def run_Ord_PI(A, B):
    total = np.concatenate([A, B], axis=0)
    PB_total = compute_Persistence_barcode(total)
    PB_A     = compute_Persistence_barcode(A)
    PB_B     = compute_Persistence_barcode(B)
    PI_total = compute_PIs_ord(PB_total)
    PI_A     = compute_PIs_ord(PB_A)
    PI_B     = compute_PIs_ord(PB_B)
    return np.concatenate([PI_A[0], PI_A[1], PI_B[0], PI_B[1]])

# ─────────────────────────────────────────────────────────────────────────────
# [Six-pack (Rips).ipynb]
# ─────────────────────────────────────────────────────────────────────────────
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

def compute_all_barcodes(A, B, max_edge=10):
    total = np.concatenate([A, B], axis=0)
    a = len(A)
    st = compute_Rips(total, max_edge=10)
    simplices, filt = divide_filtration(st)
    m = len(simplices)
    in_L        = [all(v < a for v in s) for s in simplices]
    idx_L       = [i for i, b in enumerate(in_L) if b]
    idx_KmL     = [i for i, b in enumerate(in_L) if not b]
    set_idx_KmL = set(idx_KmL)
    g2L         = {g: pos for pos, g in enumerate(idx_L)}
    Df = _build_boundary(simplices)
    Rf, lowf, Vf = _reduce_with_V(Df)
    boundary_L = [{g2L[r] for r in Df[g_idx] if r in g2L} for g_idx in idx_L]
    Rg, lowg, Vg = _reduce_with_V(boundary_L)
    row_order     = idx_L + idx_KmL
    row_remap     = {g: i for i, g in enumerate(row_order)}
    inv_row_remap = {i: g for g, i in row_remap.items()}
    Dim = [{row_remap[r] for r in Df[col_idx]} for col_idx in range(m)]
    Rim, lowim, _ = _reduce_with_V(Dim)
    Vim = [{row_remap[r] for r in Vf[col_idx]} for col_idx in range(m)]
    cycle_cols = [i for i in range(m) if not Rim[i]]
    Dker = [Vim[c] for c in cycle_cols]
    if Dker:
        _, lowker, _ = _reduce_with_V(Dker)
    else:
        lowker = []
    cycle_pos = {c: pos for pos, c in enumerate(cycle_cols)}
    Dcok = []
    for i in range(m):
        if in_L[i]:
            jL = g2L[i]
            if not Rg[jL]:
                Dcok.append({idx_L[pos] for pos in Vg[jL]})
                continue
        Dcok.append(set(Df[i]))
    _, lowcok, _ = _reduce_with_V(Dcok)
    KmL_pos = {g: pos for pos, g in enumerate(idx_KmL)}
    Drel    = [{KmL_pos[r] for r in Df[i] if r in set_idx_KmL} for i in idx_KmL]
    Rrel, lowrel, _ = _reduce_with_V(Drel)

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
                image_bars[len(simplices[sigma]) - 1].append((b, d))

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
            cok_bars[len(simplices[lc]) - 1].append((b, d))

    rel_bars = defaultdict(list)
    for pos in range(len(idx_KmL)):
        if not Rrel[pos]:
            continue
        sigma_local = max(Rrel[pos])
        sigma = idx_KmL[sigma_local]
        tau   = idx_KmL[pos]
        b, d  = filt[sigma], filt[tau]
        if abs(b - d) > 1e-12:
            rel_bars[len(simplices[sigma]) - 1].append((b, d))

    return {
        'image':    _format(image_bars),
        'kernel':   _format(kernel_bars),
        'cokernel': _format(cok_bars),
        'relative': _format(rel_bars),
    }

def compute_PIs_sixpack(barcodes, max_eps=10, px_res=0.1, sigma=0.05, normalization=False):
    """Six-pack (Rips).ipynb의 compute_PIs (H0: birth_range=(0,1), skew=False)"""
    for key in list(barcodes.keys()):
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
    vector[0] = img0_1d
    vector[1] = img_h1.flatten()
    return vector

def run_Sixpack_Rips(A, B):
    total = np.concatenate([A, B], axis=0)
    PB_total = compute_Persistence_barcode(total)
    PB_A     = compute_Persistence_barcode(A)
    PB_B     = compute_Persistence_barcode(B)

    sp_AB = compute_all_barcodes(A, B)
    sp_AB.update({'complex': PB_total, 'sub_complex': PB_A})
    sp_BA = compute_all_barcodes(B, A)
    sp_BA.update({'complex': PB_total, 'sub_complex': PB_B})

    feats = []
    for sp in [sp_AB, sp_BA]:
        for key in ['image', 'kernel', 'cokernel', 'relative', 'complex', 'sub_complex']:
            pi = compute_PIs_sixpack(sp[key])
            feats.extend(pi[0]); feats.extend(pi[1])
    return np.array(feats)

# ─────────────────────────────────────────────────────────────────────────────
# [Mixup barcode (1).ipynb]
# ─────────────────────────────────────────────────────────────────────────────
def _compute_Rips_mixup(points, max_edge=10, max_dim=2):
    rips = RipsComplex(points=points, max_edge_length=max_edge)
    return rips.create_simplex_tree(max_dimension=max_dim)

def extract_filtration(st):
    pairs = [(tuple(sorted(s)), f) for s, f in st.get_filtration()]
    return [p[0] for p in pairs], [p[1] for p in pairs]

def _reduce_matrix(columns, n):
    R = [set(col) for col in columns]
    low = [-1] * n
    pivot_to_col = {}
    for j in range(n):
        while R[j]:
            pivot = max(R[j])
            if pivot in pivot_to_col:
                R[j] ^= R[pivot_to_col[pivot]]
            else:
                pivot_to_col[pivot] = j
                low[j] = pivot
                break
        else:
            low[j] = -1
    return R, low

def compute_mixup_barcode(A: np.ndarray, B: np.ndarray, max_edge: float = 10, max_dim: int = 1) -> Dict:
    total = np.concatenate([A, B], axis=0)
    a = len(A)
    st = _compute_Rips_mixup(total, max_edge, max_dim=max_dim+1)
    simplices, filt = extract_filtration(st)
    n = len(simplices)
    simplex_dims = [len(s) - 1 for s in simplices]
    sf_to_idx = {s: i for i, s in enumerate(simplices)}
    in_L = [all(v < a for v in s) for s in simplices]
    idx_L   = [i for i, b in enumerate(in_L) if b]
    idx_KmL = [i for i, b in enumerate(in_L) if not b]
    BK_original = []
    for s in simplices:
        if len(s) <= 1:
            BK_original.append(set())
        else:
            rows = set()
            for j in range(len(s)):
                face = s[:j] + s[j+1:]
                if face in sf_to_idx:
                    rows.add(sf_to_idx[face])
            BK_original.append(rows)
    row_order = idx_L + idx_KmL
    old_to_new_row = {old: new for new, old in enumerate(row_order)}
    n_L = len(idx_L)
    BK = [{old_to_new_row[r] for r in BK_original[col_idx]} for col_idx in range(n)]
    BL = []
    for col_idx in range(n):
        if in_L[col_idx]:
            BL.append({r for r in BK[col_idx] if r < n_L})
        else:
            BL.append(set())
    RL, lowL = _reduce_matrix(BL, n)
    RK, lowK = _reduce_matrix(BK, n)
    pivotL_to_col = {lowL[j]: j for j in idx_L if lowL[j] != -1}
    pivotK_to_col = {lowK[j]: j for j in range(n) if lowK[j] != -1}
    mixup_triples = defaultdict(list)
    for sigma in idx_L:
        if RL[sigma]:
            continue
        dim = simplex_dims[sigma]
        if dim > max_dim:
            continue
        sigma_row = old_to_new_row[sigma]
        birth = filt[sigma]
        tau = pivotL_to_col.get(sigma_row, None)
        death = filt[tau] if tau is not None else np.inf
        tau_prime = pivotK_to_col.get(sigma_row, None)
        death_prime = filt[tau_prime] if tau_prime is not None else np.inf
        if not np.isinf(death) and (np.isinf(death_prime) or death_prime > death):
            death_prime = death
        if np.isinf(death) or abs(death - birth) > 1e-10:
            mixup_triples[dim].append((birth, death_prime, death))
    result = {}
    for dim in sorted(mixup_triples.keys()):
        triples = mixup_triples[dim]
        arr = np.array(triples)
        result[dim] = arr[np.argsort(arr[:, 0])] if len(arr) else np.empty((0, 3))
    for d in range(max_dim + 1):
        if d not in result:
            result[d] = np.empty((0, 3))
    return result

def compute_Interaction_PIs(barcodes, max_eps=10, px_res=0.1, sigma=0.05, normalization=False):
    vector = {}
    def make_mixup_weight(mixup_weights):
        def weight(birth, persistence, **kwargs):
            return mixup_weights
        return weight
    pers_imager_h0 = PersistenceImager()
    pers_imager_h0.pixel_size = px_res
    pers_imager_h0.birth_range = (0, 0.01)
    pers_imager_h0.pers_range = (0, max_eps)
    pers_imager_h0.kernel_params = {'sigma': [[sigma, 0], [0, sigma]]}
    bars_h0 = np.asarray(barcodes.get(0, np.zeros((0, 3))))
    if len(bars_h0) > 0:
        b, d_prime, d = bars_h0[:,0], bars_h0[:,1], bars_h0[:,2]
        mask = np.isfinite(b) & np.isfinite(d_prime) & np.isfinite(d)
        b, d_prime, d = b[mask], d_prime[mask], d[mask]
        if len(b) > 0:
            pers_imager_h0.weight = make_mixup_weight(d - d_prime)
            img_h0 = pers_imager_h0.transform(np.stack([b, d], axis=1), skew=True)
        else:
            img_h0 = np.zeros((int(1/px_res), int(max_eps/px_res)))
    else:
        img_h0 = np.zeros((int(1/px_res), int(max_eps/px_res)))
    img0_1d = np.mean(img_h0, axis=0)
    pers_imager_h1 = PersistenceImager()
    pers_imager_h1.pixel_size = px_res
    pers_imager_h1.birth_range = (0, max_eps)
    pers_imager_h1.pers_range = (0, max_eps / 2)
    pers_imager_h1.kernel_params = {'sigma': [[sigma, 0], [0, sigma]]}
    bars_h1 = np.asarray(barcodes.get(1, np.zeros((0, 3))))
    if len(bars_h1) > 0:
        b, d_prime, d = bars_h1[:,0], bars_h1[:,1], bars_h1[:,2]
        mask = np.isfinite(b) & np.isfinite(d_prime) & np.isfinite(d)
        b, d_prime, d = b[mask], d_prime[mask], d[mask]
        if len(b) > 0:
            pers_imager_h1.weight = make_mixup_weight(d - d_prime)
            img_h1 = pers_imager_h1.transform(np.stack([b, d], axis=1), skew=True)
        else:
            img_h1 = np.zeros((int(max_eps/px_res), int((max_eps/2)/px_res)))
    else:
        img_h1 = np.zeros((int(max_eps/px_res), int((max_eps/2)/px_res)))
    vector[0] = img0_1d
    vector[1] = img_h1.flatten()
    return vector

def mixup_barcode_translation(mixup_barcode):
    translation = {}
    for i in range(2):
        translated = []
        for bar in mixup_barcode[i]:
            translated.append([bar[0], bar[1]-bar[0], bar[2]-bar[0]])
        translation[i] = translated
    return translation

def compute_3d_persistence_image(mixup_barcodes, resolution=20,
                                  ranges=((0,10),(0,10),(0,10)),
                                  bandwidth=0.4, weight_func=None, normalization=False):
    translated = mixup_barcode_translation(mixup_barcodes)
    if weight_func is None:
        weight_func = lambda p: 1.0
    x_grid = np.linspace(ranges[0][0], ranges[0][1], resolution)
    y_grid = np.linspace(ranges[1][0], ranges[1][1], resolution)
    z_grid = np.linspace(ranges[2][0], ranges[2][1], resolution)
    persistence_vectors = {}
    for i in range(2):
        barcode = translated[i]
        if i == 0:
            persistence_image = np.zeros((resolution, resolution))
            for point in barcode:
                w = weight_func(point)
                gy = norm.pdf(y_grid, loc=point[1], scale=bandwidth)
                gz = norm.pdf(z_grid, loc=point[2], scale=bandwidth)
                persistence_image += w * np.outer(gy, gz)
        else:
            persistence_image = np.zeros((resolution, resolution, resolution))
            for point in barcode:
                w = weight_func(point)
                gx = norm.pdf(x_grid, loc=point[0], scale=bandwidth)
                gy = norm.pdf(y_grid, loc=point[1], scale=bandwidth)
                gz = norm.pdf(z_grid, loc=point[2], scale=bandwidth)
                persistence_image += w * np.einsum('i,j,k->ijk', gx, gy, gz)
        persistence_vectors[i] = persistence_image.flatten()
    return persistence_vectors

def run_Inter_PI(A, B):
    mb_AB = compute_mixup_barcode(A, B)
    mb_BA = compute_mixup_barcode(B, A)
    pi_AB = compute_Interaction_PIs(mb_AB)
    pi_BA = compute_Interaction_PIs(mb_BA)
    return np.concatenate([pi_AB[0], pi_AB[1], pi_BA[0], pi_BA[1]])

def run_3D_PI(A, B):
    mb_AB = compute_mixup_barcode(A, B)
    mb_BA = compute_mixup_barcode(B, A)
    pi_AB = compute_3d_persistence_image(mb_AB)
    pi_BA = compute_3d_persistence_image(mb_BA)
    return np.concatenate([pi_AB[0], pi_AB[1], pi_BA[0], pi_BA[1]])

# ─────────────────────────────────────────────────────────────────────────────
# [Six-pack (chromatic_tda) (1).ipynb]
# ─────────────────────────────────────────────────────────────────────────────
def convert_into_diagram(diagram):
    diagrams = {}
    for dim, pairs in diagram.items():
        filtered = [(float(b), float(d)) for (b, d) in pairs if np.isfinite(d)]
        diagrams[dim] = np.array(filtered) if filtered else np.zeros((0, 2))
    return diagrams

def convert_six_pack_to_diagram(six_pack):
    return {key: convert_into_diagram(dgm) for key, dgm in six_pack.items()}

def compute_six_pack_diagrams(points, labels, max_edge=10):
    chro_alpha = chro.ChromaticAlphaComplex(points, labels, max_alpha=max_edge)
    simplicial_complex = chro_alpha.get_simplicial_complex(
        sub_complex='0', full_complex='all', relative=None)
    six_pack = simplicial_complex.bars_six_pack()
    return convert_six_pack_to_diagram(six_pack)

def compute_PIs_chroma(barcodes, max_eps=10, px_res=0.1, sigma=0.025, normalization=False):
    """Six-pack (chromatic_tda).ipynb의 compute_PIs (sigma=0.025)"""
    if 0 not in barcodes: barcodes[0] = np.zeros((0, 2))
    if 1 not in barcodes: barcodes[1] = np.zeros((0, 2))
    for key in list(barcodes.keys()):
        if len(barcodes[key]) == 0:
            barcodes[key] = np.zeros((0, 2))
    vector = {}
    pers_imager_h0 = PersistenceImager()
    pers_imager_h0.pixel_size = px_res
    pers_imager_h0.birth_range = (0, 0.01)
    pers_imager_h0.pers_range = (0, max_eps)
    pers_imager_h0.weight = weights.persistence
    pers_imager_h0.weight_params = {'n': 1}
    pers_imager_h0.kernel_params = {'sigma': [[sigma, 0], [0, sigma]]}
    bars_h0 = np.array(barcodes.get(0, np.zeros((0, 2))))
    if len(bars_h0) > 0:
        img_h0 = pers_imager_h0.transform(bars_h0, skew=True)
    else:
        img_h0 = np.zeros((int(1/px_res), int(max_eps/px_res)))
    img0_1d = np.mean(img_h0, axis=0)
    pers_imager_h1 = PersistenceImager()
    pers_imager_h1.pixel_size = px_res
    pers_imager_h1.birth_range = (0, max_eps)
    pers_imager_h1.pers_range = (0, max_eps / 2)
    pers_imager_h1.weight = weights.persistence
    pers_imager_h1.weight_params = {'n': 1}
    pers_imager_h1.kernel_params = {'sigma': [[sigma, 0], [0, sigma]]}
    bars_h1 = np.array(barcodes.get(1, np.zeros((0, 2))))
    if len(bars_h1) > 0:
        img_h1 = pers_imager_h1.transform(bars_h1, skew=True)
    else:
        img_h1 = np.zeros((int(max_eps/px_res), int((max_eps/2)/px_res)))
    vector[0] = img0_1d
    vector[1] = img_h1.flatten()
    return vector

def run_Sixpack_Chroma(A, B):
    points = np.concatenate([A, B], axis=0)
    labels_AB = np.concatenate([np.zeros(len(A)), np.ones(len(B))])
    labels_BA = 1 - labels_AB

    sp_AB = compute_six_pack_diagrams(points, labels_AB)
    sp_BA = compute_six_pack_diagrams(points, labels_BA)

    feats = []
    for sp in [sp_AB, sp_BA]:
        for key in sp:
            pi = compute_PIs_chroma(sp[key])
            feats.extend(pi[0]); feats.extend(pi[1])
    return np.array(feats)

# ─────────────────────────────────────────────────────────────────────────────
# 측정 유틸
# ─────────────────────────────────────────────────────────────────────────────
def measure(fn, *args):
    gc.collect()
    tracemalloc.start()
    t0  = time.perf_counter()
    out = fn(*args)
    wall = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return wall, peak / 1e6, out

# ─────────────────────────────────────────────────────────────────────────────
# 샘플 선택 및 실험
# ─────────────────────────────────────────────────────────────────────────────
all_folders = sorted(glob.glob(os.path.join(IN_DIR, "ParamSweep_*_Output")))
indices     = RNG.choice(len(all_folders), size=N_SAMPLES, replace=False)
sample_dirs = [all_folders[i] for i in sorted(indices)]

print(f"샘플 수: {N_SAMPLES} / 전체 {len(all_folders)}")
print(f"샘플 idx: {sorted(indices.tolist())}\n")

DESCRIPTORS = {
    'Ord_PI':        run_Ord_PI,
    'Inter_PI':      run_Inter_PI,
    '3D_PI':         run_3D_PI,
    'Sixpack_Rips':  run_Sixpack_Rips,
    'Sixpack_Chroma': run_Sixpack_Chroma,
}

SEP = "=" * 72
results = {name: {'times': [], 'mems': [], 'dims': []} for name in DESCRIPTORS}

for di, folder in enumerate(sample_dirs):
    try:
        A, B = load_sample(folder)
    except Exception as e:
        print(f"  [SKIP] {os.path.basename(folder)}: {e}"); continue

    print(f"[{di+1:2d}/{N_SAMPLES}] {os.path.basename(folder)}"
          f"  |A|={len(A)}  |B|={len(B)}")

    for name, fn in DESCRIPTORS.items():
        try:
            wall, mem, out = measure(fn, A, B)
            dim = out.shape[0] if isinstance(out, np.ndarray) else -1
            results[name]['times'].append(wall)
            results[name]['mems'].append(mem)
            results[name]['dims'].append(dim)
            print(f"    {name:<16} {wall:7.2f}s  {mem:8.2f} MB  dim={dim}")
        except Exception as e:
            print(f"    {name:<16} ERROR: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 집계
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print(f"집계 결과  (mean ± std, N={N_SAMPLES})")
print(f"  {'Descriptor':<16}  {'Time mean':>10}  {'Time std':>9}  "
      f"{'Mem mean':>9}  {'Mem std':>8}  {'Vec dim':>8}")
print("  " + "-" * 68)

for name, r in results.items():
    ts = np.array(r['times']); ms = np.array(r['mems'])
    dim = r['dims'][0] if r['dims'] else -1
    print(f"  {name:<16}  {ts.mean():>9.2f}s  {ts.std():>8.2f}s  "
          f"{ms.mean():>8.2f}MB  {ms.std():>7.2f}MB  {dim:>8}")

print(f"\n{SEP}")
print("상대 비교 (Ord_PI 기준 배율)")
base_t = np.mean(results['Ord_PI']['times'])
base_m = np.mean(results['Ord_PI']['mems'])
for name, r in results.items():
    ts = np.array(r['times']); ms = np.array(r['mems'])
    print(f"  {name:<16}  time ×{ts.mean()/base_t:6.1f}  mem ×{ms.mean()/base_m:6.1f}")

# CSV 저장
import csv
out_csv = os.path.join(BASE, "Final_Results(0521)", "computational_cost_raw.csv")
with open(out_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['descriptor', 'sample', 'wall_sec', 'peak_mb', 'vec_dim'])
    for name, r in results.items():
        for i, (t, m, d) in enumerate(zip(r['times'], r['mems'], r['dims'])):
            w.writerow([name, i, f'{t:.4f}', f'{m:.3f}', d])
print(f"\nRaw 데이터 저장: {out_csv}")
