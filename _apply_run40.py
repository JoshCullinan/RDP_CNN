"""Apply run #40 cell edits: layer top-10 RecombIdentifyStats (A)-row
features as scalar broadcast channels on top of run #39's 33-channel layout.

Top-10 by user-supplied logistic-regression feature importance (importance
threshold >= 0.157):
  dMax(A), BadDists(A), SubPhPrScore(A), SimScore(A), PhPrScore(A),
  PhPrScore2(A), RCompatS(A), ListCorr3(A), PhPrScore3(A), OUList(A)

Each channel = z-score-normalized feature value (using train-set stats from
cache/recid_normalizer.json) broadcast across all MAX_SEQ_LEN positions.
Missing event / hypothesis row -> all-zero channel.

Total: N_INPUT_CHANNELS 33 -> 43.

Apply ONLY after run #39 has been evaluated and decision made to layer in.
"""
import json
from pathlib import Path

NB = Path("CNN.ipynb")
nb = json.loads(NB.read_text())

def get_cell(cid):
    for c in nb['cells']:
        if c.get('id') == cid:
            return c
    raise KeyError(cid)

def set_source(cid, src):
    cell = get_cell(cid)
    cell['source'] = src.splitlines(keepends=True)
    if cell['source'] and not cell['source'][-1].endswith('\n'):
        cell['source'][-1] += '\n'

# ============================================================
# cell-3: add RecombIdentifyStats top-N features
# ============================================================
cell3 = '''# Configuration

# Data paths
DATA_ROOT = Path("dataRaw")
TRAIN_DIRS = ["XML-1", "XML-2", "XML-3", "XML-4", "XML-5"]
TEST_DIR = "UnseenTestSet"

# Sequence encoding
MAX_SEQ_LEN = 32000
NUCLEOTIDES = ['A', 'T', 'G', 'C', '-']
N_CHANNELS = len(NUCLEOTIDES)
N_MAXCHI_WINDOWS = 4
# Run #36: +2 RDP consensus PredBPStart/End Gaussians
# Run #39 v2: +9 per-method confidence broadcast scalars
# Run #40: +10 RecombIdentifyStats (A)-row top-importance features as
#   z-score-normalized broadcast scalars. Top-10 by user-supplied logistic-
#   regression feature importance (threshold >= 0.157).
N_RDP_METHOD_CHANNELS = 9
N_RECID_CHANNELS = 10
N_INPUT_CHANNELS = (3 * N_CHANNELS + 3 + N_MAXCHI_WINDOWS + 2
                    + N_RDP_METHOD_CHANNELS + N_RECID_CHANNELS)  # 43
MAXCHI_WINDOWS = (50, 100, 200, 500)
RDP_PRED_SIGMA = 50
RDP_DROPOUT_P = 0.1
RDP_METHOD_NAMES = ('RDP', 'GENECONV', 'Bootscan', 'Maxchi', 'Chimaera',
                    'SiSscan', 'PhylPro', 'LARD', '3Seq')
RDP_METHOD_LOGP_CLIP = 30.0
# Run #40: top-10 RecombIdentifyStats (A) features ranked by importance.
RECID_FEATURES = ('dMax(A)', 'BadDists(A)', 'SubPhPrScore(A)', 'SimScore(A)',
                  'PhPrScore(A)', 'PhPrScore2(A)', 'RCompatS(A)', 'ListCorr3(A)',
                  'PhPrScore3(A)', 'OUList(A)')
# Channel-block boundaries.
RDP_BLOCK_START = 3 * N_CHANNELS + 3 + N_MAXCHI_WINDOWS  # 22
RDP_BLOCK_END = RDP_BLOCK_START + 2 + N_RDP_METHOD_CHANNELS  # 33
RECID_BLOCK_START = RDP_BLOCK_END  # 33
RECID_BLOCK_END = N_INPUT_CHANNELS  # 43

# Label generation (LEGACY)
LABEL_MODE = 'gaussian'
LABEL_SIGMA = 20
BP_WINDOW = 10
TOLERANCE = 200

# Top-K (LEGACY)
K_TOPK = 2
TOPK_TARGET_SIGMA = 5
EDGE_BUFFER = 50

# Training
BATCH_SIZE = 2
EPOCHS = 100
LR = 1e-4
VAL_SPLIT = 0.15

FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
POS_WEIGHT = 70.0

# Run #40: load z-score normalizer for RECID_FEATURES (computed offline).
import json as _json
_norm_path = Path('cache/recid_normalizer.json')
RECID_NORMALIZER = {}
if _norm_path.exists():
    _all_norm = _json.loads(_norm_path.read_text())
    RECID_NORMALIZER = {f: _all_norm[f] for f in RECID_FEATURES if f in _all_norm}

print(f"Max sequence length: {MAX_SEQ_LEN}")
print(f"Input channels: {N_INPUT_CHANNELS}")
print(f"  RDP block ({RDP_BLOCK_START}..{RDP_BLOCK_END}): "
      f"2 consensus + {N_RDP_METHOD_CHANNELS} method scalars")
print(f"  RECID block ({RECID_BLOCK_START}..{RECID_BLOCK_END}): "
      f"{N_RECID_CHANNELS} z-score normalized stats")
print(f"  RECID normalizer loaded: {len(RECID_NORMALIZER)}/{len(RECID_FEATURES)} features")
print(f"MaxChi windows (bp): {MAXCHI_WINDOWS}")
print(f"pos_weight (legacy, per-position only): {POS_WEIGHT}")
'''
set_source('cell-3', cell3)

# ============================================================
# cell-6: add _recid_channels helper, extend encode_triplet
# ============================================================
cell6 = '''def one_hot_encode(sequence, max_length=MAX_SEQ_LEN):
    nuc_idx = {'A': 0, 'T': 1, 'G': 2, 'C': 3, '-': 4}
    encoded = np.zeros((max_length, N_CHANNELS), dtype=np.float32)
    for i, nuc in enumerate(sequence[:max_length].upper()):
        idx = nuc_idx.get(nuc, 4)
        encoded[i, idx] = 1.0
    return encoded


def _seq_to_index(sequence, max_length=MAX_SEQ_LEN):
    nuc_idx = {'A': 0, 'T': 1, 'G': 2, 'C': 3, '-': 4}
    idx = np.full(max_length, -1, dtype=np.int16)
    for i, nuc in enumerate(sequence[:max_length].upper()):
        idx[i] = nuc_idx.get(nuc, 4)
    return idx


def _maxchi_features(parental_signal, windows=MAXCHI_WINDOWS, length=MAX_SEQ_LEN):
    out = np.zeros((length, len(windows)), dtype=np.float32)
    for k, w in enumerate(windows):
        padded = np.pad(parental_signal, (w, w), mode='constant')
        csum = np.cumsum(padded, dtype=np.float64)
        prefix = np.concatenate(([0.0], csum))
        left_mean = (prefix[w:w+length] - prefix[:length]) / w
        right_mean = (prefix[2*w:2*w+length] - prefix[w:w+length]) / w
        out[:, k] = (right_mean - left_mean).astype(np.float32)
    return out


def _gaussian_peak(center, sigma, length=MAX_SEQ_LEN):
    if center is None:
        return np.zeros(length, dtype=np.float32)
    try:
        c = int(center)
    except (TypeError, ValueError):
        return np.zeros(length, dtype=np.float32)
    if c < 0 or c >= length:
        return np.zeros(length, dtype=np.float32)
    pos = np.arange(length, dtype=np.float32)
    return np.exp(-0.5 * ((pos - c) / float(sigma)) ** 2).astype(np.float32)


def _rdp_pred_channels(pred_bp_start, pred_bp_end, sigma=None, length=MAX_SEQ_LEN):
    if sigma is None:
        sigma = RDP_PRED_SIGMA
    g1 = _gaussian_peak(pred_bp_start, sigma, length)
    g2 = _gaussian_peak(pred_bp_end, sigma, length)
    return np.stack([g1, g2], axis=1)


def _method_confidence_channels(p_values, length=MAX_SEQ_LEN):
    n_methods = len(RDP_METHOD_NAMES)
    out = np.zeros((length, n_methods), dtype=np.float32)
    if p_values is None:
        return out
    for m, p in enumerate(p_values[:n_methods]):
        if p is None:
            continue
        try:
            pf = float(p)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(pf) or pf <= 0:
            continue
        logp = -np.log10(pf)
        w = max(0.0, min(1.0, logp / RDP_METHOD_LOGP_CLIP))
        if w == 0.0:
            continue
        out[:, m] = w
    return out


def _recid_channels(recid_features, length=MAX_SEQ_LEN):
    """Run #40: 10 z-score-normalized scalar broadcast channels from
    RecombIdentifyStats (A)-row features. recid_features is a dict mapping
    RECID_FEATURES name -> raw value, or None if no winning hypothesis row.
    """
    n = len(RECID_FEATURES)
    out = np.zeros((length, n), dtype=np.float32)
    if not recid_features:
        return out
    for i, fname in enumerate(RECID_FEATURES):
        v = recid_features.get(fname)
        if v is None:
            continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(x):
            continue
        nm = RECID_NORMALIZER.get(fname)
        if nm is None:
            continue
        z = (x - nm['mean']) / nm['std']
        # Clip extremes to keep one outlier from blowing the channel.
        z = max(-5.0, min(5.0, z))
        out[:, i] = z
    return out


def encode_triplet(seq_recomb, seq_parent1, seq_parent2,
                   pred_bp_start=None, pred_bp_end=None,
                   method_pvalues=None, recid_features=None):
    """43-channel layout:
        [0..14]   one-hot triplet (recomb, p1, p2)
        [15..17]  match_p1, match_p2, informative
        [18..21]  MaxChi 4 windows
        [22..23]  RDP consensus PredBPStart/End Gaussians
        [24..32]  9 per-method confidence scalars
        [33..42]  10 RecombIdentifyStats (A)-row z-score scalars
    """
    enc_r = one_hot_encode(seq_recomb)
    enc_1 = one_hot_encode(seq_parent1)
    enc_2 = one_hot_encode(seq_parent2)
    r = _seq_to_index(seq_recomb); p1 = _seq_to_index(seq_parent1); p2 = _seq_to_index(seq_parent2)
    valid = (r >= 0) & (p1 >= 0) & (p2 >= 0)
    match_p1   = ((r  == p1) & valid).astype(np.float32)
    match_p2   = ((r  == p2) & valid).astype(np.float32)
    informative = ((p1 != p2) & valid).astype(np.float32)
    parental_signal = (match_p1 - match_p2).astype(np.float32)
    maxchi = _maxchi_features(parental_signal)
    rdp = _rdp_pred_channels(pred_bp_start, pred_bp_end)
    method = _method_confidence_channels(method_pvalues)
    recid = _recid_channels(recid_features)
    return np.concatenate(
        [enc_r, enc_1, enc_2,
         match_p1[:, None], match_p2[:, None], informative[:, None],
         maxchi, rdp, method, recid],
        axis=1,
    )


# Sanity
test_enc = one_hot_encode("ATGC-N", max_length=6)
print(f"One-hot shape: {test_enc.shape}")
trip = encode_triplet("ATGCAT", "ATGGGT", "ATCCAT",
                      pred_bp_start=2, pred_bp_end=4,
                      method_pvalues=[1e-50, 1e-10, None, 1e-30, 'NS', 1e-200, None, None, 1e-100],
                      recid_features={'dMax(A)': 0.8, 'BadDists(A)': 1.0, 'SubPhPrScore(A)': 0.9,
                                      'SimScore(A)': 1.0, 'PhPrScore(A)': 0.5, 'PhPrScore2(A)': 0.6,
                                      'RCompatS(A)': 2.0, 'ListCorr3(A)': 0.3, 'PhPrScore3(A)': 0.5,
                                      'OUList(A)': 1.0})
print(f"Triplet shape: {trip.shape}  (expected ({MAX_SEQ_LEN}, {N_INPUT_CHANNELS}))")
print(f"Method scalars (ch 24..33) at pos 0: {[round(x, 3) for x in trip[0, 24:33].tolist()]}")
print(f"RECID scalars  (ch 33..43) at pos 0: {[round(x, 3) for x in trip[0, 33:43].tolist()]}")
print(f"RECID scalars  (ch 33..43) at pos 5000 (broadcast - identical): "
      f"{[round(x, 3) for x in trip[5000, 33:43].tolist()]}")
'''
set_source('cell-6', cell6)

# ============================================================
# cell-10: thread recid_features into encode_triplet
# ============================================================
cell10 = '''def _parse_method_pvalues(fa_csv_path, n_methods=9):
    """Parse 9 per-method p-values from .fa.csv (run #39)."""
    fa_csv_path = Path(fa_csv_path)
    if not fa_csv_path.exists():
        return {}
    try:
        df = pd.read_csv(fa_csv_path, skiprows=15, header=None, on_bad_lines='skip')
    except Exception:
        return {}
    if df.shape[1] < 20:
        return {}
    df.columns = list(range(df.shape[1]))
    ev_num = pd.to_numeric(df[0], errors='coerce')
    begin = pd.to_numeric(df[2], errors='coerce')
    keep = ev_num.notna() & begin.notna()
    df = df[keep].copy()
    out = {}
    for _, row in df.iterrows():
        try:
            ev = int(row[0])
        except (TypeError, ValueError):
            continue
        pvals = []
        for ci in range(11, 11 + n_methods):
            v = row[ci]
            if isinstance(v, str):
                vs = v.strip()
                if vs.upper() == 'NS' or vs == '':
                    pvals.append(None); continue
                try:
                    f = float(vs)
                except ValueError:
                    pvals.append(None); continue
            else:
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    pvals.append(None); continue
            pvals.append(f if (np.isfinite(f) and f > 0) else None)
        out[ev] = pvals
    return out


def _extract_recid_winning(stats_df, event, recomb_id):
    """Find the (A)-hypothesis row whose ISeqs(A) contains the actual recomb_id.
    Return a dict mapping RECID_FEATURES -> raw value, or None if no match.
    """
    ev_rows = stats_df[stats_df['Event'] == event]
    if len(ev_rows) != 3:
        return None
    for _, sr in ev_rows.iterrows():
        ids = [int(s.strip()) for s in str(sr['ISeqs(A)']).split('$') if s.strip().isdigit()]
        if recomb_id in ids:
            out = {}
            for f in RECID_FEATURES:
                try:
                    out[f] = sr[f]
                except KeyError:
                    out[f] = None
            return out
    return None


def parse_simulation(fasta_path):
    fasta_path = Path(fasta_path)
    sim_csv = fasta_path.parent / f"{fasta_path.name}SimVSRealCompare.csv"
    stats_csv = fasta_path.parent / f"{fasta_path.name}RecombIdentifyStats.csv"
    fa_csv = fasta_path.parent / f"{fasta_path.name}.csv"

    if not sim_csv.exists() or not stats_csv.exists():
        return []

    try:
        seqs = {int(r.id): str(r.seq) for r in SeqIO.parse(fasta_path, 'fasta')}
    except Exception:
        return []

    try:
        sim = pd.read_csv(sim_csv, skipinitialspace=True, index_col=False)
        stats = pd.read_csv(stats_csv, skipinitialspace=True, index_col=False)
    except Exception:
        return []

    method_pvalues_by_ev = _parse_method_pvalues(fa_csv)

    results = []
    for _, row in sim.iterrows():
        event = row['RDPEvent']
        recomb_id = int(row['ActualRecomb'])
        bp_start = int(row['SimBPStart'])
        bp_end = int(row['SimBPEnd'])
        try:
            pred_bp_start = int(row['PredBPStart']) if pd.notna(row['PredBPStart']) else None
        except (KeyError, TypeError, ValueError):
            pred_bp_start = None
        try:
            pred_bp_end = int(row['PredBPEnd']) if pd.notna(row['PredBPEnd']) else None
        except (KeyError, TypeError, ValueError):
            pred_bp_end = None

        ev_rows = stats[stats['Event'] == event]
        if len(ev_rows) != 3:
            continue
        parent_ids = []
        for _, sr in ev_rows.iterrows():
            ids = [int(s.strip()) for s in str(sr['ISeqs(A)']).split('$') if s.strip().isdigit()]
            if recomb_id in ids:
                continue
            if ids:
                parent_ids.append(ids[0])
        if len(parent_ids) < 2:
            continue
        if not all(sid in seqs for sid in [recomb_id, parent_ids[0], parent_ids[1]]):
            continue

        try:
            ev_int = int(event)
        except (TypeError, ValueError):
            ev_int = None
        method_pvalues = method_pvalues_by_ev.get(ev_int) if ev_int is not None else None
        recid_features = _extract_recid_winning(stats, event, recomb_id)

        triplet = encode_triplet(
            seqs[recomb_id], seqs[parent_ids[0]], seqs[parent_ids[1]],
            pred_bp_start=pred_bp_start, pred_bp_end=pred_bp_end,
            method_pvalues=method_pvalues, recid_features=recid_features,
        )

        results.append({
            'input': triplet,
            'labels_gaussian': generate_labels(bp_start, bp_end, mode='gaussian'),
            'labels_bp': generate_labels(bp_start, bp_end, mode='breakpoint'),
            'labels_region': generate_labels(bp_start, bp_end, mode='region'),
            'meta': {
                'file': fasta_path.name,
                'event': event,
                'recomb_id': recomb_id,
                'parent1_id': parent_ids[0],
                'parent2_id': parent_ids[1],
                'bp_start': bp_start,
                'bp_end': bp_end,
                'actual_len': len(seqs[recomb_id].rstrip('-')),
                'has_method_pvalues': method_pvalues is not None,
                'has_recid': recid_features is not None,
            },
        })
    return results


sample_fa = sorted((DATA_ROOT / "XML-1").glob("*.fa"))[0]
sample_triplets = parse_simulation(sample_fa)
print(f"Parsed {len(sample_triplets)} triplets from {sample_fa.name}")
if sample_triplets:
    s0 = sample_triplets[0]
    print(f"Input shape:  {s0['input'].shape}")
    print(f"  has_method_pvalues={s0['meta']['has_method_pvalues']}, has_recid={s0['meta']['has_recid']}")
'''
set_source('cell-10', cell10)

# ============================================================
# cell-11: bump CACHE_VERSION v4 -> v5
# ============================================================
cell11 = get_cell('cell-11')
src11 = ''.join(cell11['source'])
src11_new = src11.replace(
    "CACHE_VERSION = 'v4'  # run #39 v2: +9 per-method scalar channels (24 -> 33 channels)",
    "CACHE_VERSION = 'v5'  # run #40: +10 RecombIdentifyStats top-importance scalars (33 -> 43 channels)",
)
assert src11_new != src11, "CACHE_VERSION v4->v5 replacement failed"
set_source('cell-11', src11_new)

# ============================================================
# cell-22: bump model save name run39 -> run40, extend dropout block
# ============================================================
cell22 = get_cell('cell-22')
src22 = ''.join(cell22['source'])

# Extend the dropout zeroing to cover RECID block too. Replace the body of _make_gen.
# Find the line  "x_i[:, RDP_BLOCK_START:RDP_BLOCK_END] = 0"
# Replace with two zero-lines covering RDP + RECID blocks (same dropout coin per sample
# means the model is forced to use sequence features alone with prob RDP_DROPOUT_P).
old_drop = "                x_i[:, RDP_BLOCK_START:RDP_BLOCK_END] = 0"
new_drop = ("                x_i[:, RDP_BLOCK_START:RDP_BLOCK_END] = 0\n"
            "                x_i[:, RECID_BLOCK_START:RECID_BLOCK_END] = 0")
assert old_drop in src22, "RDP-dropout line not found in cell-22"
src22 = src22.replace(old_drop, new_drop)

old_save = "_versioned = 'models_test/cnn_breakpoint_run39_final.keras'"
new_save = "_versioned = 'models_test/cnn_breakpoint_run40_final.keras'"
assert old_save in src22, "old save name not found"
src22 = src22.replace(old_save, new_save)
set_source('cell-22', src22)

NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n')
print("Run #40 cell edits applied.")
print("N_INPUT_CHANNELS: 33 -> 43")
print("CACHE_VERSION: v4 -> v5")
print("Model save name: cnn_breakpoint_run40_final.keras")
print("Dropout block extended to cover RECID channels.")
