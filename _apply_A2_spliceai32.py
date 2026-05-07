"""Apply A2: replace cell-18 build_cnn with SpliceAI-32 backbone.

Spec:
  - Initial 1x1 Conv1D projection from N_INPUT_CHANNELS -> W=32 filters.
  - 32 ResidualUnits in 4 dilation groups (8 RUs each at d ∈ {1, 4, 10, 25}).
  - Each RU: BN -> ReLU -> Conv(W, kernel=11) -> BN -> ReLU -> Conv(W, kernel=11) + skip.
  - Skip-out connection: 1x1 conv after each 8-RU group, summed into accumulator.
  - Output head: 1x1 Conv1D to logits with prior-aware bias, then sigmoid.

Receptive field: 1 + 20 * (8*1 + 8*4 + 8*10 + 8*25) = 6,401 positions ~= 6.4 kb.
At kernel=11, this exceeds the dilated-stack (RF ~ 1+2*32*6 = 385). Should
break the capacity ceiling of the run #38 backbone.

Param count at W=32: ~735k. Smaller than the 128-filter dilated stack but
deeper effective receptive field. Run #41 can bump W=64 if W=32 underfits.

Apply ONLY after evaluating run #39 (and possibly #40) and deciding to
pivot architectures.
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

cell18 = '''# A2 SpliceAI-32 backbone (replaces dilated-stack from runs #16/#22/#32).
#
# 32 ResidualUnits in 4 dilation groups (d=1,4,10,25; 8 RUs per group).
# Each RU: BN-ReLU-Conv(W=32, k=11)-BN-ReLU-Conv(W=32, k=11) + skip.
# Skip-out 1x1 after each group, summed into an accumulator that feeds the head.
# Receptive field: 1 + 20 * (8 + 32 + 80 + 200) = 6401 positions (6.4 kb),
# vs ~385 for the prior dilated stack. Better suited for 32k inputs where
# breakpoint signal can span kb-scale parental-disparity windows.
import math


def _residual_unit(x, W, kernel, dilation, name_prefix):
    """SpliceAI-style residual unit: BN-ReLU-Conv-BN-ReLU-Conv + skip."""
    skip = x
    y = BatchNormalization(name=f'{name_prefix}_bn1')(x)
    y = Activation('relu', name=f'{name_prefix}_relu1')(y)
    y = Conv1D(W, kernel, dilation_rate=dilation, padding='same',
               name=f'{name_prefix}_conv1')(y)
    y = BatchNormalization(name=f'{name_prefix}_bn2')(y)
    y = Activation('relu', name=f'{name_prefix}_relu2')(y)
    y = Conv1D(W, kernel, dilation_rate=dilation, padding='same',
               name=f'{name_prefix}_conv2')(y)
    return Add(name=f'{name_prefix}_add')([skip, y])


def build_cnn(input_shape=(MAX_SEQ_LEN, N_INPUT_CHANNELS),
              W=32, kernel=11,
              dilation_groups=(1, 4, 10, 25),
              units_per_group=8,
              prior_positive=0.01):
    """SpliceAI-32 backbone for per-position breakpoint detection.

    Returns a Keras Model with output shape (batch, MAX_SEQ_LEN) — sigmoid.
    """
    inputs = Input(shape=input_shape, name='triplet_input')

    # 1x1 projection to W filters.
    x = Conv1D(W, 1, padding='same', name='conv_proj')(inputs)

    # Skip accumulator (also 1x1-projected for shape match).
    skip_acc = Conv1D(W, 1, padding='same', name='skip_acc_init')(x)

    # 4 dilation groups, 8 RUs each = 32 total.
    for g, dilation in enumerate(dilation_groups):
        for u in range(units_per_group):
            x = _residual_unit(x, W, kernel, dilation,
                               name_prefix=f'ru_g{g}_u{u}')
        # Skip-out: 1x1 -> add into accumulator.
        skip_out = Conv1D(W, 1, padding='same', name=f'skip_out_g{g}')(x)
        skip_acc = Add(name=f'skip_add_g{g}')([skip_acc, skip_out])

    # Final BN-ReLU before head.
    x = BatchNormalization(name='head_bn')(skip_acc)
    x = Activation('relu', name='head_relu')(x)

    # Per-position scalar logit with prior-aware bias init.
    bias_init = -math.log((1 - prior_positive) / prior_positive)
    logits = Conv1D(1, 1, padding='same',
                    bias_initializer=tf.keras.initializers.Constant(bias_init),
                    name='conv_out')(x)
    outputs = Reshape((MAX_SEQ_LEN,), name='output')(
        Activation('sigmoid', name='sigmoid_out')(logits))
    return Model(inputs, outputs, name='SpliceAI32_PerPosition')


# Build the active per-position model.
cnn = build_cnn()
cnn.compile(
    optimizer=AdamW(learning_rate=LR, weight_decay=1e-5),
    loss=weighted_bce(POS_WEIGHT),
    metrics=[tf.keras.metrics.AUC(curve='PR', name='aupr')],
)
cnn.summary()
'''
set_source('cell-18', cell18)

# Bump model save name to runNN where caller chooses (default 41 — A2 first try).
cell22 = get_cell('cell-22')
src22 = ''.join(cell22['source'])
import re
m = re.search(r"_versioned = 'models_test/cnn_breakpoint_run(\d+)_final\.keras'", src22)
if m:
    old_run = m.group(1)
    new_run = '41'  # caller can re-run with custom number if doing >1 A2 variant
    src22 = src22.replace(
        f"_versioned = 'models_test/cnn_breakpoint_run{old_run}_final.keras'",
        f"_versioned = 'models_test/cnn_breakpoint_run{new_run}_final.keras'",
    )
    set_source('cell-22', src22)

NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n')
print("A2 SpliceAI-32 cell-18 swap applied.")
print("New build_cnn = 32 RUs in 4 dilation groups @ W=32, kernel=11.")
print(f"Model save name: cnn_breakpoint_run41_final.keras")
