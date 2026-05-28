# SpikeNet Tuning Guide

## How the LIF neuron works (the short version)

At each timestep `t`, a neuron does three things:

```
1. Reset:    rst  = spk_prev × b × w_norm
2. Integrate: mem  = (mem - rst) × β + input_current × (1 − β)
3. Spike:    spk  = 1  if  mem / w_norm > b  else  0
```

Where:
- `mem` — membrane potential, clamped to `[0, 1]`
- `β` (beta) — membrane decay; how much the previous membrane carries forward
- `b` — learnable threshold bias
- `w_norm = Σ w²` — L2 norm of the weights for that neuron (per output neuron)
- `input_current = x @ w` — weighted sum of incoming spikes

The **effective firing threshold in membrane space** is `b × w_norm`. A neuron fires when `mem > b × w_norm`.

---

## Parameters reference

### Layer parameters (`SpikingDenseLayer`, `SpikingConv2D`)

| Parameter | Default | What it does |
|---|---|---|
| `w_init_mean` | `0.01` | Mean of the weight initialization. A nonzero value creates a **positive bias** in the input current that scales with `in_features`. Use `0.0` for layers receiving high-activity input. |
| `w_init_std` | `0.15` | Spread of weights. Controls `w_norm ≈ w_init_std²` at initialization, which determines the initial threshold scale. |
| `beta_init_mean` | `0.7` | Initial membrane decay. Higher = more temporal integration (longer memory). Lower = neuron resets faster. Range `(0, 1)`. |
| `b_init_mean` | `1.0` | Starting value of the learnable threshold. The critical parameter for controlling firing rate — see calibration below. |
| `b_init_std` | `0.01` | Spread in initial threshold values across neurons. Keep small; threshold diversity mostly emerges from training. |
| `input_noise_std` | `0.02` | Gaussian noise added to the input current during training only. Acts as regularization. Set to `0.0` to disable. |
| `clamp_membrane` | `True` | Clamps `mem` to `[0, 1]` after each step. Keeps dynamics bounded. Disable only if you know what you're doing. |
| `time_reduction` | `no_time_reduction` | How to collapse the time dimension into a single output vector. See below. |

### Time reduction options

| Option | Output | When to use |
|---|---|---|
| `no_time_reduction` | `(batch, time, features)` — full spike train | For hidden layers; next layer receives spikes over time |
| `max_membrane_potential` | `(batch, features)` — max membrane over time | For output layer; good for classification |
| `spike_rate` | `(batch, features)` — mean spike count | Alternative output encoding |

Use `no_time_reduction` for every hidden layer. Use `max_membrane_potential` or `spike_rate` for the **output layer only**.

### Network / training parameters

| Parameter | Default | What it does |
|---|---|---|
| `learning_rate` | `1e-4` | Step size for SGD. SNNs are sensitive — start low, increase if convergence is too slow. |
| `momentum` | `0.9` | SGD momentum. |
| `weight_decay` | `1e-5` | L2 regularization on weights. Increase to slow weight growth. |
| `max_grad_norm` | `1.0` | Gradient clipping. **Do not disable.** Without this, weights in deeper layers can explode in a single epoch when upstream layers over-fire. |
| `epochs` | `10` | Number of training passes. |

---

## Calibration: setting `b_init_mean`

The equilibrium firing rate of a layer can be estimated analytically:

```
r_spike ≈ E[input_current] × (1 − β) / (b × w_norm × β)
```

At initialization, `w_norm ≈ w_init_std²` (invariant to layer width due to the `sqrt(1/in_features)` scaling). So to target a firing rate `r*`:

```
b_init_mean = E[input_current] × (1 − β) / (r* × w_init_std² × β)
```

And the expected input current depends on the upstream layer:

```
E[input_current] = in_features × upstream_firing_rate × w_init_mean
```

**Quick reference** (β=0.7, w_init_std=0.15, w_init_mean=0.01):

| in_features | upstream rate | target rate | b_init_mean |
|---|---|---|---|
| 196 (14×14 MNIST) | 13% (raw image) | 20% | ~8 |
| 196 | 13% | 10% | ~16 |
| 512 | 20% | 20% | ~25 |
| 512 | 50% | 20% | ~62 |

**Shortcut**: set `w_init_mean=0.0` for layers where the upstream firing rate is high or uncertain. This zeros out the systematic bias, making the equilibrium rate depend only on weight variance (which is predictable) instead of mean × firing_rate (which is not).

---

## Diagnostic guide

### Firing rate interpretation

| What you observe | Meaning |
|---|---|
| Layer fires > 80% | Threshold too low — neurons fire on almost anything |
| Layer fires < 5% | Threshold too high — or weights have grown too large |
| Layer fires 0% while upstream fires > 80% | Weight explosion (see below) |
| Loss stuck at `log(num_classes)` (~2.3 for 10 classes) | Output layer is uniform — likely saturated or silent |

---

### "A layer fires too much (> 70–80%)"

The effective threshold `b × w_norm` is too low relative to the input current.

**Fix options (in order of preference):**

1. Increase `b_init_mean` for that layer. Use the calibration formula above, or just double it and re-run.
2. Lower `w_init_mean` toward `0.0`. Removes the systematic positive bias that scales with layer size.
3. Reduce `input_noise_std`. At high firing rates, noise amplifies the problem.

---

### "A later layer fires 0% while an earlier one fires > 80%"

This is the weight explosion pattern:

- Upstream fires at high rate → strong gradient signal to downstream weights
- Weights grow 10–100× in one epoch
- `w_norm = Σ w²` grows proportionally → `1/w_norm → 0`
- Effective threshold becomes `mem / w_norm < b` for all `mem ≤ 1.0`
- Layer goes completely silent

**Fix:**

1. **First, fix the upstream layer.** A downstream layer cannot stabilize if it's receiving 90%+ activity. Use the calibration formula to bring upstream firing below 30%.
2. **Enable gradient clipping** (`max_grad_norm=1.0` in `net.train()`). This is the safety net.
3. If the layer is the output layer: set `w_init_mean=0.0` on it to decouple it from the upstream activity level.

---

### "Loss decreases but accuracy doesn't improve"

The network is learning a representation but the output layer isn't discriminating.

- Check that your output layer uses `time_reduction=max_membrane_potential` or `spike_rate`
- Check that the output layer has a meaningful firing rate (5–50%)
- If the output layer fires uniformly, it has no class-selective neurons yet — try more epochs or a higher learning rate for the first few epochs

---

### "Loss oscillates or explodes"

- Learning rate is too high: try `1e-4` or lower
- `max_grad_norm` too loose: try `0.5` or `0.3`
- `beta_init_mean` too close to 1.0 (near 0.99): the membrane has very long memory and small inputs accumulate without bound before a spike resets them

---

### "Accuracy plateaus around 1/N (random chance)"

The network never learned class-discriminative features. Usually one of:

- All output neurons always fire (uniform output → uniform `LogSoftmax`)
- All output neurons never fire (zero output)
- `time_reduction` on a hidden layer accidentally collapses the spike train too early

---

## Using `dry_run` to validate initial weights

`net.dry_run(data)` runs one forward pass (eval mode, no noise) right after you compile and initialise the network — before any training. It prints a colour-coded table of the firing rate for every spiking layer:

```
     Dry run — initial weight check
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Layer              ┃ Shape     ┃ Firing rate ┃ Status                                   ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ SpikingDenseLayer  │ 196 → 128 │       21.4% │ ✓                                        │
│ SpikingDenseLayer  │ 128 → 10  │       38.2% │ ✓                                        │
└────────────────────┴───────────┴─────────────┴──────────────────────────────────────────┘
All layers within healthy range (5–80 %).
```

**Colour coding:**

| Rate | Colour | Meaning |
|---|---|---|
| 5–50 % | green ✓ | healthy |
| 50–80 % | yellow △ | slightly high; consider raising `b_init_mean` |
| > 80 % | red ✗ | too high — raise `b_init_mean` or lower `w_init_mean` |
| < 5 % | red ✗ | too low — lower `b_init_mean` |

**Recommended workflow before every training run:**

```python
net.compiled(input_features=1, output_features=10, data_shape=sample_x.shape)
net.initialize_parameters()
net.summarise()
net.dry_run(data)          # check rates — fix any red before proceeding
m = net.train(data, ...)
```

If any layer shows red, adjust the flagged layer's `b_init_mean` (use the calibration formula above), call `net.initialize_parameters()` again to re-randomize with the new value, and re-run `dry_run` until all layers are green or yellow.

**Why eval mode?** `dry_run` disables input noise (`self.training = False`) so you see the pure weight-driven baseline. If the baseline is already over-firing before any noise is added, training will be unstable regardless of noise settings.

---

## Tracking training with `Metrics`

`net.train()` returns a `Metrics` object instead of a plain list. It holds both epoch-level and per-batch data, and supports `+=` to concatenate multiple training phases.

```python
m = net.train(data, learning_rate=1e-3, epochs=3)
m += net.train(data, learning_rate=1e-4, epochs=7)  # fine-tune at lower LR

m.print()   # rich table with per-epoch loss and accuracy
m.plot()    # loss + accuracy plots with intra-epoch oscillation and phase separator
```

`m.plot()` shows:
- **thin transparent line** — per-batch values (reveals within-epoch oscillation)
- **thick line** — epoch averages
- **dashed vertical line** — marks where the `+=` phase boundary falls
- **dashed horizontal line** — test set reference (auto-run at end of each `train()` call)

Useful patterns:

```python
# Gate training on dry run result
if net.dry_run(data):
    m = net.train(data, epochs=10)
    m.plot()

# Access final metrics programmatically
print(m.loss, m.accuracy)
print(m.test)     # {"loss": ..., "accuracy": ...}
```

---

## Typical configuration for a dense SNN on MNIST

```python
net += Flatten()
net += SpikingDenseLayer(512, b_init_mean=8.0)          # ~20% firing
net += SpikingDenseLayer(10,
    time_reduction=max_membrane_potential,
    w_init_mean=0.0,                                    # decouple from upstream rate
)

net.compiled(input_features=1, output_features=10, data_shape=sample_x.shape)
net.initialize_parameters()
net.dry_run(data)                                       # validate before training

m = net.train(data, learning_rate=1e-3, epochs=5, max_grad_norm=1.0)
m.plot()
```
