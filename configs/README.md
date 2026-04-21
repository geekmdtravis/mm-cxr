# configs/

Per-backbone model configuration YAMLs, one file per (stage, backbone).

## Layout

- `stage1/<backbone>.yaml` — abnormality gate, `num_classes: 1`
- `stage2/<backbone>.yaml` — pathology classifier, `num_classes: 14`

## Schema

Each YAML is loaded by `mm_cxr_diag.models.CXRModelConfig.from_yaml()`.
Keys map 1:1 to `CXRModelConfig` fields:

| Key                | Type                | Notes                                                   |
| ------------------ | ------------------- | ------------------------------------------------------- |
| `model`            | str                 | Backbone name — one of the registered backbones.        |
| `hidden_dims`      | list[int] \| null   | Classifier MLP widths. `null` → linear head.            |
| `dropout`          | float               | Dropout between classifier layers.                      |
| `num_classes`      | int                 | 1 for Stage 1, 14 for Stage 2.                          |
| `tabular_features` | int                 | Number of clinical features fed into the head (usually 4). |
| `freeze_backbone`  | bool                | Freeze conv/transformer weights; head still trains.     |

## Training hyperparameters

These YAMLs intentionally do not include training hyperparameters
(learning rate, epochs, batch size, loss, scheduler). Those will join a
richer `TrainConfig` schema in M4 (CLI milestone).

## Recommended defaults

- **Stage 1**: `densenet121` — binary task, smaller backbone is enough
  and calibrates better.
- **Stage 2**: `densenet201` — harder 14-way task benefits from extra
  capacity. `vit_l_16` is the stretch goal if compute allows.
