# Physics-agnostic Pretraining for Neural Operators on Irregular Mesh

Th manuscript is published at
[[FROM CHEAP GEOMETRY TO EXPENSIVE PHYSICS: A
PHYSICS-AGNOSTIC PRETRAINING FRAMEWORK FOR
NEURAL OPERATORS]](https://openreview.net/forum?id=iCprPzyrRp)

### Environment requirements
```
pytorch, numpy, pyyml, einops, scipy
```

### Datasets:
We provide four datasets: 2D Stress, 2D AirfRans, 3D Inductor, 2D Electrostatic. Each dataset contains the geometry point cloud, preprocessed occupancy field, and the PDE field from two different query methods. Please find more details in the manuscript.

**Note:** Datasets will be released soon.

### Train autoencoder:
DDP is used for parallel training, so you can use 1 or more GPUs.

```
torchrun --nproc_per_node=2 main_ae.py --config Configs/VAE/Stress.yaml
```

Please update `--nproc_per_node`, `batch_size`, and `accum_iter` (gradient accumulation steps) in the config file according to your hardware setup.

### Train neural operators:
```
torchrun --nproc_per_node=1 main_no.py --config Configs/NO/Stress_GNOT.yaml
```
or
```
torchrun --nproc_per_node=1 main_no.py --config Configs/NO/Stress_VAE_GNOT.yaml
```

Please update `--nproc_per_node`, `batch_size`, and `accum_iter` (gradient accumulation steps) in the config file according to your hardware setup.

To use encoded latent embeddings as inputs for operator training, set `use_VAE: true` in the config file and provide the path to the pretrained autoencoder checkpoint via `vae_pth`.

Edit the `use_mesh` argument to change the query method.

The repo integrates pretrained encoders with the following transformer based neural operators:
- GNOT: from code implemented in `models/cgpt.py` — [GNOT paper](https://proceedings.mlr.press/v202/hao23c)
- Transolver: from code implemented in `models/Transolver.py` — [Transolver paper](https://arxiv.org/abs/2402.02366)
- LNO: from code implemented in `models/LNO.py` — [LNO paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/39f6d5c2e310a5a629dcfc4d517aa0d1-Abstract-Conference.html)