# PhysHazeDiffusion

This is a physically constrained branch of the original HazeGen residual training code.

## Core Idea

The diffusion model no longer generates a free RGB haze residual. It generates a
single haze density/transmission field, represented as a 3-channel VAE carrier:

![PhysHazeDiffusion architecture](figures/phys_hazegen_architecture_cn_4k.png)

```text
diffusion(clean, depth, prompt) -> density d(x)
airlight_head(clean) or real haze bank -> atmospheric light A
I(x) = J(x) * (1 - d(x)) + A * d(x)
```

This keeps depth as a geometry cue and prevents it from directly changing RGB
colors.

## Stage 1

Stage 1 uses paired synthetic clean/hazy data:

```text
paired clean/hazy -> estimate A and density target
VAE(density carrier) -> diffusion target z0
airlight_head(clean) -> A
```

Run:

```bash
bash run_train_phys_hazegen.sh
```

Set `STAGE=stage1` in the script if you only want Stage 1.

## Stage 2

Stage 2 does real-domain adaptation without learning a free RGB pseudo target:

```text
teacher -> pseudo density
real hazy images -> airlight bank
CLIP prompt branch -> directional clean-to-haze constraint
student learns density distribution + real haze parameter distribution
```

The CLIP branch is directional:

```text
CLIP(pred_hazy) - CLIP(clean)  aligned with  CLIP(haze_prompt) - CLIP(clear_prompt)
```

This asks whether the image change moves toward the haze domain, instead of
forcing the generated image itself toward a hazy-image prototype.

## Inference

Run:

```bash
bash run_inference_phys_hazegen.sh
```

Atmospheric light can come from:

```text
AIRLIGHT_SOURCE=head   # predict A from clean image
AIRLIGHT_SOURCE=bank   # sample A from real haze bank
AIRLIGHT_SOURCE=fixed  # use FIXED_AIRLIGHT
```

## Important Files

- `train_phys_hazegen.py`: two-stage physical training entry.
- `inference_phys_hazegen.py`: physical composition inference.
- `phys_haze_utils.py`: density/A estimation and composition utilities.
- `configs/train/phys_stage.yaml`: 5-channel ControlNet config.
- `run_train_phys_hazegen.sh`: default training command.
- `run_inference_phys_hazegen.sh`: default inference command.

## Notes

The `weights` directory is a symlink to the original project weights to avoid
duplicating large checkpoint files.
