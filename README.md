<div align="center">

# Not Just a Subject: Capturing Adaptive Scene-Specific Variance for Visual Storytelling

### ACCV 2026

[*SeungJu Cha*](https://openreview.net/profile?id=~SeungJu_Cha1)&nbsp;·&nbsp;[*Ye-Chan Kim*](https://openreview.net/profile?id=~Ye-Chan_Kim1)&nbsp;·&nbsp;[*Kwanyoung Lee*](https://github.com/mobled37)&nbsp;·&nbsp;[*Dong-Jin Kim*](https://openreview.net/profile?id=~Dong-Jin_Kim1)

**📄 Paper: coming soon · arXiv: coming soon**

<!-- Replace the line above with: [Paper](PAPER_URL) · [arXiv](ARXIV_URL) -->

<p align="center">
  <img src="figs/pipe.png" width="100%" alt="EOS-Story pipeline">
</p>

<p><b>EOS-Story pipeline.</b> Cross-Contextual Bridging captures scene-specific variance, while Identity Anchoring and Identity-Aware Self-Attention preserve subject identity throughout generation.</p>

</div>

---

## Qualitative Results

EOS-Story maintains subject identity across diverse scenes while responding to additional camera-angle and composition controls.

<p align="center">
  <img src="figs/add_qual_view.png" width="100%" alt="Additional EOS-Story qualitative results with viewpoint and composition controls">
</p>

## Official Implementation

This repository contains the official SDXL implementation of **EOS-Story**. It supports single-subject stories, multi-subject stories, and optional pose control with precomputed OpenPose maps.

EOS-Story combines three components:

- **Cross-Contextual Bridging (CCB)** transfers scene-specific appearance through EOS embeddings.
- **Identity Anchoring (IA)** anchors subject features during early denoising steps.
- **Identity-Aware Self-Attention (IASA)** preserves identity while retaining scene variation.

## Installation

Python 3.10 and a CUDA-capable NVIDIA GPU are recommended.

```bash
conda create -n eos-story python=3.10
conda activate eos-story
pip install -r requirements.txt
```

The SDXL checkpoint is downloaded from `stabilityai/stable-diffusion-xl-base-1.0` on first use. If authentication is required, log in with the Hugging Face CLI or set `HF_TOKEN`.

## Quick Start

Run commands from the repository root:

```bash
bash test.sh single
bash test.sh multi
bash test.sh pose
```

Run all three examples with:

```bash
bash test.sh all
```

| Example | Configuration |
| --- | --- |
| Single subject | Phoenix, 2 scenes, seed 12, λ = 0.55 |
| Multiple subjects | Man and woman, 5 scenes, seed 44, λ = 0.6 |
| OpenPose ControlNet | Man, 2 scenes, seed 12, λ = 0.55 |

Use `--dry_run` to validate inputs and output paths without loading SDXL:

```bash
bash test.sh all --dry_run
```

Set `PYTHON` when a specific environment is needed:

```bash
PYTHON=/path/to/env/bin/python bash test.sh multi
```

## Custom Stories

Stories are defined in YAML. A single-subject story uses one `concept_token` and one `subject` description:

```yaml
animals:
  - concept_token: phoenix
    subject: a phoenix with bright orange feathers
    style: A fiery and majestic illustration of
    settings:
      - rising from fiery ashes
      - soaring through a glowing sky
```

For multiple subjects, use lists with the same order and length. Each concept token must appear in its corresponding subject description.

```yaml
multi_characters:
  - concept_token: [man, woman]
    subject: [a man in a suit, a woman in a red dress]
    style: A hyper-realistic digital painting of
    settings:
      - with a woman in the park
      - buying a flower with a woman
```

Generate any benchmark with:

```bash
python run_eosstory.py --benchmark benchmark/consistory+.yaml
```

`generate_all.sh` runs every YAML file in `benchmark/`. The fixed multi-subject example uses seed 44 and λ = 0.6; all other benchmarks use the paper defaults.

```bash
bash generate_all.sh
```

## OpenPose ControlNet

EOS-Story accepts precomputed OpenPose maps and does not extract poses from source photographs. Pose paths in YAML are resolved relative to that YAML file.

```yaml
humans:
  - concept_token: man
    subject: a man in a suit
    style: A hyper-realistic digital painting of
    settings:
      - standing in a park
      - sitting on a bench in a park
    pose:
      - ../poses/person_openpose.png
      - ../poses/sitting_openpose.png
```

The following pose layouts are supported:

- One map shared by all scenes.
- One map per scene, in `settings` order.
- Identity maps followed by scene maps, one map per generated image.

Command-line pose maps override the YAML values:

```bash
python run_eosstory.py \
  --benchmark benchmark/test_controlnet.yaml \
  --pose poses/person_openpose.png poses/sitting_openpose.png \
  --controlnet_scale 0.9
```

## Generation Settings

| Setting | Default |
| --- | --- |
| Backbone | SDXL-base-1.0 |
| Scheduler | EulerDiscreteScheduler |
| Resolution | 1024 × 1024 |
| Denoising steps | 50 |
| CFG scale | 7.0 |
| Seed | 12 |
| Identity Anchoring | Steps 2–9 |
| Identity-Aware Self-Attention | Steps 10–50 |
| Scene-preservation weight λ | 0.55 |
| EOS infusion | First EOS token |

Use `--seed` to change the random seed and `--sa_scale` to set λ in `[0, 1]`. ControlNet options are exposed through `--controlnet_model`, `--controlnet_scale`, `--control_guidance_start`, and `--control_guidance_end`.

## Code Structure

```text
EOS-Story_ACCV2026/
├── benchmark/               # Story definitions
├── poses/                   # Example OpenPose maps
├── figs/                    # README figures
├── results/                 # Generated scene images
├── run_eosstory.py          # Main generation flow
├── eosstory_utils.py        # YAML, prompt, token, latent, and I/O utilities
├── pipeline_eosstory.py     # SDXL pipeline and CCB
├── attention_processor.py   # IA and IASA
├── utils.py                 # Mask and correspondence helpers
├── test.sh
└── generate_all.sh
```

The paper components map directly to the implementation:

| Component | Implementation |
| --- | --- |
| Cross-Contextual Bridging | `pipeline_eosstory.py`: `cross_contextual_bridging()` |
| Identity Anchoring | `attention_processor.py`: `identity_anchoring()` |
| Identity-Aware Self-Attention | `attention_processor.py`: `identity_aware_self_attention()` |
| Prompt and token preparation | `eosstory_utils.py`: `prepare_story_inputs()` |

`run_eosstory.py` now follows a short path: load the benchmark, load SDXL, prepare one story, register IA/IASA, generate with CCB, and save scenes.

## Results

Only scene images are exported. Identity images are generated internally for EOS-Story but are not saved.

```text
results/
└── consistory_multi_seed44_sa0.6/
    └── multi_characters/
        └── multi_characters_0/
            └── result/
                ├── scene_0.png
                ├── scene_1.png
                └── ...
```

## Extensions

The same EOS-Story formulation supports pose-guided generation, different SDXL-family checkpoints, and stories containing multiple subjects.

<table>
  <tr>
    <td width="50%" rowspan="2" align="center" valign="top">
      <b>Model Variations</b><br>
      <sub>EOS-Story across Juggernaut-XL, Playground v2.5, and RealVisXL.</sub><br><br>
      <img src="figs/model_variation.png" width="100%" alt="EOS-Story results across different SDXL-family checkpoints">
    </td>
    <td width="50%" align="center" valign="top">
      <b>OpenPose ControlNet</b><br>
      <sub>Pose guidance with consistent subject appearance.</sub><br><br>
      <img src="figs/controlnet.png" width="100%" alt="EOS-Story results with OpenPose ControlNet">
    </td>
  </tr>
  <tr>
    <td width="50%" align="center" valign="top">
      <b>Multi-Subject Storytelling</b><br>
      <sub>Multiple identities preserved throughout a shared story.</sub><br><br>
      <img src="figs/multi.png" width="100%" alt="EOS-Story multi-subject qualitative results">
    </td>
  </tr>
</table>

## Acknowledgments

The SDXL pipeline is adapted from Hugging Face Diffusers. Identity-Aware Self-Attention builds on Consistory, and the benchmarks build on Consistory+ from One-Prompt-One-Story. Upstream attribution and licenses are included in `NOTICE` and `THIRD_PARTY_LICENSES/`.

## Citation

```bibtex
@inproceedings{cha2026eosstory,
  title     = {Not Just a Subject: Capturing Adaptive Scene-Specific Variance for Visual Storytelling},
  author    = {Cha, SeungJu and Kim, Ye-Chan and Lee, Kwanyoung and Kim, Dong-Jin},
  booktitle = {Asian Conference on Computer Vision},
  year      = {2026}
}
```
