"""Generate SDXL stories with EOS-Story."""

import argparse
import math
from pathlib import Path

import torch
from diffusers import AutoencoderKL, EulerDiscreteScheduler
from diffusers.models import UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from eosstory_utils import (
    create_shared_latents,
    load_benchmark,
    load_pose_images,
    prepare_story_inputs,
    register_story_attention,
    save_scenes,
    set_seed,
    story_subjects,
)
from pipeline_eosstory import EOSStoryXLPipeline


REPO_ROOT = Path(__file__).resolve().parent
SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
NEGATIVE_PROMPT = "low quality, black and white, naked"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate visual stories with EOS-Story (SDXL)."
    )
    parser.add_argument(
        "--benchmark", type=Path, default=REPO_ROOT / "benchmark" / "consistory+.yaml"
    )
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument(
        "--sa_scale",
        type=float,
        default=0.55,
        help="Scene-preservation weight lambda (0 to 1).",
    )
    parser.add_argument(
        "--pose",
        type=Path,
        nargs="+",
        default=None,
        help="OpenPose map(s): shared, per scene, or identities followed by scenes.",
    )
    parser.add_argument(
        "--controlnet_model", default="thibaud/controlnet-openpose-sdxl-1.0"
    )
    parser.add_argument("--controlnet_scale", type=float, default=0.9)
    parser.add_argument("--control_guidance_start", type=float, default=0.0)
    parser.add_argument("--control_guidance_end", type=float, default=1.0)
    parser.add_argument(
        "--max_cases", type=int, default=None, help="Run only the first N story cases."
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate inputs and report output paths without loading SDXL.",
    )
    args = parser.parse_args()

    if not math.isfinite(args.sa_scale) or not 0 <= args.sa_scale <= 1:
        parser.error("--sa_scale must be finite and between 0 and 1.")
    if not math.isfinite(args.controlnet_scale) or args.controlnet_scale < 0:
        parser.error("--controlnet_scale must be finite and nonnegative.")
    if not 0 <= args.control_guidance_start < args.control_guidance_end <= 1:
        parser.error("Control guidance must satisfy 0 <= start < end <= 1.")
    if args.seed < 0:
        parser.error("--seed must be nonnegative.")
    if args.max_cases is not None and args.max_cases < 1:
        parser.error("--max_cases must be positive.")
    return args, parser


def load_sdxl_pipeline(device, torch_dtype):
    """Load SDXL components in the original initialization order."""
    variant = "fp16"
    scheduler = EulerDiscreteScheduler.from_pretrained(
        SDXL_MODEL,
        subfolder="scheduler",
        torch_dtype=torch_dtype,
        variant=variant,
    )
    vae = AutoencoderKL.from_pretrained(
        SDXL_MODEL, subfolder="vae", torch_dtype=torch_dtype
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        SDXL_MODEL,
        subfolder="tokenizer",
        torch_dtype=torch_dtype,
        variant=variant,
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        SDXL_MODEL,
        subfolder="tokenizer_2",
        torch_dtype=torch_dtype,
        variant=variant,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        SDXL_MODEL,
        subfolder="text_encoder",
        torch_dtype=torch_dtype,
        variant=variant,
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        SDXL_MODEL,
        subfolder="text_encoder_2",
        torch_dtype=torch_dtype,
        variant=variant,
    )
    unet = UNet2DConditionModel.from_pretrained(
        SDXL_MODEL, subfolder="unet", torch_dtype=torch_dtype, variant=variant
    )
    pipeline = EOSStoryXLPipeline(
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        unet=unet,
        scheduler=scheduler,
    )
    pipeline.to(device)
    return pipeline


def generate_story(pipeline, entry, args, device):
    """Prepare one story, enable EOS-Story components, and generate its scenes."""
    story = prepare_story_inputs(pipeline, entry)
    subjects = story_subjects(entry)

    assert len(story.scene_prompts) == len(story.scene_token_positions)
    assert (
        len(story.identity_prompts)
        == len(story.identity_token_positions)
        == len(subjects)
    )

    batch_size = len(story.identity_prompts) + len(story.scene_prompts)
    latents = create_shared_latents(pipeline, args.seed, batch_size, device)
    register_story_attention(pipeline.unet, story, subjects, args.sa_scale)

    generated_images = pipeline(
        latents=latents,
        prompt=story.scene_prompts,
        negative_prompt=[NEGATIVE_PROMPT],
        num_inference_steps=50,
        guidance_scale=7.0,
        enable_ccb=True,
        identity_prompt=story.identity_prompts,
        token_idx=story.scene_token_positions,
        subject=subjects,
        style_prompt=story.style_prompts,
        pooled_prompts=story.pooled_prompts,
        pose=load_pose_images(entry.get("pose")),
        controlnet_model=args.controlnet_model,
        controlnet_conditioning_scale=args.controlnet_scale,
        control_guidance_start=args.control_guidance_start,
        control_guidance_end=args.control_guidance_end,
    ).images

    return generated_images[len(subjects) :]


def output_directory(args, benchmark):
    has_pose = any(
        entry.get("pose") is not None
        for stories in benchmark.values()
        for entry in stories
    )
    suffix = "_pose" if has_pose else ""
    output_root = (
        REPO_ROOT / "results" / f"{args.benchmark.stem}_seed{args.seed}{suffix}"
    )
    if args.sa_scale != 0.55:
        output_root = output_root.with_name(f"{output_root.name}_sa{args.sa_scale}")
    return output_root


def benchmark_counts(benchmark):
    story_entries = [
        story for category_stories in benchmark.values() for story in category_stories
    ]
    return len(story_entries), sum(len(story["settings"]) for story in story_entries)


def main():
    args, parser = parse_args()
    benchmark = load_benchmark(args, parser)
    output_root = output_directory(args, benchmark)
    story_count, scene_count = benchmark_counts(benchmark)
    print(
        f"Generating {scene_count} scenes from {story_count} stories -> {output_root}"
    )

    if args.dry_run:
        return
    if not torch.cuda.is_available():
        raise RuntimeError(
            "SDXL generation requires a CUDA GPU. Use --dry_run to validate inputs."
        )

    device = torch.device("cuda")
    set_seed(args.seed)
    pipeline = load_sdxl_pipeline(device, torch.float16)

    with torch.no_grad():
        for category, stories in benchmark.items():
            for story_index, entry in enumerate(stories):
                images = generate_story(pipeline, entry, args, device)
                result_dir = (
                    output_root / category / f"{category}_{story_index}" / "result"
                )
                save_scenes(images, result_dir)
                del images
                torch.cuda.empty_cache()
                print(f"Saved {result_dir}")


if __name__ == "__main__":
    main()
