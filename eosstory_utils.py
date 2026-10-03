"""Input preparation and reproducibility utilities for EOS-Story."""

import os
import random
import re
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import yaml
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image

from attention_processor import (
    AttentionStore,
    FeatureStore,
    register_eosstory_attention,
)


SPECIAL_TOKENS = {"<|startoftext|>", "<|endoftext|>"}


class StoryInputs(NamedTuple):
    """Prompts and subject-token positions consumed by the SDXL pipeline."""

    identity_prompts: list[str]
    style_prompts: list[str]
    scene_prompts: list[str]
    pooled_prompts: list[str]
    identity_token_positions: dict[str, list[int]]
    scene_token_positions: list[dict[str, list[int] | int]]


def load_benchmark(args, parser):
    """Validate a benchmark and resolve its precomputed pose-map paths."""
    args.benchmark = args.benchmark.expanduser().resolve()
    if not args.benchmark.is_file():
        parser.error(f"Benchmark not found: {args.benchmark}")

    with args.benchmark.open(encoding="utf-8") as handle:
        benchmark = yaml.safe_load(handle)
    if not isinstance(benchmark, dict) or not benchmark:
        parser.error(
            "Benchmark must be a nonempty mapping of categories to story lists."
        )

    for category, stories in benchmark.items():
        if (
            not isinstance(category, str)
            or not category
            or category in {".", ".."}
            or "/" in category
            or "\\" in category
        ):
            parser.error("Category names must be nonempty single path components.")
        if not isinstance(stories, list):
            parser.error(f"Category {category!r} must contain a list of stories.")
        for story in stories:
            _validate_story(story, category, args, parser)

    if args.max_cases is not None:
        remaining = args.max_cases
        benchmark = dict(benchmark)
        for category, stories in benchmark.items():
            benchmark[category] = stories[:remaining]
            remaining -= len(benchmark[category])

    if not any(benchmark.values()):
        parser.error("Benchmark has no story cases.")
    return benchmark


def _validate_story(story, category, args, parser):
    if not isinstance(story, dict):
        parser.error(f"Each story in {category!r} must be a mapping.")
    if not isinstance(story.get("style"), str):
        parser.error("Story style must be a string.")

    concepts = story.get("concept_token")
    identities = story.get("subject")
    if isinstance(concepts, list) or isinstance(identities, list):
        if (
            not isinstance(concepts, list)
            or not isinstance(identities, list)
            or not concepts
            or len(concepts) != len(identities)
        ):
            parser.error(
                "Multi-subject concept_token and subject must be nonempty lists of equal length."
            )
        if not all(
            isinstance(value, str) and value.strip() for value in concepts + identities
        ):
            parser.error(
                "Concept tokens and identity descriptions must be nonempty strings."
            )
        if len({concept.strip().casefold() for concept in concepts}) != len(concepts):
            parser.error("Multi-subject concept tokens must be distinct.")
    elif not all(
        isinstance(value, str) and value.strip() for value in [concepts, identities]
    ):
        parser.error(
            "Single-subject concept_token and subject must be nonempty strings."
        )

    scenes = story.get("settings")
    if (
        not isinstance(scenes, list)
        or not scenes
        or not all(isinstance(scene, str) and scene.strip() for scene in scenes)
    ):
        parser.error(
            "Each story must contain a nonempty settings list of scene strings."
        )

    pose_paths = args.pose if args.pose is not None else story.get("pose")
    if pose_paths is None:
        return
    if isinstance(pose_paths, (str, Path)):
        pose_paths = [pose_paths]
    identity_count = len(concepts) if isinstance(concepts, list) else 1
    if not isinstance(pose_paths, list) or len(pose_paths) not in {
        1,
        len(scenes),
        identity_count + len(scenes),
    }:
        parser.error(
            "Pose maps must be shared (1), per scene, or identities followed by scenes."
        )

    resolved_paths = []
    for pose_path in pose_paths:
        if not isinstance(pose_path, (str, Path)) or not str(pose_path).strip():
            parser.error("Pose entries must be nonempty local image paths.")
        pose_path = Path(pose_path).expanduser()
        if not pose_path.is_absolute() and args.pose is None:
            pose_path = args.benchmark.parent / pose_path
        pose_path = pose_path.resolve()
        if not pose_path.is_file():
            parser.error(f"Pose map not found: {pose_path}")
        resolved_paths.append(pose_path)
    story["pose"] = resolved_paths


def _tokenize(tokenizer, text, truncation=True):
    encoded = tokenizer(
        text,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=truncation,
        return_tensors="pt",
    )
    return tokenizer.convert_ids_to_tokens(encoded.input_ids[0])


def _content_tokens(tokenizer, text):
    return [
        token for token in _tokenize(tokenizer, text) if token not in SPECIAL_TOKENS
    ]


def _find_unique_token_positions(prompt_tokens, target_tokens):
    """Find the first unused prompt position for every target token in order."""
    positions = []
    used_positions = set()
    for target in target_tokens:
        for position, token in enumerate(prompt_tokens):
            if token == target and position not in used_positions:
                positions.append(position)
                used_positions.add(position)
                break
        else:
            return []
    return sorted(positions)


def _find_single_subject_positions(prompt_tokens, subject_tokens):
    """Preserve the original single-subject token matching behavior."""
    positions = []
    subject_set = set(subject_tokens)
    found = {token: False for token in subject_tokens}
    for position, token in enumerate(prompt_tokens):
        if token in subject_set and not found[token]:
            positions.append(position)
            found[token] = True
            if all(found.values()):
                return positions
    return []


def _indefinite_article(noun):
    return "an" if noun.lower().startswith(("a", "e", "i", "o", "u")) else "a"


def prepare_single_story(pipe, entry):
    """Build single-subject prompts and their CLIP token positions."""
    tokenizer = pipe.tokenizer
    concept = entry["concept_token"]
    identity = entry["subject"].lower()
    style = entry["style"]

    identity_prompts = [f"{style} {identity}"]
    style_prompts = [f"{style} {_indefinite_article(concept)} {concept}"]
    scene_prompts = [f"{style_prompts[0]} {scene}" for scene in entry["settings"]]
    pooled_prompts = identity_prompts + [
        f"{identity_prompts[0]} {scene}" for scene in entry["settings"]
    ]

    style_token_count = len(
        [
            token
            for token in _tokenize(tokenizer, style, False)
            if token not in SPECIAL_TOKENS
        ]
    )
    identity_tokens = _tokenize(tokenizer, identity, False)
    concept_tokens = _content_tokens(tokenizer, concept)
    positions = [
        position + style_token_count
        for position in _find_single_subject_positions(identity_tokens, concept_tokens)
    ]
    identity_token_positions = {concept: positions or [0]}

    scene_text = [prompt.replace(style, "") for prompt in scene_prompts]
    encoded_scenes = tokenizer(
        scene_text,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    scene_token_positions = []
    for token_ids in encoded_scenes.input_ids:
        scene_tokens = tokenizer.convert_ids_to_tokens(token_ids)
        positions = [
            position + style_token_count
            for position in _find_single_subject_positions(scene_tokens, concept_tokens)
        ]
        scene_token_positions.append({concept: positions or [0]})

    return StoryInputs(
        identity_prompts,
        style_prompts,
        scene_prompts,
        pooled_prompts,
        identity_token_positions,
        scene_token_positions,
    )


def prepare_multi_story(pipe, entry):
    """Build multi-subject prompts and their CLIP token positions."""
    tokenizer = pipe.tokenizer
    concepts = entry["concept_token"]
    identities = [identity.lower() for identity in entry["subject"]]
    style = entry["style"]

    identity_prompts = [f"{style} {identity}" for identity in identities]
    style_prompts = [
        f"{style} {_indefinite_article(concept)} {concept}" for concept in concepts
    ]
    scene_prompts = [f"{style_prompts[0]} {scene}" for scene in entry["settings"]]

    pooled_prompts = list(identity_prompts)
    for scene in entry["settings"]:
        full_scene = f"{style} {identities[0]} {scene}"
        for concept, identity in zip(concepts[1:], identities[1:]):
            full_scene = re.sub(
                r"\b(?:a|an|the)\s+" + re.escape(concept) + r"\b",
                lambda match: identity,
                full_scene,
                flags=re.IGNORECASE,
            )
        pooled_prompts.append(full_scene)

    concept_tokens = {
        concept: _content_tokens(tokenizer, concept) for concept in concepts
    }
    identity_token_positions = {}
    for concept, prompt in zip(concepts, identity_prompts):
        positions = _find_unique_token_positions(
            _tokenize(tokenizer, prompt), concept_tokens[concept]
        )
        if not positions:
            raise ValueError(
                f"Concept token {concept!r} was not found in its identity prompt."
            )
        identity_token_positions[concept] = positions

    scene_token_positions = []
    for prompt in scene_prompts:
        prompt_tokens = _tokenize(tokenizer, prompt)
        scene_token_positions.append(
            {
                concept: _find_unique_token_positions(
                    prompt_tokens, concept_tokens[concept]
                )
                or 0
                for concept in concepts
            }
        )

    return StoryInputs(
        identity_prompts,
        style_prompts,
        scene_prompts,
        pooled_prompts,
        identity_token_positions,
        scene_token_positions,
    )


def prepare_story_inputs(pipe, entry):
    if isinstance(entry["concept_token"], list):
        return prepare_multi_story(pipe, entry)
    return prepare_single_story(pipe, entry)


def story_subjects(entry):
    subjects = entry["concept_token"]
    return subjects if isinstance(subjects, list) else [subjects]


def register_story_attention(unet, story_inputs, subjects, scene_weight):
    """Register Identity Anchoring and Identity-Aware Self-Attention."""
    scene_count = len(story_inputs.scene_prompts)
    attention_store = AttentionStore(
        save=True,
        token_idx=story_inputs.scene_token_positions,
        subject=subjects,
        identity_idx=story_inputs.identity_token_positions,
        num_story=scene_count,
        save_res=[1024, 4096],
        mask_type="otsu",
        normalize=True,
    )
    feature_store = FeatureStore(
        subject=subjects,
        num_story=scene_count,
        save_res=[1024, 4096],
        blend_type="bbox",
    )
    register_eosstory_attention(
        unet,
        attnstore=attention_store,
        subject_num=len(subjects),
        token_idx=story_inputs.scene_token_positions,
        subject=subjects,
        identity_idx=story_inputs.identity_token_positions,
        injection_start=10,
        injection_end=50,
        sa_scale=scene_weight,
        IA_layer=[68, 139],
        IASA_layer=[68, 139],
        topa=None,
        model_type="SDXL",
        attn_res=[1024, 4096],
        mask_step=12,
        blending_start=2,
        blending_end=9,
        high_frequency=False,
        kernel=9,
        sigma=1.5,
        blend_type="bbox",
        feature_store=feature_store,
        cos_map_layer=127,
    )


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def create_shared_latents(pipe, seed, batch_size, device):
    """Sample the full batch, then share its first latent across all prompts."""
    generator = torch.Generator("cuda").manual_seed(seed)
    shape = (batch_size, pipe.unet.config.in_channels, 128, 128)
    latents = randn_tensor(
        shape, generator=generator, device=device, dtype=torch.float16
    )
    return latents[:1].repeat(batch_size, 1, 1, 1)


def load_pose_images(pose_paths):
    if pose_paths is None:
        return None
    pose_images = []
    for path in pose_paths:
        with Image.open(path) as image:
            pose_images.append(image.convert("RGB"))
    return pose_images


def save_scenes(images, directory):
    directory.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(images):
        image.save(directory / f"scene_{index}.png")
