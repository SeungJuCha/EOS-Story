# Copyright 2024 The HuggingFace Team. All rights reserved.
# Portions adapted from Hugging Face Diffusers 0.29.0 under Apache-2.0.
# See THIRD_PARTY_LICENSES/diffusers-LICENSE.txt.
# Modified for EOS-Story prompt infusion and story generation.

import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import (
    FromSingleFileMixin,
    IPAdapterMixin,
    StableDiffusionXLLoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from diffusers.models import (
    AutoencoderKL,
    ControlNetModel,
    ImageProjection,
    UNet2DConditionModel,
)
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    FusedAttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    is_invisible_watermark_available,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import (
    StableDiffusionXLPipelineOutput,
)

if is_invisible_watermark_available():
    from diffusers.pipelines.stable_diffusion_xl.watermark import (
        StableDiffusionXLWatermarker,
    )

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False
logger = logging.get_logger(__name__)


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(
        dim=list(range(1, noise_pred_text.ndim)), keepdim=True
    )
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    noise_cfg = (
        guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    )
    return noise_cfg


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return (timesteps, num_inference_steps)


class EOSStoryXLPipeline(
    DiffusionPipeline,
    StableDiffusionMixin,
    FromSingleFileMixin,
    StableDiffusionXLLoraLoaderMixin,
    TextualInversionLoaderMixin,
    IPAdapterMixin,
):
    """SDXL story generation with Cross-Contextual Bridging and optional OpenPose ControlNet. Identity prompts precede scene prompts; the runner exports scenes only."""

    model_cpu_offload_seq = (
        "text_encoder->text_encoder_2->image_encoder->controlnet->unet->vae"
    )
    _optional_components = [
        "tokenizer",
        "tokenizer_2",
        "text_encoder",
        "text_encoder_2",
        "image_encoder",
        "feature_extractor",
        "controlnet",
    ]
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
        "add_text_embeds",
        "add_time_ids",
        "negative_pooled_prompt_embeds",
        "negative_add_time_ids",
    ]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
        force_zeros_for_empty_prompt: bool = True,
        add_watermarker: Optional[bool] = None,
        controlnet: Optional[ControlNetModel] = None,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            controlnet=controlnet,
        )
        self.register_to_config(
            force_zeros_for_empty_prompt=force_zeros_for_empty_prompt
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_normalize=False,
            do_convert_rgb=True,
        )
        self.default_sample_size = self.unet.config.sample_size
        add_watermarker = (
            add_watermarker
            if add_watermarker is not None
            else is_invisible_watermark_available()
        )
        if add_watermarker:
            self.watermark = StableDiffusionXLWatermarker()
        else:
            self.watermark = None

    def encode_prompt(
        self,
        prompt: str,
        prompt_2: Optional[str] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[str] = None,
        negative_prompt_2: Optional[str] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        """
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        device = device or self._execution_device
        if lora_scale is not None and isinstance(
            self, StableDiffusionXLLoraLoaderMixin
        ):
            self._lora_scale = lora_scale
            if self.text_encoder is not None:
                if not USE_PEFT_BACKEND:
                    adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
                else:
                    scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None:
                if not USE_PEFT_BACKEND:
                    adjust_lora_scale_text_encoder(self.text_encoder_2, lora_scale)
                else:
                    scale_lora_layers(self.text_encoder_2, lora_scale)
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        tokenizers = (
            [self.tokenizer, self.tokenizer_2]
            if self.tokenizer is not None
            else [self.tokenizer_2]
        )
        text_encoders = (
            [self.text_encoder, self.text_encoder_2]
            if self.text_encoder is not None
            else [self.text_encoder_2]
        )
        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2
            prompt_embeds_list = []
            prompts = [prompt, prompt_2]
            sentence_length = []
            for (prompt, tokenizer, text_encoder) in zip(
                prompts, tokenizers, text_encoders
            ):
                if isinstance(self, TextualInversionLoaderMixin):
                    prompt = self.maybe_convert_prompt(prompt, tokenizer)
                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids
                eos_mask = text_input_ids == tokenizer.eos_token_id
                eos_indices = eos_mask.float().argmax(dim=1)
                sentence_length.append(eos_indices)
                untruncated_ids = tokenizer(
                    prompt, padding="longest", return_tensors="pt"
                ).input_ids
                if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and (
                    not torch.equal(text_input_ids, untruncated_ids)
                ):
                    removed_text = tokenizer.batch_decode(
                        untruncated_ids[:, tokenizer.model_max_length - 1 : -1]
                    )
                    logger.warning(
                        f"The following part of your input was truncated because CLIP can only handle sequences up to {tokenizer.model_max_length} tokens: {removed_text}"
                    )
                prompt_embeds = text_encoder(
                    text_input_ids.to(device), output_hidden_states=True
                )
                pooled_prompt_embeds = prompt_embeds[0]
                if clip_skip is None:
                    prompt_embeds = prompt_embeds.hidden_states[-2]
                else:
                    prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]
                prompt_embeds_list.append(prompt_embeds)
            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        zero_out_negative_prompt = (
            negative_prompt is None and self.config.force_zeros_for_empty_prompt
        )
        if (
            do_classifier_free_guidance
            and negative_prompt_embeds is None
            and zero_out_negative_prompt
        ):
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        elif do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt
            negative_prompt = (
                batch_size * [negative_prompt]
                if isinstance(negative_prompt, str)
                else negative_prompt
            )
            negative_prompt_2 = (
                batch_size * [negative_prompt_2]
                if isinstance(negative_prompt_2, str)
                else negative_prompt_2
            )
            uncond_tokens: List[str]
            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} != {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`: {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches the batch size of `prompt`."
                )
            else:
                uncond_tokens = [negative_prompt, negative_prompt_2]
            negative_prompt_embeds_list = []
            for (negative_prompt, tokenizer, text_encoder) in zip(
                uncond_tokens, tokenizers, text_encoders
            ):
                if isinstance(self, TextualInversionLoaderMixin):
                    negative_prompt = self.maybe_convert_prompt(
                        negative_prompt, tokenizer
                    )
                max_length = prompt_embeds.shape[1]
                uncond_input = tokenizer(
                    negative_prompt,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                negative_prompt_embeds = text_encoder(
                    uncond_input.input_ids.to(device), output_hidden_states=True
                )
                negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]
                negative_prompt_embeds_list.append(negative_prompt_embeds)
            negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)
        if self.text_encoder_2 is not None:
            prompt_embeds = prompt_embeds.to(
                dtype=self.text_encoder_2.dtype, device=device
            )
        else:
            prompt_embeds = prompt_embeds.to(dtype=self.unet.dtype, device=device)
        (bs_embed, seq_len, _) = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )
        if do_classifier_free_guidance:
            seq_len = negative_prompt_embeds.shape[1]
            if self.text_encoder_2 is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(
                    dtype=self.text_encoder_2.dtype, device=device
                )
            else:
                negative_prompt_embeds = negative_prompt_embeds.to(
                    dtype=self.unet.dtype, device=device
                )
            negative_prompt_embeds = negative_prompt_embeds.repeat(
                1, num_images_per_prompt, 1
            )
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(
            1, num_images_per_prompt
        ).view(bs_embed * num_images_per_prompt, -1)
        if do_classifier_free_guidance:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(
                1, num_images_per_prompt
            ).view(bs_embed * num_images_per_prompt, -1)
        if self.text_encoder is not None:
            if isinstance(self, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
                unscale_lora_layers(self.text_encoder, lora_scale)
        if self.text_encoder_2 is not None:
            if isinstance(self, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
                unscale_lora_layers(self.text_encoder_2, lora_scale)
        if not torch.equal(sentence_length[0], sentence_length[1]):
            pass
        return (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            sentence_length[0],
        )

    def encode_image(
        self, image, device, num_images_per_prompt, output_hidden_states=None
    ):
        dtype = next(self.image_encoder.parameters()).dtype
        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values
        image = image.to(device=device, dtype=dtype)
        if output_hidden_states:
            image_enc_hidden_states = self.image_encoder(
                image, output_hidden_states=True
            ).hidden_states[-2]
            image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(
                num_images_per_prompt, dim=0
            )
            uncond_image_enc_hidden_states = self.image_encoder(
                torch.zeros_like(image), output_hidden_states=True
            ).hidden_states[-2]
            uncond_image_enc_hidden_states = (
                uncond_image_enc_hidden_states.repeat_interleave(
                    num_images_per_prompt, dim=0
                )
            )
            return (image_enc_hidden_states, uncond_image_enc_hidden_states)
        else:
            image_embeds = self.image_encoder(image).image_embeds
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_embeds = torch.zeros_like(image_embeds)
            return (image_embeds, uncond_image_embeds)

    def prepare_ip_adapter_image_embeds(
        self,
        ip_adapter_image,
        ip_adapter_image_embeds,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
    ):
        if ip_adapter_image_embeds is None:
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]
            if len(ip_adapter_image) != len(
                self.unet.encoder_hid_proj.image_projection_layers
            ):
                raise ValueError(
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                )
            image_embeds = []
            for (single_ip_adapter_image, image_proj_layer) in zip(
                ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
            ):
                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                (single_image_embeds, single_negative_image_embeds) = self.encode_image(
                    single_ip_adapter_image, device, 1, output_hidden_state
                )
                single_image_embeds = torch.stack(
                    [single_image_embeds] * num_images_per_prompt, dim=0
                )
                single_negative_image_embeds = torch.stack(
                    [single_negative_image_embeds] * num_images_per_prompt, dim=0
                )
                if do_classifier_free_guidance:
                    single_image_embeds = torch.cat(
                        [single_negative_image_embeds, single_image_embeds]
                    )
                    single_image_embeds = single_image_embeds.to(device)
                image_embeds.append(single_image_embeds)
        else:
            repeat_dims = [1]
            image_embeds = []
            for single_image_embeds in ip_adapter_image_embeds:
                if do_classifier_free_guidance:
                    (
                        single_negative_image_embeds,
                        single_image_embeds,
                    ) = single_image_embeds.chunk(2)
                    single_image_embeds = single_image_embeds.repeat(
                        num_images_per_prompt,
                        *repeat_dims * len(single_image_embeds.shape[1:]),
                    )
                    single_negative_image_embeds = single_negative_image_embeds.repeat(
                        num_images_per_prompt,
                        *repeat_dims * len(single_negative_image_embeds.shape[1:]),
                    )
                    single_image_embeds = torch.cat(
                        [single_negative_image_embeds, single_image_embeds]
                    )
                else:
                    single_image_embeds = single_image_embeds.repeat(
                        num_images_per_prompt,
                        *repeat_dims * len(single_image_embeds.shape[1:]),
                    )
                image_embeds.append(single_image_embeds)
        return image_embeds

    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        prompt_2,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        negative_prompt_2=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        ip_adapter_image=None,
        ip_adapter_image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )
        if callback_steps is not None and (
            not isinstance(callback_steps, int) or callback_steps <= 0
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type {type(callback_steps)}."
            )
        if callback_on_step_end_tensor_inputs is not None and (
            not all(
                (
                    k in self._callback_tensor_inputs
                    for k in callback_on_step_end_tensor_inputs
                )
            )
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (
            not isinstance(prompt, str) and (not isinstance(prompt, list))
        ):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )
        elif prompt_2 is not None and (
            not isinstance(prompt_2, str) and (not isinstance(prompt_2, list))
        ):
            raise ValueError(
                f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}"
            )
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_2 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_2`: {negative_prompt_2} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    f"`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds` {negative_prompt_embeds.shape}."
                )
        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )
        if negative_prompt_embeds is not None and negative_pooled_prompt_embeds is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` also have to be passed. Make sure to generate `negative_pooled_prompt_embeds` from the same text encoder that was used to generate `negative_prompt_embeds`."
            )
        if ip_adapter_image is not None and ip_adapter_image_embeds is not None:
            raise ValueError(
                "Provide either `ip_adapter_image` or `ip_adapter_image_embeds`. Cannot leave both `ip_adapter_image` and `ip_adapter_image_embeds` defined."
            )
        if ip_adapter_image_embeds is not None:
            if not isinstance(ip_adapter_image_embeds, list):
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be of type `list` but is {type(ip_adapter_image_embeds)}"
                )
            elif ip_adapter_image_embeds[0].ndim not in [3, 4]:
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be a list of 3D or 4D tensors but is {ip_adapter_image_embeds[0].ndim}D"
                )

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def _get_add_time_ids(
        self,
        original_size,
        crops_coords_top_left,
        target_size,
        dtype,
        text_encoder_projection_dim=None,
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids)
            + text_encoder_projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features
        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    def upcast_vae(self):
        dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
                LoRAXFormersAttnProcessor,
                LoRAAttnProcessor2_0,
                FusedAttnProcessor2_0,
            ),
        )
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(dtype)
            self.vae.decoder.conv_in.to(dtype)
            self.vae.decoder.mid_block.to(dtype)

    def get_guidance_scale_embedding(
        self,
        w: torch.Tensor,
        embedding_dim: int = 512,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            w (`torch.Tensor`):
                Generate embedding vectors with a specified guidance scale to subsequently enrich timestep embeddings.
            embedding_dim (`int`, *optional*, defaults to 512):
                Dimension of the embeddings to generate.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                Data type of the generated embeddings.

        Returns:
            `torch.Tensor`: Embedding vectors with shape `(len(w), embedding_dim)`.
        """
        assert len(w.shape) == 1
        w = w * 1000.0
        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def clip_skip(self):
        return self._clip_skip

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def denoising_end(self):
        return self._denoising_end

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[
                Callable[[int, int, Dict], None],
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        identity_prompt: Optional[Union[str, List[str]]] = None,
        subject: Optional[Union[str, List[str]]] = None,
        style_prompt: Optional[Union[str, List[str]]] = None,
        token_idx=None,
        enable_ccb: bool = False,
        pooled_prompts: Optional[Union[str, List[str]]] = None,
        pose: Optional[PipelineImageInput] = None,
        controlnet_model: str = "thibaud/controlnet-openpose-sdxl-1.0",
        controlnet_conditioning_scale: float = 0.9,
        control_guidance_start: float = 0.0,
        control_guidance_end: float = 1.0,
        **kwargs,
    ):
        """Generate identities and scenes in one batch. enable_ccb adds each present subject's appearance residual to the first scene EOS. pose accepts precomputed OpenPose maps shared by scenes, one per scene, or one per identity and scene."""
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)
        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        original_size = original_size or (height, width)
        target_size = target_size or (height, width)
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )
        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = denoising_end
        self._interrupt = False
        self._do_classifier_free_guidance = True if guidance_scale > 1.0 else False
        story_bs = (
            len(prompt)
            if isinstance(prompt, list)
            else prompt_embeds.shape[0]
            if prompt is None and prompt_embeds is not None
            else 1
        )
        subject_bs = 0
        if identity_prompt is not None:
            if isinstance(prompt, str):
                prompt = [prompt]
            if isinstance(identity_prompt, str):
                identity_prompt = [identity_prompt]
                subject_bs = 1
                prompt = identity_prompt + prompt
            elif isinstance(identity_prompt, list):
                subject_bs = len(identity_prompt)
                prompt = identity_prompt + prompt
        if enable_ccb:
            if num_images_per_prompt != 1 or not self._do_classifier_free_guidance:
                raise ValueError(
                    "Identity injection requires num_images_per_prompt=1 and guidance_scale > 1."
                )
            if (
                not subject_bs
                or subject is None
                or len(subject) != subject_bs
                or (style_prompt is None)
                or (len(style_prompt) != subject_bs)
            ):
                raise ValueError(
                    "Provide one identity prompt and style prompt per subject."
                )
            if len(set(subject)) != len(subject):
                raise ValueError("Subject concept tokens must be distinct.")
            if token_idx is None or len(token_idx) != story_bs:
                raise ValueError("Provide one subject-token mapping per scene.")
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = self._execution_device
        if negative_prompt is not None:
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt]
                negative_prompt = negative_prompt * batch_size
            elif isinstance(negative_prompt, list):
                negative_prompt = negative_prompt * batch_size
        lora_scale = (
            self._cross_attention_kwargs.get("scale", None)
            if self._cross_attention_kwargs is not None
            else None
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            prompt_length,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self._do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self._clip_skip,
        )
        if enable_ccb:
            (style_embeds, _, _, _, style_lengths) = self.encode_prompt(
                prompt=style_prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=False,
            )
            style_eos_embeddings = {
                key: style_embeds[i, style_lengths[i], :]
                for (i, key) in enumerate(subject)
            }
            prompt_embeds = cross_contextual_bridging(
                prompt_embeds, prompt_length, story_bs, token_idx, style_eos_embeddings
            )
        if pooled_prompts is not None:
            (
                _,
                _,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
                _,
            ) = self.encode_prompt(
                prompt=pooled_prompts,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=self._do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt_2,
                lora_scale=lora_scale,
                clip_skip=self._clip_skip,
            )
        (timesteps, num_inference_steps) = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        control_image = None
        control_mask = None
        if pose is not None:
            if (
                not math.isfinite(controlnet_conditioning_scale)
                or controlnet_conditioning_scale < 0
            ):
                raise ValueError(
                    "controlnet_conditioning_scale must be finite and nonnegative."
                )
            if not 0 <= control_guidance_start < control_guidance_end <= 1:
                raise ValueError("Control guidance must satisfy 0 <= start < end <= 1.")
            control_image = self.control_image_processor.preprocess(
                pose, height=height, width=width
            )
            if control_image.shape[1] != 3:
                raise ValueError("Pose maps must have three RGB channels.")
            pose_count = control_image.shape[0]
            if pose_count == batch_size:
                control_mask = torch.ones(batch_size, device=device)
            else:
                if pose_count == 1:
                    control_image = control_image.repeat(story_bs, 1, 1, 1)
                elif pose_count != story_bs:
                    raise ValueError(
                        "Provide one shared pose, one pose per scene, or poses for all identities and scenes."
                    )
                control_image = torch.cat(
                    [
                        torch.zeros_like(control_image[:1]).repeat(subject_bs, 1, 1, 1),
                        control_image,
                    ]
                )
                control_mask = torch.cat(
                    [
                        torch.zeros(subject_bs, device=device),
                        torch.ones(story_bs, device=device),
                    ]
                )
            if self.controlnet is None:
                controlnet = ControlNetModel.from_pretrained(
                    controlnet_model, torch_dtype=self.unet.dtype
                )
                self.register_modules(controlnet=controlnet.to(device))
            control_image = control_image.to(device=device, dtype=self.controlnet.dtype)
            control_image = control_image.repeat_interleave(
                num_images_per_prompt, dim=0
            )
            control_mask = control_mask.repeat_interleave(num_images_per_prompt)
            if self._do_classifier_free_guidance:
                control_image = torch.cat([control_image, control_image])
                control_mask = torch.cat([control_mask, control_mask])
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim
        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids
        if self._do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat(
                [negative_pooled_prompt_embeds, add_text_embeds], dim=0
            )
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)
        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(
            batch_size * num_images_per_prompt, 1
        )
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        if (
            self._denoising_end is not None
            and isinstance(self._denoising_end, float)
            and (self._denoising_end > 0)
            and (self._denoising_end < 1)
        ):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - self._denoising_end * self.scheduler.config.num_train_timesteps
                )
            )
            num_inference_steps = len(
                list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps))
            )
            timesteps = timesteps[:num_inference_steps]
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self._guidance_scale - 1).repeat(
                batch_size * num_images_per_prompt
            )
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for (i, t) in enumerate(timesteps):
                if self.interrupt:
                    continue
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self._do_classifier_free_guidance
                    else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )
                added_cond_kwargs = {
                    "text_embeds": add_text_embeds,
                    "time_ids": add_time_ids,
                }
                controlnet_kwargs = {}
                if control_image is not None:
                    keep = float(
                        i / len(timesteps) >= control_guidance_start
                        and (i + 1) / len(timesteps) <= control_guidance_end
                    )
                    if keep and controlnet_conditioning_scale:
                        (down_samples, mid_sample) = self.controlnet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            controlnet_cond=control_image,
                            conditioning_scale=controlnet_conditioning_scale * keep,
                            guess_mode=False,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                        )
                        residual_mask = control_mask[:, None, None, None]
                        controlnet_kwargs = {
                            "down_block_additional_residuals": [
                                sample * residual_mask.to(sample.dtype)
                                for sample in down_samples
                            ],
                            "mid_block_additional_residual": mid_sample
                            * residual_mask.to(mid_sample.dtype),
                        }
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self._cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                    **controlnet_kwargs,
                )[0]
                if self._do_classifier_free_guidance:
                    (noise_pred_uncond, noise_pred_text) = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self._guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                if self._do_classifier_free_guidance and self._guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(
                        noise_pred,
                        noise_pred_text,
                        guidance_rescale=self._guidance_rescale,
                    )
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds
                    )
                    add_text_embeds = callback_outputs.pop(
                        "add_text_embeds", add_text_embeds
                    )
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )
                    add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)
                    negative_add_time_ids = callback_outputs.pop(
                        "negative_add_time_ids", negative_add_time_ids
                    )
                if i == len(timesteps) - 1 or (
                    i + 1 > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)
                if XLA_AVAILABLE:
                    xm.mark_step()
        if not output_type == "latent":
            needs_upcasting = (
                self.vae.dtype == torch.float16 and self.vae.config.force_upcast
            )
            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(
                    next(iter(self.vae.post_quant_conv.parameters())).dtype
                )
            if latents.dtype != self.vae.dtype:
                if torch.backends.mps.is_available():
                    self.vae = self.vae.to(latents.dtype)
            has_latents_mean = (
                hasattr(self.vae.config, "latents_mean")
                and self.vae.config.latents_mean is not None
            )
            has_latents_std = (
                hasattr(self.vae.config, "latents_std")
                and self.vae.config.latents_std is not None
            )
            if has_latents_mean and has_latents_std:
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean)
                    .view(1, 4, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                latents_std = (
                    torch.tensor(self.vae.config.latents_std)
                    .view(1, 4, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                latents = (
                    latents * latents_std / self.vae.config.scaling_factor
                    + latents_mean
                )
            else:
                latents = latents / self.vae.config.scaling_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents
        if not output_type == "latent":
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)
            image = self.image_processor.postprocess(image, output_type=output_type)
        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return StableDiffusionXLPipelineOutput(images=image)


def cross_contextual_bridging(
    prompt_embeds, prompt_length, story_bs, token_idx, style_eos_embeddings
):
    """CCB: add identity-minus-style appearance residuals to the first scene EOS."""
    subject_bs = prompt_embeds.shape[0] - story_bs
    identity_eos = {}
    for (i, key) in enumerate(style_eos_embeddings):
        identity_eos[key] = prompt_embeds[i, prompt_length[i], :]
    for i in range(story_bs):
        scene_index = i + subject_bs
        scene_eos = prompt_embeds[scene_index, prompt_length[scene_index], :]
        if subject_bs > 1:
            appearance_residuals = [
                identity_eos[key] - style_eos_embeddings[key]
                for key in token_idx[i]
                if token_idx[i][key] != 0
            ]
            if not appearance_residuals:
                continue
            updated_eos = scene_eos + sum(appearance_residuals)
            prompt_embeds[scene_index, prompt_length[scene_index], :] = updated_eos
            continue
        for key in token_idx[i]:
            if token_idx[i][key] != 0:
                updated_eos = scene_eos + identity_eos[key] - style_eos_embeddings[key]
                prompt_embeds[scene_index, prompt_length[scene_index], :] = updated_eos
    return prompt_embeds
