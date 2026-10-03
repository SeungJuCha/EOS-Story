"""Adaptive Identity Preservation: Identity Anchoring and Identity-Aware Self-Attention."""
from diffusers.models.attention_processor import AttnProcessor, Attention
import torch
from einops import rearrange
import torch.nn.functional as F
import math
from utils import cluster_mask_1d
from scipy.ndimage import label, find_objects
import numpy as np
from utils import gaussian_smooth, otsu_mask


def register_eosstory_attention(unet, **attention_config):
    """Share one EOS-Story processor across the UNet's attention layers."""
    processor = EOSStoryAttnProcessor(**attention_config)
    processors = {name: processor for name in unet.attn_processors.keys()}
    unet.set_attn_processor(processors)


class FeatureStore:
    def __init__(
        self,
        save_res=[64, 256, 1024, 4096, 16384],
        subject=None,
        num_story=None,
        blend_type="bbox",
    ):
        self.feature_store = {}
        self.correspondence_map = {res: {} for res in save_res}
        self.subject = subject
        self.num_story = num_story
        self.save_res = save_res
        self.blend_type = blend_type

    def __call__(
        self,
        feature,
        step,
        cur_att_layer,
        num_att_layers,
        cos_map_layer,
        topa=None,
        story_mask=None,
        subject_mask=None,
    ):
        if feature.shape[1] in self.save_res:
            if cur_att_layer == cos_map_layer:
                self.feature_store[feature.shape[1]] = feature
                self.get_correspondence_map(topa, story_mask, subject_mask)

    def get_correspondence_map(self, topa=None, story_mask=None, subject_mask=None):
        (bs, hw, c) = self.feature_store[1024].shape
        res = int(hw**0.5)
        subject_feat = {}
        for (i, sub) in enumerate(self.subject):
            subject_feat[sub] = self.feature_store[1024][i : i + 1]
        story_feature = self.feature_store[1024][len(self.subject) :]
        for sub in self.subject:
            sub_mask = subject_mask[sub]
            story_masks = story_mask[sub]
            sub_mask_expand = torch.zeros_like(story_masks, dtype=story_masks.dtype).to(
                story_masks.device
            )
            for (i, m) in enumerate(story_masks):
                if m.sum() != 0:
                    sub_mask_expand[i] = sub_mask
            if self.blend_type == "bbox":
                sub_bboxes = get_bbox_coords(sub_mask_expand.reshape(-1, res, res))
                story_bboxes = get_bbox_coords(story_masks.reshape(-1, res, res))
                sub_feats = subject_feat[sub].view(1, res, res, -1)
                cor_map_list = []
                for (i, (subj_bbox, story_bbox)) in enumerate(
                    zip(sub_bboxes, story_bboxes)
                ):
                    (sx0, sy0, sx1, sy1) = subj_bbox
                    (tx0, ty0, tx1, ty1) = story_bbox
                    (sx0, sy0, sx1, sy1) = map(
                        lambda x: max(0, min(x, res - 1)), [sx0, sy0, sx1, sy1]
                    )
                    (tx0, ty0, tx1, ty1) = map(
                        lambda x: max(0, min(x, res - 1)), [tx0, ty0, tx1, ty1]
                    )
                    target_h = min(ty1 - ty0 + 1, res)
                    target_w = min(tx1 - tx0 + 1, res)
                    crop_sub_mask = sub_mask.reshape(res, res)[
                        sy0 : sy1 + 1, sx0 : sx1 + 1
                    ]
                    resize_sub_mask = (
                        F.interpolate(
                            crop_sub_mask.float().unsqueeze(0).unsqueeze(0),
                            size=(target_h, target_w),
                            mode="nearest",
                        )
                        .squeeze(0)
                        .squeeze(0)
                        .to(crop_sub_mask.dtype)
                    )
                    sub_feat = sub_feats[:, sy0 : sy1 + 1, sx0 : sx1 + 1, :]
                    sub_feat_resized = (
                        F.interpolate(
                            sub_feat.permute(0, 3, 1, 2),
                            size=(target_h, target_w),
                            mode="bilinear",
                            align_corners=False,
                        )
                        .permute(0, 2, 3, 1)
                        .squeeze(0)
                    )
                    sub_feat_resized = sub_feat_resized.reshape(
                        -1, sub_feat_resized.shape[-1]
                    )[resize_sub_mask.reshape(-1), :]
                    story_mask_2d = story_masks[i].reshape(res, res).bool()
                    story_feat = story_feature[i, story_mask_2d.reshape(-1), :]
                    cor_map = self.compute_correspondence_map(
                        sub_feat_resized, story_feat, topa
                    )
                    cor_map_list.append(cor_map)
            elif self.blend_type == "mask":
                sub_mask_expand = sub_mask_expand.bool()
                story_masks = story_masks.bool()
                cor_map_list = []
                for (i, (subj_mask, story_mask)) in enumerate(
                    zip(sub_mask_expand, story_masks)
                ):
                    if not story_mask.any():
                        cor_map = torch.zeros_like(story_mask, dtype=torch.int64).to(
                            story_mask.device
                        )
                        cor_map_list.append(cor_map)
                        continue
                    else:
                        subj_feat = subject_feat[sub][:, subj_mask, :].squeeze(0)
                        story_feat = story_feature[i, story_mask, :]
                        cor_map = self.compute_correspondence_map(
                            subj_feat, story_feat, topa
                        )
                        cor_map_list.append(cor_map)
            self.correspondence_map[hw][sub] = cor_map_list

    def compute_correspondence_map(self, sub_feat, story_feat, topa):
        if story_feat.dim() == 3:
            (H, W, C) = sub_feat.shape
            sub_feat = sub_feat.reshape(-1, C)
            story_feat = story_feat.reshape(-1, C)
        if sub_feat.numel() == 0 or story_feat.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=story_feat.device)
        sub_feat = F.normalize(sub_feat, dim=1)
        story_feat = F.normalize(story_feat, dim=1)
        sim_matrix = torch.matmul(story_feat, sub_feat.T)
        (max_sim, dense_argmax) = torch.max(sim_matrix, dim=1)
        correspondence_map = dense_argmax.clone()
        if topa is not None:
            num_pixels = max_sim.numel()
            k = max(int(num_pixels * topa), 1)
            threshold = torch.topk(max_sim, k, largest=True).values[-1]
            valid_mask = max_sim >= threshold
        else:
            threshold = max_sim.mean() + max_sim.std()
            valid_mask = max_sim >= threshold
        correspondence_map[~valid_mask] = -1
        return correspondence_map


class AttentionStore:
    @staticmethod
    def get_empty_store():
        return {"cross_attnprob": [], "self_attnprob": []}

    def __init__(
        self,
        token_idx=None,
        identity_idx=None,
        save_res=[64, 256, 1024, 4096, 16384],
        iter=None,
        save=False,
        subject=None,
        num_story=None,
        mask_type="otsu",
        normalize=True,
    ):
        self.attention_store = {}
        self.step_store = self.get_empty_store()
        self.curr_step_index = 0
        self.idx = token_idx
        self.identity_idx = identity_idx
        self.subject = subject
        self.save_res = save_res
        self.save = save
        self.num_story = num_story
        self.reference_store = {}
        self.reference_mask = {}
        self.story_mask = None
        self.subject_mask = None
        self.mask_type = mask_type
        self.normalize = normalize

    def __call__(
        self,
        attention_map,
        hidden,
        is_cross,
        mask_step=5,
        cur_att_layer=None,
        num_att_layers=None,
        cos_map_layer=None,
        step=None,
    ):
        attn_type = "cross" if is_cross else "self"
        if step is not None:
            self.step = step
        if attention_map.shape[1] in self.save_res:
            self.step_store[attn_type + "_attnprob"].append(attention_map)
        if cur_att_layer == cos_map_layer:
            self.attention_store = self.step_store
            self.step_store = self.get_empty_store()
            (
                self.story_bbox,
                self.subject_bbox,
                self.story_mask,
                self.subject_mask,
            ) = self.attn_map()
            self.attention_store.clear()
        if cur_att_layer == num_att_layers * 2 - 1:
            self.step_store = self.get_empty_store()

    def attn_map(self):
        cross_prob_dict = {64: [], 256: [], 1024: [], 4096: [], 16384: []}
        self_prob_dict = {64: [], 256: [], 1024: [], 4096: [], 16384: []}
        CA = self.attention_store["cross_attnprob"]
        SA = self.attention_store["self_attnprob"]
        for attn_prob in CA:
            res = attn_prob.shape[1]
            (uc_attn_prob, c_attn_prob) = attn_prob.chunk(2, dim=0)
            cross_prob_dict[res].append(c_attn_prob)
        for attn_prob in SA:
            res = attn_prob.shape[1]
            (uc_attn_prob, c_attn_prob) = attn_prob.chunk(2, dim=0)
            self_prob_dict[res].append(c_attn_prob)
        story_bbox = {hw: None for hw in self.save_res}
        subject_bbox = {hw: None for hw in self.save_res}
        story_mask = {hw: None for hw in self.save_res}
        subject_mask = {hw: None for hw in self.save_res}
        base_story_mask_1024 = None
        base_subject_mask_1024 = None
        base_story_bbox_1024 = None
        base_subject_bbox_1024 = None

        def _upsample_binary_flat_mask(
            flat_mask: torch.Tensor, in_res: int, out_res: int
        ):
            """flat_mask: [B, in_res*in_res] or [in_res*in_res] with 0/1 values.
            returns tensor with shape [B, out_res*out_res] or [out_res*out_res], still binary.
            """
            squeeze_back = False
            if flat_mask.ndim == 1:
                flat_mask = flat_mask.unsqueeze(0)
                squeeze_back = True
            B = flat_mask.shape[0]
            m = flat_mask.view(B, 1, in_res, in_res).float()
            m_up = F.interpolate(m, size=(out_res, out_res), mode="nearest")
            m_up = m_up.view(B, out_res * out_res)
            return m_up.squeeze(0) if squeeze_back else m_up

        for hw in self.save_res:
            if hw == 1024:
                up_block_ca = cross_prob_dict[hw][40:49]
                up_block_sa = self_prob_dict[hw][40:49]
                up_block_st = [
                    torch.matmul(a, b) for (a, b) in zip(up_block_sa, up_block_ca)
                ]
                hw_story_bbox = {key: [] for key in self.subject}
                hw_story_mask = {key: [] for key in self.subject}
                hw_subject_bbox = {key: [] for key in self.subject}
                hw_subject_mask = {key: [] for key in self.subject}
                for attn_probs in up_block_st:
                    intermediate_mask = {key: [] for key in self.subject}
                    story_bs = self.num_story
                    total_bs = attn_probs.shape[0]
                    id_bs = total_bs - story_bs
                    for (i, (sub, sub_attn)) in enumerate(
                        zip(self.identity_idx.keys(), attn_probs[:id_bs])
                    ):
                        tokens = self.identity_idx[sub]
                        if len(tokens) == 1:
                            a = sub_attn[:, tokens[0]]
                        else:
                            a = torch.stack(
                                [sub_attn[:, token] for token in tokens], dim=0
                            ).mean(dim=0)
                        hw_subject_mask[list(hw_subject_mask.keys())[i]].append(a)
                    for (i, (id, attn)) in enumerate(zip(self.idx, attn_probs[id_bs:])):
                        zero_mask = torch.zeros_like(attn[:, 0])
                        for sub in id.keys():
                            if id[sub] == 0:
                                intermediate_mask[sub].append(zero_mask)
                            else:
                                tokens = id[sub]
                                if len(tokens) == 1:
                                    a = attn[:, tokens[0]]
                                else:
                                    a = torch.stack(
                                        [attn[:, token] for token in tokens], dim=0
                                    ).mean(dim=0)
                                intermediate_mask[sub].append(a)
                    for sub in intermediate_mask.keys():
                        assert (
                            len(intermediate_mask[sub]) == story_bs
                        ), "different masked region"
                        stack_mask = torch.stack(intermediate_mask[sub], dim=0)
                        hw_story_mask[sub].append(stack_mask)
                for sub in hw_story_mask.keys():
                    mask = torch.stack(hw_story_mask[sub], dim=0).mean(dim=0)
                    box_mask = []
                    for m in mask:
                        if m.sum() != 0:
                            if self.mask_type == "otsu":
                                m = torch.from_numpy(
                                    otsu_mask(m, normalize=self.normalize)
                                ).to(mask.device)
                            elif self.mask_type == "cluster":
                                m = (
                                    cluster_mask_1d(
                                        m, n_clusters=2, normalize=self.normalize
                                    )
                                    .int()
                                    .to(mask.device)
                                )
                            box_mask.append(m)
                        else:
                            box_mask.append(torch.zeros_like(m, dtype=torch.int))
                    mask = torch.stack(box_mask, dim=0)
                    hw_story_mask[sub] = mask
                    hw_story_bbox[sub] = self.get_bbox(mask)
                for sub in hw_subject_mask.keys():
                    mask = torch.stack(hw_subject_mask[sub], dim=0).mean(dim=0)
                    if self.mask_type == "otsu":
                        m = torch.from_numpy(
                            otsu_mask(mask, normalize=self.normalize)
                        ).to(mask.device)
                    elif self.mask_type == "cluster":
                        m = (
                            cluster_mask_1d(
                                mask, n_clusters=2, normalize=self.normalize
                            )
                            .int()
                            .to(mask.device)
                        )
                    hw_subject_mask[sub] = m
                    m = self.get_bbox(m)
                    hw_subject_bbox[sub] = m.squeeze(0)
                base_story_mask_1024 = hw_story_mask
                base_subject_mask_1024 = hw_subject_mask
                base_story_bbox_1024 = hw_story_bbox
                base_subject_bbox_1024 = hw_subject_bbox
                story_bbox[hw] = hw_story_bbox
                subject_bbox[hw] = hw_subject_bbox
                story_mask[hw] = hw_story_mask
                subject_mask[hw] = hw_subject_mask
            elif hw == 4096:
                assert (
                    base_story_mask_1024 is not None
                ), "1024-level masks must be computed before 4096 upsampling. Ensure self.save_res ordering includes 1024 before 4096."
                hw_story_mask = {}
                hw_story_bbox = {}
                hw_subject_mask = {}
                hw_subject_bbox = {}
                for (sub, m32) in base_story_mask_1024.items():
                    bs = m32.shape[0]
                    m64 = _upsample_binary_flat_mask(m32, in_res=32, out_res=64).view(
                        bs, 4096
                    )
                    hw_story_mask[sub] = m64.int().to(m32.device)
                    hw_story_bbox[sub] = self.get_bbox(hw_story_mask[sub])
                for (sub, m32) in base_subject_mask_1024.items():
                    m64 = _upsample_binary_flat_mask(m32, in_res=32, out_res=64).view(
                        4096
                    )
                    hw_subject_mask[sub] = m64.int().to(m32.device)
                    bb = self.get_bbox(hw_subject_mask[sub])
                    hw_subject_bbox[sub] = bb.squeeze(0)
                story_bbox[hw] = hw_story_bbox
                subject_bbox[hw] = hw_subject_bbox
                story_mask[hw] = hw_story_mask
                subject_mask[hw] = hw_subject_mask
            else:
                continue
        return (story_bbox, subject_bbox, story_mask, subject_mask)

    def get_bbox(self, mask):
        bbox = []
        if mask.ndim == 1:
            res = int(math.sqrt(mask.shape[0]))
            mask = mask.unsqueeze(0)
            bs = 1
        else:
            bs = mask.shape[0]
            res = int(math.sqrt(mask.shape[1]))
        for i in range(mask.shape[0]):
            mask_2d = mask[i].view(res, res).cpu().numpy()
            (labeled_mask, num_features) = label(mask_2d)
            if num_features == 0:
                bbox.append(
                    torch.zeros_like(torch.from_numpy(mask_2d), dtype=torch.int)
                )
                continue
            max_area = 0
            largest_bbox = None
            for j in range(1, num_features + 1):
                component = labeled_mask == j
                area = component.sum()
                if area > max_area:
                    max_area = area
                    slices = find_objects(component)[0]
                    (y_min, y_max) = (slices[0].start, slices[0].stop)
                    (x_min, x_max) = (slices[1].start, slices[1].stop)
                    largest_bbox = torch.tensor(
                        [x_min, y_min, x_max, y_max], dtype=torch.int
                    )
            bbox_mask = np.zeros_like(mask_2d, dtype=np.int32)
            (x_min, y_min, x_max, y_max) = largest_bbox
            bbox_mask[y_min:y_max, x_min:x_max] = 1
            bbox.append(torch.from_numpy(bbox_mask))
        bbox = torch.stack(bbox, dim=0).view(bs, -1).to(device=mask.device)
        return bbox


class EOSStoryAttnProcessor(AttnProcessor):
    Model_type = {"SD": 16, "SDXL": 70}

    def __init__(
        self,
        attnstore: AttentionStore,
        subject_num=None,
        token_idx=None,
        subject=None,
        identity_idx=None,
        injection_start=0,
        injection_end=50,
        topa=None,
        cos_map_layer=None,
        IA_layer=None,
        IASA_layer=None,
        layer_idx=None,
        sa_scale=None,
        total_steps=50,
        model_type="SDXL",
        attn_res=[64, 256, 1024, 4096, 16384],
        mask_step=12,
        blending_start=5,
        blending_end=15,
        blend_type="bbox",
        high_frequency=False,
        kernel=5,
        sigma=1,
        feature_store: FeatureStore = None,
        identity_anchoring: bool = True,
    ):
        super().__init__()
        self.total_steps = total_steps
        self.total_layers = self.Model_type.get(model_type, 16)
        self.start_step = injection_start
        self.end_step = injection_end
        self.IA_layer = IA_layer
        self.IASA_layer = IASA_layer
        self.subject_num = subject_num
        self.token_idx = token_idx
        self.subject = subject
        self.sa_scale = sa_scale
        self.identity_idx = identity_idx
        self.topa = topa
        self.cur_step = 0
        self.cur_att_layer = 0
        self.num_att_layers = self.total_layers
        self.attnstore = attnstore
        self.attn_res = attn_res
        self.mask_step = mask_step
        self.blending_start = blending_start
        self.blending_end = blending_end
        self.feature_store = feature_store
        self.cos_map_layer = cos_map_layer
        self.blend_type = blend_type
        self.high_frequency = high_frequency
        self.kernel = kernel
        self.sigma = sigma
        self.enable_identity_anchoring = identity_anchoring

    def update_num_attn_layers(self, num_layers):
        """
        Update the number of attention layers (called from register_attention_control)
        """
        self.num_attn_layers = num_layers

    def after_step(self):
        """
        Called after each denoising step (MasaCtrl style)
        """
        pass

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        out = self.compute_attention(
            attn, hidden_states, encoder_hidden_states, attention_mask
        )
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers * 2:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.after_step()
        return out

    def compute_attention(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        """
        Main injection control
        """
        cur_transformer_layer = self.cur_att_layer // 2
        residual = hidden_states
        (bs, seq_len, _) = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, seq_len, bs)
        Q = attn.to_q(hidden_states)
        is_cross = encoder_hidden_states is not None
        res = hidden_states.shape[2]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        if attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )
        K = attn.to_k(encoder_hidden_states)
        V = attn.to_v(encoder_hidden_states)
        Q = attn.head_to_batch_dim(Q)
        if (
            self.start_step <= self.cur_step <= self.end_step
            and (not is_cross)
            and (self.IASA_layer[0] <= self.cur_att_layer <= self.IASA_layer[1])
        ):
            hidden_states_out = self.identity_aware_self_attention(
                attn, Q, K, V, bs, seq_len
            )
        else:
            K = attn.head_to_batch_dim(K)
            V = attn.head_to_batch_dim(V)
            attn_probs = attn.get_attention_scores(Q, K, attention_mask)
            if (
                self.attnstore is not None
                and self.blending_start - 1 <= self.cur_step <= self.blending_end
            ):
                self.attnstore(
                    rearrange(attn_probs, "(b h) s t -> b h s t", h=attn.heads).mean(1),
                    encoder_hidden_states,
                    is_cross,
                    mask_step=self.cur_step,
                    cur_att_layer=self.cur_att_layer,
                    num_att_layers=self.num_att_layers,
                    cos_map_layer=self.cos_map_layer,
                    step=self.cur_step,
                )
            hidden_states_out = torch.bmm(attn_probs, V)
        hidden_states_out = attn.batch_to_head_dim(hidden_states_out)
        hidden_states_out = attn.to_out[0](hidden_states_out)
        hidden_states_out = attn.to_out[1](hidden_states_out)
        if (
            self.cur_att_layer == self.cos_map_layer
            and self.blending_start - 1 <= self.cur_step <= self.blending_end
            and (hidden_states_out.shape[1] in [1024])
        ):
            self.feature_store(
                hidden_states_out.chunk(2, dim=0)[1],
                self.cur_step,
                self.cur_att_layer,
                self.num_att_layers,
                self.cos_map_layer,
                self.topa,
                self.attnstore.story_mask[1024],
                self.attnstore.subject_mask[1024],
            )
        if (
            self.blending_start <= self.cur_step <= self.blending_end
            and hidden_states_out.shape[1] in [1024]
            and (self.enable_identity_anchoring == True)
        ):
            if not is_cross:
                hidden_states_out = self.identity_anchoring(
                    hidden_states_out, is_cross, hidden_states_out.shape[1]
                )
        if attn.residual_connection:
            hidden_states_out = hidden_states_out + residual
        hidden_states_out = hidden_states_out / attn.rescale_output_factor
        return hidden_states_out

    def identity_aware_self_attention(self, attn, Q, K, V, bs, seq_len):
        """Identity-Aware Self-Attention: attend to scene tokens and size-matched identity keys/values."""
        ex_out = torch.empty_like(Q)
        res = int(seq_len**0.5)
        attn_sub_mask = self.attnstore.subject_mask[seq_len]
        attn_story_mask = self.attnstore.story_mask[seq_len]
        subject_num = len(attn_sub_mask.keys())
        subjects = list(attn_sub_mask.keys())
        story_num = bs // 2 - subject_num
        subject_key_dict = {}
        subject_value_dict = {}
        attn_sub_box = self.attnstore.subject_bbox[seq_len]
        attn_story_box = self.attnstore.story_bbox[seq_len]
        for (i, sub) in enumerate(subjects):
            mask = attn_sub_box[sub]
            if mask.dtype != torch.bool:
                mask = mask.bool()
            subject_key_dict[sub] = K[i + bs // 2 : i + bs // 2 + 1, mask, :]
            subject_value_dict[sub] = V[i + bs // 2 : i + bs // 2 + 1, mask, :]
        for i in range(bs):
            start_idx = i * attn.heads
            end_idx = start_idx + attn.heads
            curr_q = Q[start_idx:end_idx]
            curr_k = K[i : i + 1]
            curr_v = V[i : i + 1]
            if i >= bs // 2 + subject_num:
                hidden_states_list = []
                for sub in subjects:
                    story_mask = attn_story_mask[sub][i - bs // 2 - subject_num]
                    sub_box = attn_sub_box[sub].reshape(res, res).unsqueeze(0)
                    sub_mask = attn_sub_mask[sub].reshape(res, res).unsqueeze(0)
                    story_box = (
                        attn_story_box[sub][i - bs // 2 - subject_num]
                        .reshape(res, res)
                        .unsqueeze(0)
                    )
                    sub_coords = get_bbox_coords(sub_box)[0]
                    story_coords = get_bbox_coords(story_box)[0]
                    sub_h = sub_coords[3] - sub_coords[1] + 1
                    sub_w = sub_coords[2] - sub_coords[0] + 1
                    story_h = story_coords[3] - story_coords[1] + 1
                    story_w = story_coords[2] - story_coords[0] + 1
                    sub_k = subject_key_dict[sub]
                    sub_v = subject_value_dict[sub]
                    if self.high_frequency:
                        sub_v_low = gaussian_smooth(
                            sub_v, kernel_size=self.kernel, sigma=self.sigma
                        )
                        sub_v_high = sub_v - sub_v_low
                    else:
                        sub_v_high = sub_v
                    sub_k = sub_k.contiguous().view(1, sub_h, sub_w, sub_k.shape[-1])
                    sub_v_high = sub_v_high.contiguous().view(
                        1, sub_h, sub_w, sub_v_high.shape[-1]
                    )
                    sub_k = sub_k.permute(0, 3, 1, 2)
                    sub_v_high = sub_v_high.permute(0, 3, 1, 2)
                    sub_k = F.interpolate(
                        sub_k,
                        size=(story_h, story_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    sub_v_high = F.interpolate(
                        sub_v_high,
                        size=(story_h, story_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    sub_k = (
                        sub_k.permute(0, 2, 3, 1)
                        .contiguous()
                        .view(1, -1, sub_k.shape[1])
                    )
                    sub_v_high = (
                        sub_v_high.permute(0, 2, 3, 1)
                        .contiguous()
                        .view(1, -1, sub_v_high.shape[1])
                    )
                    sub_mask = sub_mask[
                        :,
                        sub_coords[1] : sub_coords[3] + 1,
                        sub_coords[0] : sub_coords[2] + 1,
                    ]
                    sub_mask = F.interpolate(
                        sub_mask.unsqueeze(0).float(),
                        size=(story_h, story_w),
                        mode="nearest",
                    )
                    sub_mask = (
                        sub_mask.contiguous()
                        .view(1, 1, story_h * story_w, 1)
                        .permute(0, 2, 1, 3)
                        .squeeze(-1)
                        .bool()
                    )
                    sub_k = sub_k[sub_mask.expand_as(sub_k)].view(
                        1, -1, sub_k.shape[-1]
                    )
                    sub_v_high = sub_v_high[sub_mask.expand_as(sub_v_high)].view(
                        1, -1, sub_v_high.shape[-1]
                    )
                    new_k = torch.cat([curr_k, sub_k], dim=1)
                    new_v = torch.cat([curr_v, sub_v_high], dim=1)
                    new_k = attn.head_to_batch_dim(new_k).contiguous()
                    new_v = attn.head_to_batch_dim(new_v).contiguous()
                    attn_probs = attn.get_attention_scores(curr_q, new_k)
                    hidden_states = torch.bmm(attn_probs, new_v)
                    hidden_states_list.append(hidden_states)
                if len(hidden_states_list) > 1:
                    foreground_hd = hidden_states_list[0]
                    for j in range(1, len(hidden_states_list)):
                        sub = subjects[j]
                        story_sub_mask = self.attnstore.story_mask[seq_len][sub][
                            i - bs // 2 - subject_num
                        ]
                        story_sub_mask = story_sub_mask.bool()
                        channel_hd = hidden_states_list[j]
                        foreground_hd[:, story_sub_mask, :] = channel_hd[
                            :, story_sub_mask, :
                        ]
                    hidden_states = foreground_hd
                else:
                    hidden_states = hidden_states_list[0]
            else:
                curr_k = attn.head_to_batch_dim(curr_k).contiguous()
                curr_v = attn.head_to_batch_dim(curr_v).contiguous()
                attn_probs = attn.get_attention_scores(curr_q, curr_k)
                hidden_states = torch.bmm(attn_probs, curr_v)
            ex_out[start_idx:end_idx] = hidden_states
        hidden_states_out = ex_out
        del ex_out
        return hidden_states_out

    def identity_anchoring(self, hidden_states_out, is_cross, resolution):
        """Identity Anchoring: transfer aligned identity features into each scene subject."""
        story_bs = self.attnstore.num_story
        total_bs = hidden_states_out.shape[0] // 2
        (uc_hidden_states, c_hidden_states) = hidden_states_out.chunk(2, dim=0)
        id_bs = total_bs - story_bs
        subject_hidden_states = c_hidden_states[:id_bs]
        story_hidden_states = c_hidden_states[id_bs : id_bs + story_bs]
        subject_feature_dict = {key: None for key in self.subject}
        for (i, key) in enumerate(self.subject):
            subject_feature_dict[key] = subject_hidden_states[i]
        if resolution in self.attn_res:
            for key in self.subject:
                if (
                    not is_cross
                    and self.IA_layer[0] <= self.cur_att_layer <= self.IA_layer[1]
                ):
                    story_hidden_states = self.blend_identity_features(
                        key,
                        subject_feature_dict[key],
                        story_hidden_states,
                        resolution,
                        blend_type=self.blend_type,
                    )
        hidden_states_out = torch.cat(
            [uc_hidden_states, subject_hidden_states, story_hidden_states], dim=0
        )
        return hidden_states_out

    def blend_identity_features(
        self, key, subject_feature, story_hidden_states, resolution, blend_type="bbox"
    ):
        """Blend corresponding identity and scene features with the paper weight lambda."""
        (
            repositioned_story_hidden_states,
            story_bbox_coords,
        ) = self.align_identity_features(
            self.attnstore.story_mask[resolution][key],
            self.attnstore.subject_mask[resolution][key],
            subject_feature,
            story_hidden_states,
            mode="correspondence",
            topa=self.topa,
            key=key,
            blend_type=blend_type,
        )
        mask = self.attnstore.story_mask[resolution][key].bool()
        if mask.dim() == 2 and story_hidden_states.shape[-1] > 1:
            mask = mask.unsqueeze(-1).expand(-1, -1, story_hidden_states.shape[-1])
        story_hidden_states[mask] = (
            self.sa_scale * story_hidden_states[mask]
            + (1 - self.sa_scale) * repositioned_story_hidden_states[mask]
        )
        return story_hidden_states

    def align_identity_features(
        self,
        story_mask,
        subject_mask,
        subject_feature,
        c_img_hidden,
        mode="direct",
        topa=None,
        key=None,
        blend_type="bbox",
    ):
        import torch.nn.functional as F
        import numpy as np
        import torch

        def apply_correspondence_map(subj_feat, story_feat, correspondence_map):
            if subj_feat.dim() == 3 and story_feat.dim() == 3:
                (H, W, C) = story_feat.shape
                subj_feat = subj_feat.reshape(-1, C)
                story_feat = story_feat.reshape(-1, C)
            mapped_sub_feat = torch.zeros_like(story_feat, dtype=subj_feat.dtype).to(
                story_feat.device
            )
            valid_mask = correspondence_map != -1
            mapped_sub_feat[valid_mask] = subj_feat[correspondence_map[valid_mask]]
            return mapped_sub_feat

        story_bs = story_mask.shape[0]
        res = int(np.sqrt(story_mask.shape[1]))
        story_masks = story_mask.view(story_bs, res, res)
        subject_mask = subject_mask.view(res, res)
        subject_mask_expand = torch.zeros_like(story_masks, dtype=story_mask.dtype).to(
            story_mask.device
        )
        for (i, m) in enumerate(story_masks):
            if m.sum() != 0:
                subject_mask_expand[i] = subject_mask
        if blend_type == "bbox":
            story_bbox_coords = get_bbox_coords(story_masks)
            subject_bbox_coords = get_bbox_coords(subject_mask_expand)
            subject_feature_ = subject_feature.view(res, res, -1)
            mask_c_img_hidden_list = []
            for (i, (subj_bbox, story_bbox)) in enumerate(
                zip(subject_bbox_coords, story_bbox_coords)
            ):
                (sx0, sy0, sx1, sy1) = subj_bbox
                (tx0, ty0, tx1, ty1) = story_bbox
                if sx0 == sx1 == sy0 == sy1 == 0 or tx0 == tx1 == ty0 == ty1 == 0:
                    mask_c_img_hidden = torch.zeros_like(c_img_hidden[i]).to(
                        c_img_hidden.device
                    )
                    mask_c_img_hidden_list.append(mask_c_img_hidden)
                    continue
                (sx0, sy0, sx1, sy1) = map(
                    lambda x: max(0, min(x, res - 1)), [sx0, sy0, sx1, sy1]
                )
                (tx0, ty0, tx1, ty1) = map(
                    lambda x: max(0, min(x, res - 1)), [tx0, ty0, tx1, ty1]
                )
                crop_sub_mask = subject_mask[sy0 : sy1 + 1, sx0 : sx1 + 1]
                resize_sub_mask = (
                    F.interpolate(
                        crop_sub_mask.float().unsqueeze(0).unsqueeze(0),
                        size=(ty1 - ty0 + 1, tx1 - tx0 + 1),
                        mode="nearest",
                    )
                    .squeeze(0)
                    .squeeze(0)
                    .to(crop_sub_mask.dtype)
                )
                subj_feat = subject_feature_[sy0 : sy1 + 1, sx0 : sx1 + 1, :]
                target_h = min(ty1 - ty0 + 1, res)
                target_w = min(tx1 - tx0 + 1, res)
                subj_feat_resized = (
                    F.interpolate(
                        subj_feat.permute(2, 0, 1).unsqueeze(0),
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    .squeeze(0)
                    .permute(1, 2, 0)
                )
                subj_feat_resized = subj_feat_resized.reshape(
                    -1, subj_feat_resized.shape[-1]
                )[resize_sub_mask.reshape(-1).bool(), :]
                story_feat = c_img_hidden[i][story_masks[i].reshape(-1).bool(), :]
                if mode == "direct":
                    modified_story_feat = subj_feat_resized
                elif mode == "correspondence":
                    correspondence_map = self.feature_store.correspondence_map[
                        res * res
                    ][key][i]
                    modified_story_feat = apply_correspondence_map(
                        subj_feat_resized, story_feat, correspondence_map
                    )
                else:
                    raise ValueError(f"Unsupported mode: {mode}")
                mask_c_img_hidden = torch.zeros_like(c_img_hidden[i]).to(
                    c_img_hidden.device
                )
                mask_c_img_hidden[
                    story_masks[i].reshape(-1).bool()
                ] = modified_story_feat
                mask_c_img_hidden_list.append(mask_c_img_hidden)
        elif blend_type == "mask":
            story_bbox_coords = get_bbox_coords(story_masks)
            (b, h, w) = subject_mask_expand.shape
            subject_mask_expand = subject_mask_expand.reshape(b, -1).bool()
            story_masks = story_masks.reshape(b, -1).bool()
            mask_c_img_hidden_list = []
            for (i, (subj_mask, story_mask)) in enumerate(
                zip(subject_mask_expand, story_masks)
            ):
                if not story_mask.any():
                    mask_c_img_hidden = torch.zeros_like(c_img_hidden[i]).to(
                        c_img_hidden.device
                    )
                    mask_c_img_hidden_list.append(mask_c_img_hidden)
                    continue
                else:
                    subject_feat = subject_feature[subj_mask, :]
                    story_feat = c_img_hidden[i, story_mask, :]
                if mode == "direct":
                    modified_story_feat = subject_feat
                elif mode == "correspondence":
                    correspondence_map = self.feature_store.correspondence_map[
                        res * res
                    ][key][i]
                    modified_story_feat = apply_correspondence_map(
                        subject_feat, story_feat, correspondence_map
                    )
                else:
                    raise ValueError(f"Unsupported mode: {mode}")
                mask_c_img_hidden = torch.zeros_like(c_img_hidden[i]).to(
                    c_img_hidden.device
                )
                mask_c_img_hidden[story_mask.bool()] = modified_story_feat
                mask_c_img_hidden_list.append(mask_c_img_hidden)
        else:
            raise ValueError(f"Unsupported blend type: {self.blend_type}")
        repositioned_img_out_hidden_states = torch.stack(mask_c_img_hidden_list, dim=0)
        return (repositioned_img_out_hidden_states, story_bbox_coords)


def get_bbox_coords(masks):
    coords = []
    for i in range(masks.shape[0]):
        mask = masks[i]
        nonzero = mask.nonzero(as_tuple=False)
        if nonzero.numel() == 0:
            coords.append((0, 0, 0, 0))
        else:
            y_min = nonzero[:, 0].min()
            y_max = nonzero[:, 0].max()
            x_min = nonzero[:, 1].min()
            x_max = nonzero[:, 1].max()
            coords.append((x_min.item(), y_min.item(), x_max.item(), y_max.item()))
    return coords
