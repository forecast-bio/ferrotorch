//! Pretrained model registry.
//!
//! Contains metadata for known pretrained models. Model entries document where
//! weights come from (URLs), expected checksums, and parameter counts. The
//! registry is static and compiled into the binary.

/// Format of the serialized weights file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightsFormat {
    /// HuggingFace SafeTensors format (`.safetensors`).
    SafeTensors,
    /// Native ferrotorch state dict format (`.fts`).
    FerrotorchStateDict,
}

/// Metadata for a pretrained model.
///
/// Marked `#[non_exhaustive]` so future fields (e.g. license, license_url,
/// expected dtype, training resolution) can be added in a minor version
/// without breaking external code. External callers must use the registry
/// accessors ([`get_model_info`], [`list_models`]) — there is no public
/// constructor and struct-literal construction from outside this crate is
/// rejected at compile time. A workspace-level grep for `ModelInfo {`
/// outside `ferrotorch-hub/` returned zero hits at audit time.
///
/// # Note
///
/// `weights_sha256` must be a real, locally-computed digest of the file
/// hosted at `weights_url`. The download path
/// (`download::download_and_verify`) returns
/// `Err(FerrotorchError::InvalidArgument)` when the digest is the all-zero
/// placeholder (`"0".repeat(64)`) — this surfaces a missing pin rather
/// than silently skipping integrity verification (security audit #6,
/// follow-up #739). Calling [`crate::load_pretrained`] on a model whose
/// entry still carries the placeholder will therefore fail fast with a
/// descriptive error.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ModelInfo {
    /// Short identifier (e.g. `"resnet50"`).
    pub name: &'static str,
    /// Human-readable description.
    pub description: &'static str,
    /// URL where the pretrained weights can be downloaded.
    pub weights_url: &'static str,
    /// SHA-256 hex digest of the weights file for integrity verification.
    pub weights_sha256: &'static str,
    /// Serialization format of the weights file.
    pub format: WeightsFormat,
    /// Total number of learnable parameters.
    pub num_parameters: usize,
}

/// Static registry of known pretrained models.
///
/// Each entry pins the SHA-256 of the upstream weights file. The download
/// path (see `download::download_and_verify`) **fails fast** when an entry
/// still carries the all-zero placeholder digest — silent skip-on-placeholder
/// was the security audit's #6 finding (#739). Real checksums were
/// downloaded, hashed locally, and pinned in #749 (Section A); the URLs
/// point at authoritative public mirrors (mostly the `timm` HF org for
/// torchvision-style backbones, since the prior `ferrotorch/*` HF org did
/// not yet host real weights).
///
/// Any entry that retains the placeholder (`weights_sha256` =
/// `"0".repeat(64)`) is documented inline with a `// SHA-256 placeholder:
/// <reason>` comment naming the missing-mirror constraint. Such entries
/// will return `Err(InvalidArgument)` from `load_pretrained` until a real
/// public mirror in a format ferrotorch reads is identified.
static MODELS: &[ModelInfo] = &[
    ModelInfo {
        name: "resnet18",
        description: "ResNet-18 trained on ImageNet-1K (top-1 acc ~69.8%)",
        weights_url: "https://huggingface.co/timm/resnet18.a1_in1k/resolve/main/model.safetensors",
        weights_sha256: "80c49dee3da4822c009c5a7fe591e9223c5a2cfcf95a4067ca4dfb5a7b89c612",
        format: WeightsFormat::SafeTensors,
        num_parameters: 11_689_512,
    },
    ModelInfo {
        name: "resnet34",
        description: "ResNet-34 trained on ImageNet-1K (top-1 acc ~73.3%)",
        weights_url: "https://huggingface.co/timm/resnet34.a1_in1k/resolve/main/model.safetensors",
        weights_sha256: "829a220f9529d2b1ffdc8719d3273463001e0daf411eaf1916865d9db62c0ed2",
        format: WeightsFormat::SafeTensors,
        num_parameters: 21_797_672,
    },
    ModelInfo {
        name: "resnet50",
        description: "ResNet-50 trained on ImageNet-1K (top-1 acc ~76.1%)",
        weights_url: "https://huggingface.co/timm/resnet50.a1_in1k/resolve/main/model.safetensors",
        weights_sha256: "773525d5821de224f8f30c33377b7a795d7863e08522698200d3217d3f2a41bb",
        format: WeightsFormat::SafeTensors,
        num_parameters: 25_557_032,
    },
    ModelInfo {
        name: "vgg11",
        description: "VGG-11 trained on ImageNet-1K (top-1 acc ~69.0%)",
        weights_url: "https://huggingface.co/timm/vgg11.tv_in1k/resolve/main/model.safetensors",
        weights_sha256: "682c023808812148d4e60c76d80ad83dbeaf5d50eb1a952a0b836895003151e5",
        format: WeightsFormat::SafeTensors,
        num_parameters: 132_863_336,
    },
    ModelInfo {
        name: "vgg16",
        description: "VGG-16 trained on ImageNet-1K (top-1 acc ~71.6%)",
        weights_url: "https://huggingface.co/timm/vgg16.tv_in1k/resolve/main/model.safetensors",
        weights_sha256: "57b026918159a6bf9faf8405c3a551903768e7138989d9c6224a14227203fad8",
        format: WeightsFormat::SafeTensors,
        num_parameters: 138_357_544,
    },
    ModelInfo {
        name: "vit_b_16",
        description: "Vision Transformer (ViT-B/16) trained on ImageNet-1K (top-1 acc ~81.1%)",
        weights_url: "https://huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k/resolve/main/model.safetensors",
        weights_sha256: "32aa17d6e17b43500f531d5f6dc9bc93e56ed8841b8a75682e1bb295d722405b",
        format: WeightsFormat::SafeTensors,
        num_parameters: 86_567_656,
    },
    ModelInfo {
        name: "efficientnet_b0",
        description: "EfficientNet-B0 trained on ImageNet-1K (top-1 acc ~77.7%)",
        weights_url: "https://huggingface.co/timm/efficientnet_b0.ra_in1k/resolve/main/model.safetensors",
        weights_sha256: "d569899762ea9b1384ee07f4af64805cf8caa1c55f9253ebb1080dc40e87a2cd",
        format: WeightsFormat::SafeTensors,
        num_parameters: 5_288_548,
    },
    ModelInfo {
        name: "swin_tiny",
        description: "Swin Transformer Tiny trained on ImageNet-1K (top-1 acc ~81.2%)",
        weights_url: "https://huggingface.co/timm/swin_tiny_patch4_window7_224.ms_in1k/resolve/main/model.safetensors",
        weights_sha256: "fb01861f793143135fa0d6cd97b1631e4b33eaa3ee162bbea9e62de1c76ebac1",
        format: WeightsFormat::SafeTensors,
        num_parameters: 28_288_354,
    },
    ModelInfo {
        name: "convnext_tiny",
        description: "ConvNeXt Tiny trained on ImageNet-1K (top-1 acc ~82.1%)",
        weights_url: "https://huggingface.co/timm/convnext_tiny.fb_in1k/resolve/main/model.safetensors",
        weights_sha256: "08b9dc9c3a3a29421de7996761e176501896d1ae7fc3085cf56a643772329276",
        format: WeightsFormat::SafeTensors,
        num_parameters: 28_589_128,
    },
    ModelInfo {
        name: "unet",
        description: "U-Net for semantic segmentation (Carvana / generic)",
        weights_url: "https://huggingface.co/ferrotorch/unet/resolve/main/model.safetensors",
        // SHA-256 placeholder: no authoritative public SafeTensors mirror
        // identified for a Carvana/medical-style U-Net of the architecture
        // ferrotorch_vision::models::UNet expects (~31M params, 4 down + 4 up
        // blocks, output 1 channel). HF search of public unet repos returned
        // either gated repos, .pt-only checkpoints, or stable-diffusion UNets
        // (entirely different architecture). Tracked as a follow-up to #739;
        // until a mirror is published, `load_pretrained("unet")` returns
        // `Err(InvalidArgument)` with a clear message.
        weights_sha256: "0000000000000000000000000000000000000000000000000000000000000000",
        format: WeightsFormat::SafeTensors,
        num_parameters: 31_037_633,
    },
    ModelInfo {
        name: "yolo",
        description: "YOLOv3 backbone for object detection (Darknet-53)",
        // Darknet-53 is the canonical YOLOv3 backbone. We point at timm's
        // Darknet-53 ImageNet checkpoint (`darknet53.c2ns_in1k`); the
        // ferrotorch YOLO model loads only the backbone weights, so this is
        // the right artifact for `pretrained=true` semantics. Original
        // YOLOv3 detection-head weights from pjreddie's repo are .weights
        // (Darknet binary), not SafeTensors, and Ultralytics' HF YOLOv3
        // mirror is gated.
        weights_url: "https://huggingface.co/timm/darknet53.c2ns_in1k/resolve/main/model.safetensors",
        weights_sha256: "2c4e3810fa8d8f67764cdbf72a18b34d516bff223649870ae8a5653e6aadb890",
        format: WeightsFormat::SafeTensors,
        num_parameters: 61_949_149,
    },
    // CL-436: MobileNetV2, MobileNetV3-Small, DenseNet-121, Inception v3.
    // Parameter counts reflect the *real* architectures from the original
    // papers — our simplified implementations will differ slightly.
    ModelInfo {
        name: "mobilenet_v2",
        description: "MobileNetV2 trained on ImageNet-1K (top-1 acc ~72.0%)",
        weights_url: "https://huggingface.co/timm/mobilenetv2_100.ra_in1k/resolve/main/model.safetensors",
        weights_sha256: "55ea4fb3a96010b933710ebf44fc965b070d5cd054d3f35e44f2d1320b4a714e",
        format: WeightsFormat::SafeTensors,
        num_parameters: 3_504_872,
    },
    ModelInfo {
        name: "mobilenet_v3_small",
        description: "MobileNetV3-Small trained on ImageNet-1K (top-1 acc ~67.7%)",
        weights_url: "https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k/resolve/main/model.safetensors",
        weights_sha256: "46d2c063b18125884c48937afa4c49e18128869e52e8db96df48bf0a4d7ff697",
        format: WeightsFormat::SafeTensors,
        num_parameters: 2_542_856,
    },
    ModelInfo {
        name: "densenet121",
        description: "DenseNet-121 trained on ImageNet-1K (top-1 acc ~74.4%)",
        weights_url: "https://huggingface.co/timm/densenet121.tv_in1k/resolve/main/model.safetensors",
        weights_sha256: "c894c6d9caa317a8ca1942986dee7a16a86c77734a4d691d2abe05389cfef358",
        format: WeightsFormat::SafeTensors,
        num_parameters: 7_978_856,
    },
    ModelInfo {
        name: "inception_v3",
        description: "Inception v3 trained on ImageNet-1K (top-1 acc ~77.5%)",
        weights_url: "https://huggingface.co/timm/inception_v3.tv_in1k/resolve/main/model.safetensors",
        weights_sha256: "9c452c63a67fd03a38a78c401ba12d4be0e926df63c93403de3a29f6df20fefb",
        format: WeightsFormat::SafeTensors,
        num_parameters: 27_161_264,
    },
    // #1130: Faster R-CNN with ResNet-50 FPN backbone for object detection.
    // Pinned from torchvision 0.21 `FasterRCNN_ResNet50_FPN_Weights.COCO_V1`,
    // re-keyed into ferrotorch's `named_parameters()` layout by
    // `scripts/pin_pretrained_weights.py`. See the README at
    // `huggingface.co/ferrotorch/fasterrcnn_resnet50_fpn` for the verbatim
    // upstream license and the per-model state_dict mapping notes.
    // Parameter count matches torchvision's `sum(p.numel() ... )` for the
    // upstream pretrained model exactly.
    // #1141: re-pinned with FPN biases included. The previous safetensors
    // intentionally dropped torchvision's FPN bias params (8 × 256 floats)
    // because ferrotorch's FPN was built with `bias=false`; that drop was
    // the root cause of #1141 (FPN max-abs-diff ~0.77 at p2, propagating
    // through the RPN to a 924/1000 post-NMS proposal mismatch).
    ModelInfo {
        name: "fasterrcnn_resnet50_fpn",
        description: "Faster R-CNN with ResNet-50 + FPN backbone for object detection (#1130, COCO_V1; #1141 FPN-bias fix)",
        weights_url: "https://huggingface.co/ferrotorch/fasterrcnn_resnet50_fpn/resolve/main/model.safetensors",
        weights_sha256: "1d8a19e81e91f5ce86ce5a65127dda566d6ae1fb7e2e64596d1ecf373ed06494",
        format: WeightsFormat::SafeTensors,
        num_parameters: 41_810_455,
    },
    // #1130: Mask R-CNN with ResNet-50 FPN backbone + mask head.
    // Pinned from torchvision 0.21 `MaskRCNN_ResNet50_FPN_Weights.COCO_V1`.
    // #1141: re-pinned with FPN biases included (same fix as fasterrcnn).
    ModelInfo {
        name: "maskrcnn_resnet50_fpn",
        description: "Mask R-CNN with ResNet-50 + FPN backbone for instance segmentation (#1130, COCO_V1; #1141 FPN-bias fix)",
        weights_url: "https://huggingface.co/ferrotorch/maskrcnn_resnet50_fpn/resolve/main/model.safetensors",
        weights_sha256: "dc472afa1ba8bb321c142b05c7f4a6ca20ee0ae191087d4e8f1030af7cfb3d2e",
        format: WeightsFormat::SafeTensors,
        num_parameters: 44_456_562,
    },
    // #1130: DeepLabV3 with ResNet-50 dilated backbone + ASPP head.
    // Pinned from torchvision 0.21
    // `DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1`.
    ModelInfo {
        name: "deeplabv3_resnet50",
        description: "DeepLabV3 with ResNet-50 dilated backbone for semantic segmentation (#1130, COCO+VOC labels)",
        weights_url: "https://huggingface.co/ferrotorch/deeplabv3_resnet50/resolve/main/model.safetensors",
        weights_sha256: "88133b09e057aa20609436d8333ca812378a7bc727fe305275e2690eb2375dc1",
        format: WeightsFormat::SafeTensors,
        num_parameters: 42_004_074,
    },
    // #1130: FCN with ResNet-50 backbone + FCN head.
    // Pinned from torchvision 0.21
    // `FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1`.
    ModelInfo {
        name: "fcn_resnet50",
        description: "FCN with ResNet-50 backbone for semantic segmentation (#1130, COCO+VOC labels)",
        weights_url: "https://huggingface.co/ferrotorch/fcn_resnet50/resolve/main/model.safetensors",
        weights_sha256: "8419d91ad57f4156e3a6add39abd43caf0a3761083743fe0a5dddf470ffdabf7",
        format: WeightsFormat::SafeTensors,
        num_parameters: 35_322_218,
    },
    // #1146: LRASPP MobileNetV3-Large dilated backbone for lightweight
    // semantic segmentation. Pinned from torchvision 0.21
    // `LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1`
    // (21 classes — Pascal VOC label set). Mapped via
    // `scripts/pin_pretrained_weights.py::map_lraspp_keys` with 0 drops
    // (the IntermediateLayerGetter `backbone.X` ↔ `backbone.features.X`
    // remap is handled inside the mapper).
    ModelInfo {
        name: "lraspp_mobilenet_v3_large",
        description: "LRASPP with MobileNetV3-Large dilated backbone for lightweight semantic segmentation (#1146, COCO+VOC labels)",
        weights_url: "https://huggingface.co/ferrotorch/lraspp_mobilenet_v3_large/resolve/main/model.safetensors",
        weights_sha256: "00c25a1d022772ca379a48bccdd052829a5d18535a4fff49623a516ef500a159",
        format: WeightsFormat::SafeTensors,
        num_parameters: 3_221_538,
    },
    // #1130: SSD300 with VGG-16 backbone for object detection.
    // Pinned from torchvision 0.21 `SSD300_VGG16_Weights.COCO_V1`.
    ModelInfo {
        name: "ssd300_vgg16",
        description: "SSD300 with VGG-16 backbone for object detection (#1130, COCO_V1)",
        weights_url: "https://huggingface.co/ferrotorch/ssd300_vgg16/resolve/main/model.safetensors",
        weights_sha256: "2db78702af742ec5882bc62e068e5337f366bc1dc00f069c34bbce91c5109dfe",
        format: WeightsFormat::SafeTensors,
        num_parameters: 35_641_826,
    },
    // #1143: RetinaNet with ResNet-50 + FPN(P3-P7) backbone for object
    // detection. Pinned from torchvision 0.21
    // `RetinaNet_ResNet50_FPN_Weights.COCO_V1` (legacy / canonical, NOT
    // _v2). The 9 anchors-per-location config + LastLevelP6P7 stride-2 convs
    // distinguish this from FasterRCNN. Param count matches torchvision's
    // upstream pretrained sum exactly.
    ModelInfo {
        name: "retinanet_resnet50_fpn",
        description: "RetinaNet with ResNet-50 + FPN(P3-P7) backbone for object detection (#1143, COCO_V1)",
        weights_url: "https://huggingface.co/ferrotorch/retinanet_resnet50_fpn/resolve/main/model.safetensors",
        weights_sha256: "2f3593e7a2a1c15c5f2f7e6327e3c3d9de3cb4839922956ffec14b22f362b448",
        format: WeightsFormat::SafeTensors,
        num_parameters: 34_014_999,
    },
    // #1144: FCOS anchor-free one-stage detector with ResNet-50 + FPN(P3-P7)
    // backbone for object detection. Pinned from torchvision 0.21
    // `FCOS_ResNet50_FPN_Weights.COCO_V1`. Distinct from RetinaNet:
    // single anchor per cell + centerness branch + GroupNorm heads;
    // shares the FPN P3-P7 with LastLevelP6P7 structure.
    ModelInfo {
        name: "fcos_resnet50_fpn",
        description: "FCOS with ResNet-50 + FPN(P3-P7) backbone for anchor-free object detection (#1144, COCO_V1)",
        weights_url: "https://huggingface.co/ferrotorch/fcos_resnet50_fpn/resolve/main/model.safetensors",
        weights_sha256: "f6446fb9456ed6845f142eff160eae6b67313e6690079b4512a15e274d06e325",
        format: WeightsFormat::SafeTensors,
        num_parameters: 32_269_600,
    },
    // #1145: Keypoint R-CNN with ResNet-50 FPN backbone for COCO person
    // keypoint detection. Pinned from torchvision 0.21
    // `KeypointRCNN_ResNet50_FPN_Weights.COCO_V1`. Same FasterRCNN body as
    // #1141 (with FPN biases) but with `num_classes=2` (bg + person) for
    // the box predictor and an 8-conv KeypointRCNNHeads + single-deconv
    // KeypointRCNNPredictor outputting 17 keypoint heatmap channels. The
    // SHA-256 below is the placeholder pin; updated by
    // `scripts/pin_pretrained_weights.py keypointrcnn_resnet50_fpn` after
    // upload to ferrotorch/keypointrcnn_resnet50_fpn on HF.
    ModelInfo {
        name: "keypointrcnn_resnet50_fpn",
        description: "Keypoint R-CNN with ResNet-50 + FPN backbone for COCO person keypoint detection (#1145, COCO_V1)",
        weights_url: "https://huggingface.co/ferrotorch/keypointrcnn_resnet50_fpn/resolve/main/model.safetensors",
        weights_sha256: "73e282340493d58731dc08314df5f4f483fd537f55b3bb2fc188c17cfd922dfb",
        format: WeightsFormat::SafeTensors,
        num_parameters: 59_137_258,
    },
    // #1147: SmolLM-135M (HuggingFaceTB/SmolLM-135M) — first pinned
    // causal LM (Llama architecture, 135M params, tie_word_embeddings).
    // Apache-2.0 license. Mirrored byte-for-byte from upstream by
    // `scripts/pin_pretrained_llm_weights.py` (no key remapping required:
    // upstream uses the HF Llama naming convention that
    // `LlamaForCausalLM::load_hf_state_dict` already consumes; the pin
    // script verifies every key + shape against the ferrotorch-llama
    // expected set before upload). The mirror also ships
    // `_value_parity_{input,output,token_ids}.{txt,bin,json}` so the
    // `scripts/verify_causal_lm_inference.py` harness (and the
    // `conformance_pretrained_causal_lm` cargo test) can compare
    // ferrotorch's prefill logits against a frozen `transformers==4.50.3`
    // reference forward pass without re-running it in CI.
    ModelInfo {
        name: "smollm-135m",
        description: "SmolLM-135M (HuggingFaceTB/SmolLM-135M): 135M-param Llama-architecture causal LM, Apache 2.0, real-artifact baseline for causal LM parity vs transformers (#1147)",
        weights_url: "https://huggingface.co/ferrotorch/smollm-135m/resolve/main/model.safetensors",
        weights_sha256: "c7a387d6fe81ca6dd304aeb809bda3932ff1bbef3ca41c9484502f2f448dc093",
        format: WeightsFormat::SafeTensors,
        num_parameters: 134_515_008,
    },
    // #1148: all-MiniLM-L6-v2 (sentence-transformers/all-MiniLM-L6-v2) —
    // first pinned BERT-family encoder-only sentence-embedding model.
    // 22M params, 6 layers, 384 hidden, GELU FFN, post-norm residual,
    // learned absolute position embeddings. Sentence pipeline = mean
    // pool over attention mask + L2 normalize. Apache-2.0. Mirrored
    // byte-for-byte from upstream by `scripts/pin_pretrained_text_weights.py`
    // (HF key layout matches `BertModel::named_parameters()` exactly;
    // the pin script verifies every parameter key + shape and confirms
    // the only un-mapped upstream keys are `embeddings.position_ids`
    // (a buffer regenerated each forward) and `pooler.*` (unused by
    // sentence-transformers) — both intentionally dropped by
    // `BertModel::load_hf_state_dict` and surfaced in the returned
    // `DropReport` so the FPN-bias silent-drop bug (#1141) cannot
    // recur). The mirror also ships `_value_parity_{input,output,token_ids}`
    // so the `scripts/verify_text_embedding_inference.py` harness (and
    // the `conformance_pretrained_text_embedding` cargo test) can
    // compare ferrotorch's sentence embedding against a frozen
    // `sentence_transformers==5.4.1` reference forward pass without
    // re-running it in CI.
    ModelInfo {
        name: "all-MiniLM-L6-v2",
        description: "all-MiniLM-L6-v2 (sentence-transformers/all-MiniLM-L6-v2): 22M-param BERT-family sentence-embedding model, Apache 2.0, real-artifact baseline for sentence-embedding parity vs sentence_transformers (#1148)",
        weights_url: "https://huggingface.co/ferrotorch/all-MiniLM-L6-v2/resolve/main/model.safetensors",
        weights_sha256: "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db",
        format: WeightsFormat::SafeTensors,
        num_parameters: 22_565_376,
    },
    // #1149: whisper-tiny-encoder (openai/whisper-tiny) — first pinned
    // Whisper-family audio encoder. 4-layer 6-head encoder, d_model=384,
    // encoder_ffn_dim=1536, num_mel_bins=80, max_source_positions=1500,
    // pre-norm residual, GELU FFN, sinusoidal positional embedding
    // (shipped as a parameter in the state dict). MIT-licensed. The
    // mirror carries ONLY the encoder slice of openai/whisper-tiny — the
    // decoder + proj_out keys are dropped during the pin (see
    // `scripts/pin_pretrained_whisper_weights.py`). Mirrored
    // byte-for-byte from upstream for the encoder keys (HF layout
    // matches `WhisperEncoder::named_parameters()` exactly; the pin
    // script verifies every parameter key + shape and refuses to pin if
    // any encoder key is unmapped). The mirror also ships
    // `_value_parity_{audio,mel,encoder_output}.bin` so the
    // `scripts/verify_audio_encoder_inference.py` harness (and the
    // `conformance_pretrained_whisper_encoder` cargo test) can compare
    // ferrotorch's encoder output against a frozen `transformers==4.50.3`
    // reference forward pass without re-running it in CI.
    ModelInfo {
        name: "whisper-tiny-encoder",
        description: "whisper-tiny encoder (openai/whisper-tiny): 8.2M-param Whisper-family audio encoder, MIT, real-artifact baseline for audio encoder parity vs transformers (#1149)",
        weights_url: "https://huggingface.co/ferrotorch/whisper-tiny-encoder/resolve/main/model.safetensors",
        weights_sha256: "4ce29194b87ef05385203f8b09914f5c3b060200c2b503d6d420459ffb80a294",
        format: WeightsFormat::SafeTensors,
        num_parameters: 8_208_384,
    },
    // #1150: sd-v1-5-vae-decoder (runwayml/stable-diffusion-v1-5 vae/ subfolder)
    // — first Stable-Diffusion sub-model pin (Phase B.3a). Decoder half
    // of AutoencoderKL: post_quant_conv (Conv2d 4 -> 4, k=1) + Decoder
    // (conv_in 4 -> 512, UNetMidBlock2D with one 1-head spatial
    // attention sandwiched between two ResnetBlock2D at 512ch, four
    // UpDecoderBlock2D with 3 resnets each and nearest-2x upsample on
    // all but the last block, GroupNorm32 + SiLU + conv_out 128 -> 3).
    // ~49.5M-param decoder slice. CreativeML Open RAIL-M licensed. The
    // mirror carries ONLY the decoder slice of `vae/diffusion_pytorch_model.safetensors`
    // — the encoder + quant_conv keys are dropped during the pin (see
    // `scripts/pin_pretrained_diffusion_weights.py`). The deprecated
    // VAE-attention key names (`query/key/value/proj_attn`) are
    // renamed to the canonical `to_q/to_k/to_v/to_out.0` form during
    // the pin (mirroring diffusers's `_convert_deprecated_attention_blocks`
    // so the on-disk mirror does not need a Rust-side rename pass), and
    // the upstream `[C, C, 1, 1]` "conv-as-linear" weights are squeezed
    // to `[C, C]` to fit ferrotorch's `Linear` shape. Mirrored
    // byte-for-byte from upstream for the decoder keys; the pin script
    // verifies every parameter key + shape and refuses to pin if any
    // decoder key is unmapped. The mirror also ships
    // `_value_parity_{latent,image}.bin` so the
    // `scripts/verify_diffusion_inference.py` harness (and the
    // `conformance_pretrained_diffusion` cargo test) can compare
    // ferrotorch's decoded image against a frozen `diffusers==0.38.0`
    // reference forward pass without re-running it in CI.
    ModelInfo {
        name: "sd-v1-5-vae-decoder",
        description: "Stable Diffusion 1.5 VAE decoder (runwayml/stable-diffusion-v1-5 vae/): 49.5M-param decoder half of AutoencoderKL, RAIL-M, real-artifact baseline for SD VAE decoder parity vs diffusers (#1150)",
        weights_url: "https://huggingface.co/ferrotorch/sd-v1-5-vae-decoder/resolve/main/model.safetensors",
        weights_sha256: "5210b518f8d4e829355197aa79855c206678e91d13467a580123222c75c5a131",
        format: WeightsFormat::SafeTensors,
        num_parameters: 49_490_199,
    },
    // #1151: sd-v1-5-unet (runwayml/stable-diffusion-v1-5 unet/ subfolder)
    // — second Stable-Diffusion sub-model pin (Phase B.3b). Full
    // UNet2DConditionModel: time_embedding + conv_in 4 -> 320, four
    // down-blocks (CrossAttnDownBlock2D × 3 + DownBlock2D),
    // UNetMidBlock2DCrossAttn, four up-blocks (UpBlock2D +
    // CrossAttnUpBlock2D × 3), conv_norm_out + conv_out 320 -> 4.
    // ~860M-param noise predictor at the heart of SD-1.5 sampling.
    // CreativeML Open RAIL-M licensed. Mirrored byte-for-byte from
    // upstream — ferrotorch-diffusion's `UNet2DConditionModel`
    // consumes the diffusers key layout natively, so no key remap is
    // needed on the pin path. The pin script verifies every upstream
    // key sits under one of the seven UNet top-level prefixes
    // (`time_embedding. / conv_in. / down_blocks. / mid_block. /
    // up_blocks. / conv_norm_out. / conv_out.`) and refuses to pin
    // any architecture variant outside the SD-1.5 block alphabet. The
    // mirror also ships
    // `_value_parity_{noisy_latent,timestep,text_embedding,predicted_noise}.bin`
    // — a frozen 4-tuple probe so the
    // `scripts/verify_diffusion_inference.py` harness and the
    // `conformance_pretrained_diffusion` cargo test can compare
    // ferrotorch's predicted noise against a frozen `diffusers==0.38.0`
    // reference forward pass without re-running it in CI. The probe
    // uses a synthetic encoder_hidden_states (deterministic
    // `randn(1, 77, 768)`) because the CLIP text encoder is not yet
    // pinned (Phase B.3c).
    ModelInfo {
        name: "sd-v1-5-unet",
        description: "Stable Diffusion 1.5 UNet (runwayml/stable-diffusion-v1-5 unet/): 860M-param noise predictor (4 down-blocks, mid + 4 up-blocks, cross-attention to CLIP-ViT-L/14), RAIL-M, real-artifact baseline for SD UNet parity vs diffusers (#1151)",
        weights_url: "https://huggingface.co/ferrotorch/sd-v1-5-unet/resolve/main/model.safetensors",
        weights_sha256: "2a79ed44ee0eb33080c28498a200a3c79f112db86ddf5bfc81744793d56ab8b9",
        format: WeightsFormat::SafeTensors,
        num_parameters: 859_520_964,
    },
    // #1152: sd-v1-5-clip-text-encoder (runwayml/stable-diffusion-v1-5
    // text_encoder/ subfolder) — third and final Stable-Diffusion
    // sub-model pin (Phase B.3c). The text tower of CLIP-ViT-L/14:
    // CLIPTextEmbeddings (token + learned absolute position lookup),
    // 12 transformer layers (pre-LN, causal self-attention with
    // q/k/v/out projections all biased, QuickGELU MLP), and a final
    // LayerNorm. hidden_size=768, intermediate_size=3072,
    // num_attention_heads=12 (head_dim=64), max_position_embeddings=77,
    // vocab_size=49408, hidden_act=quick_gelu, layer_norm_eps=1e-5.
    // ~123M-param text conditioner; SD-1.5 feeds its
    // `last_hidden_state` straight into the UNet's cross-attention.
    // CreativeML Open RAIL-M licensed. Mirrored byte-for-byte from
    // upstream — ferrotorch-diffusion's `ClipTextEncoder` consumes the
    // HF key layout natively (modulo the int64
    // `text_model.embeddings.position_ids` buffer, which is
    // regenerated each forward and dropped via DropReport). The pin
    // script verifies the hidden_act is `quick_gelu`, refuses any
    // unknown key prefix, and dumps a tokenized parity probe
    // (`_value_parity_input_ids.bin` for the fixed prompt
    // "a photograph of an astronaut riding a horse" padded to 77, and
    // `_value_parity_last_hidden_state.bin` for the reference forward
    // output) so the harness can verify ferrotorch's
    // `last_hidden_state` against a frozen transformers reference
    // without re-running the upstream model.
    ModelInfo {
        name: "sd-v1-5-clip-text-encoder",
        description: "Stable Diffusion 1.5 CLIP text encoder (runwayml/stable-diffusion-v1-5 text_encoder/; the text tower of openai/clip-vit-large-patch14): 123M-param causal-self-attention text conditioner (12 layers, hidden=768, heads=12, QuickGELU MLP, max_pos=77, vocab=49408), RAIL-M, real-artifact baseline for SD CLIP text encoder parity vs transformers (#1152)",
        weights_url: "https://huggingface.co/ferrotorch/sd-v1-5-clip-text-encoder/resolve/main/model.safetensors",
        weights_sha256: "52de4b2426c9e31a63dadec5d111f766af7304b1ab205872b060c274727861de",
        format: WeightsFormat::SafeTensors,
        num_parameters: 123_060_480,
    },
    // #1155: optimizer-trajectories-v1 — Phase C.2 frozen-gradient
    // optimizer-step parity fixtures. The mirror is a fixture *bundle*
    // (`bundle.tar`) plus 130 individual `.bin`/`meta.json` files
    // partitioned across 10 per-config subfolders
    // (sgd_plain / sgd_momentum / sgd_nesterov / adam_default /
    //  adam_explicit / adamw_decoupled / rmsprop_default /
    //  rmsprop_momentum / adagrad_default / adagrad_explicit). Each
    // subfolder ships:
    //   * initial_params.bin     — params before step 0
    //   * gradients_step_K.bin   — frozen gradient at step K (K=0..9)
    //   * final_params.bin       — params after 10 torch.optim steps
    //   * meta.json              — config + shapes + dtype
    // The reference trajectories are produced by torch.optim against a
    // fixed 3-layer MLP (Linear(64,32) ReLU Linear(32,16) ReLU
    // Linear(16,8)) under MSELoss with seeded inputs/targets, so the
    // harness can verify ferrotorch's Optimizer::step math without any
    // autograd interaction on the ferrotorch side
    // (`scripts/verify_optimizer_inference.py` +
    //  `ferrotorch-optim/examples/optimizer_trajectory_dump.rs` +
    //  `ferrotorch-optim/tests/conformance_optimizer_trajectories.rs`).
    // `weights_url`/`weights_sha256` point at the tar bundle so this
    // registry entry has the same shape as the rest of the table; the
    // verify harness itself pulls individual files via hf_hub_download
    // (it does not call `download_and_verify` on the tar). The
    // FerrotorchStateDict format tag indicates "not a HF safetensors
    // checkpoint" — the bundle is a single-file convenience archive
    // and not consumed by the safetensors loader.
    ModelInfo {
        name: "optimizer-trajectories-v1",
        description: "Phase C.2 frozen-gradient optimizer trajectory fixtures: SGD/Adam/AdamW/RMSprop/Adagrad x 10 configs, 3-layer MLP (64-32-16-8), 10 steps each. Reference trajectories from torch.optim for byte-comparing ferrotorch Optimizer::step math. Apache 2.0; real-artifact baseline for Optimizer parity vs torch.optim (#1155).",
        weights_url: "https://huggingface.co/ferrotorch/optimizer-trajectories-v1/resolve/main/bundle.tar",
        weights_sha256: "81775049cad752d2650192e2d5ed3e51c2e1dd8effa9ad912d15c382de63c8ea",
        format: WeightsFormat::FerrotorchStateDict,
        num_parameters: 0,
    },
    // #1156: dataloader-batches-v1 — Phase C.3 DataLoader iteration
    // parity fixtures. The mirror is a fixture *bundle* (`bundle.tar`)
    // plus per-config subfolders (sequential / sequential_droplast /
    // shuffled_seeded / shuffled_droplast / batch_size_3). Each subfolder
    // ships one `meta.json` and one `batch_XXXX.bin` per batch
    // (each .bin is a multi-tensor file holding [B, 8] features and
    // [B] labels for that batch).
    //
    // The reference batch sequences are produced by
    // `torch.utils.data.DataLoader` against a fixed 10-item dict-style
    // dataset, so the harness can verify ferrotorch-data's
    // `DataLoader::iter` against torch's iteration without re-running
    // torch at verification time
    // (`scripts/verify_dataloader_inference.py` +
    //  `ferrotorch-data/examples/dataloader_iterate_dump.rs` +
    //  `ferrotorch-data/tests/conformance_dataloader_iteration.rs`).
    //
    // For shuffled configs the harness compares SET-equality (rust's
    // `rand` crate uses a different PRNG than torch's `torch.Generator`
    // so the *order* of items cannot byte-match; only the *multiset* of
    // items is asserted). Sequential configs use ORDER-equality. This
    // is the same trade-off documented in
    // `conformance_data_loader::dataloader_shuffle_coverage`.
    //
    // `weights_url`/`weights_sha256` point at the tar bundle so this
    // registry entry has the same shape as the rest of the table; the
    // verify harness itself pulls individual files via hf_hub_download
    // (it does not call `download_and_verify` on the tar).
    ModelInfo {
        name: "dataloader-batches-v1",
        description: "Phase C.3 DataLoader iteration parity fixtures: 5 torch.utils.data configs (sequential, sequential_droplast, shuffled_seeded, shuffled_droplast, batch_size_3) over a fixed 10-item dict dataset with [8]-dim f32 features and integer labels. Reference batch sequences from torch.utils.data.DataLoader for verifying ferrotorch-data's DataLoader iteration order, drop_last semantics, and shuffle item coverage. Apache 2.0; real-artifact baseline for DataLoader parity vs torch.utils.data (#1156).",
        weights_url: "https://huggingface.co/ferrotorch/dataloader-batches-v1/resolve/main/bundle.tar",
        weights_sha256: "c6a9f938f27f174b3fc74bd26f6083464c6bf37e3d4ab7ddaa0109c62bd15ce7",
        format: WeightsFormat::FerrotorchStateDict,
        num_parameters: 0,
    },
    // #1157: gcn-cora (PyG GCNConv-pair trained on Cora) — first
    // pinned graph neural network. Two-layer GCN matching
    // `torch_geometric.nn.GCNConv` defaults (add_self_loops=True,
    // normalize=True, improved=False, bias=True), hidden=16, trained
    // for 200 epochs with Adam(lr=0.01, weight_decay=5e-4) + cross-
    // entropy on the Cora train_mask under torch.manual_seed(42). MIT-
    // licensed (Planetoid Cora is freely available, the trained
    // weights are originally produced here). The mirror ships the
    // canonical PyG state_dict (`conv{1,2}.lin.weight`, `conv{1,2}.bias`
    // — no key drops or renames) plus `_value_parity_{x,edge_index,y,
    // logits}.bin` so the `scripts/verify_gnn_inference.py` harness
    // (and the `conformance_gcn_cora` cargo test) can compare
    // ferrotorch's full-graph logits against a frozen `torch_geometric==2.7.0`
    // reference forward pass without re-running the 200-epoch training
    // loop in CI. Frozen test_acc on Cora's test_mask: 80.4%.
    ModelInfo {
        name: "gcn-cora",
        description: "GCN-on-Cora (PyG GCNConv-pair, 1433->16->7, trained 200 epochs): first ferrotorch graph neural network real-artifact baseline, parity vs torch_geometric (#1157).",
        weights_url: "https://huggingface.co/ferrotorch/gcn-cora/resolve/main/model.safetensors",
        weights_sha256: "7566ea9e517959e3dd30ce006ac1cf542d72c805f6f63a996c9e537737890cdc",
        format: WeightsFormat::SafeTensors,
        num_parameters: 23_063,
    },
    // #1158: ppo-cartpole-v1 (sb3/ppo-CartPole-v1, mirrored byte-for-byte
    // from the canonical sb3 zoo zip): first reinforcement-learning policy
    // pinned to ferrotorch (Phase D.2). ActorCriticPolicy with
    // FlattenExtractor (identity on 1-D obs), two separate Tanh-MLP trunks
    // (`mlp_extractor.policy_net` and `mlp_extractor.value_net`, each
    // Linear(4 -> 64) -> Tanh -> Linear(64 -> 64) -> Tanh, no shared MLP
    // weights — only the FlattenExtractor is "shared"), and discrete
    // Categorical action head (`action_net: Linear(64 -> 2)`) + scalar
    // value head (`value_net: Linear(64 -> 1)`). 9_155 trainable f32
    // parameters total. Apache-2.0 (inherited from stable-baselines3).
    // No `log_std` parameter — discrete CartPole-v1 uses Categorical not
    // DiagGaussian. Mirrored byte-for-byte from upstream — ferrotorch-rl's
    // `MlpPolicy::named_parameters` returns exactly the sb3 key layout so
    // the pin needs no key remap. The pin script verifies every upstream
    // key + shape against the expected 12-key set and refuses to upload
    // any architecture variant outside the discrete-action MlpPolicy
    // alphabet. The mirror also ships
    // `_value_parity_{obs,action_logits,value}.bin` so the
    // `scripts/verify_rl_inference.py` harness (and the
    // `conformance_ppo_cartpole` cargo test) can compare ferrotorch's
    // forward pass against a frozen `stable_baselines3==2.8.0` reference
    // without re-running the upstream policy in CI.
    ModelInfo {
        name: "ppo-cartpole-v1",
        description: "PPO MlpPolicy for CartPole-v1 (sb3/ppo-CartPole-v1): 9.2k-param ActorCriticPolicy (4 -> 64 -> 64 Tanh trunks + Categorical action head + scalar value head), Apache 2.0, real-artifact baseline for RL policy parity vs stable_baselines3 (#1158).",
        weights_url: "https://huggingface.co/ferrotorch/ppo-cartpole-v1/resolve/main/model.safetensors",
        weights_sha256: "89c360d918f0e0582761cb8c0ecb9f2ed48606cd839ac5765d84a6df6b4d3769",
        format: WeightsFormat::SafeTensors,
        num_parameters: 9_155,
    },
    // #1159: ml-sklearn-parity-v1 — Phase D.3 sklearn parity fixtures for
    // the tabular/classical-ML gap. The mirror is a fixture *bundle*
    // (`bundle.tar`) plus 5 per-config subfolders (pca_n4 /
    // standard_scaler / one_hot_encoder / kfold_5 /
    // train_test_split_80_20). Each config ships its inputs + sklearn
    // reference outputs in the same `[u32 num_tensors]` +
    // `[u32 ndim][u32 shape][f32]` multi-tensor format the dataloader /
    // optimizer pins use, plus a `meta.json` recording the config and
    // tolerance.
    //
    // Two configs (kfold_5, train_test_split_80_20) ship JSON-only
    // outputs (integer index lists) because the rust `rand` SmallRng
    // shuffles in a different order than numpy's PRNG — these are
    // verified with SET-equality, not ORDER-equality (Option B from
    // #1156). The remaining 3 configs are ORDER/MAX_ABS comparable:
    // PCA tolerates per-PC sign flip and uses cosine_sim ≥ 0.9999;
    // StandardScaler and OneHotEncoder are essentially exact f32
    // arithmetic (both sklearn and ferrolearn use biased variance /n;
    // OneHotEncoder is integer-valued).
    //
    // `weights_url`/`weights_sha256` point at the tar bundle so this
    // registry entry has the same shape as the other parity bundles;
    // the verify harness itself pulls individual files via
    // `hf_hub_download` and does not consume the tar. The
    // FerrotorchStateDict format tag indicates "not a HF safetensors
    // checkpoint" — the bundle is a single-file convenience archive
    // and not consumed by the safetensors loader. Companion files:
    //   * `scripts/pin_pretrained_ml_fixtures.py`
    //   * `scripts/verify_ml_inference.py`
    //   * `ferrotorch-ml/examples/ml_op_dump.rs`
    //   * `ferrotorch-ml/tests/conformance_sklearn_parity.rs`
    ModelInfo {
        name: "ml-sklearn-parity-v1",
        description: "Phase D.3 sklearn parity fixtures: PCA(n=4), StandardScaler, OneHotEncoder, KFold(5,shuffle,rs=42), train_test_split(0.2,rs=42). 5 configs over a fixed deterministic dataset (np.random.RandomState(42).randn(100,10) + i%4 labels). BSD-3-Clause; real-artifact baseline for ferrotorch-ml vs scikit-learn (#1159).",
        weights_url: "https://huggingface.co/ferrotorch/ml-sklearn-parity-v1/resolve/main/bundle.tar",
        weights_sha256: "baafb9b5a10669cad13c39320923f9fa5482291dac8ab2395503390bc4bc4a3e",
        format: WeightsFormat::FerrotorchStateDict,
        num_parameters: 0,
    },
    // #1161: training-trajectory-v1 — Phase E end-to-end training parity
    // fixtures. The mirror is a fixture *bundle* (`bundle.tar`) plus the
    // following individual files:
    //   * initial_state.safetensors / epoch_0_state.safetensors  — start
    //   * epoch_{1..5}_state.safetensors                         — post-epoch
    //   * X_full.bin / y_full.bin                                — dataset
    //   * meta.json                                              — config
    //
    // The reference trajectory is produced by running `torch.optim.Adam`
    // against a fixed 3-layer MLP (Linear(64,32) ReLU Linear(32,16) ReLU
    // Linear(16,8)) under `F.mse_loss(reduction='mean')` with sequential
    // iteration over a deterministic 100-sample regression dataset
    // (batch_size=4, drop_last=False — 25 batches per epoch, 5 epochs,
    // 125 optimizer steps total). The per-epoch state_dict captures the
    // *combined* behavior of forward (linear + relu), loss (MSE mean),
    // backward (live autograd, not frozen gradients), optimizer (Adam
    // state initialization + per-step update math), and DataLoader
    // sequential iteration. If any one of those diverges from torch the
    // harness will catch it as state_dict drift after epoch K.
    //
    // Distinguishing this from #1155: that pin verifies optimizer math
    // *in isolation* by replaying frozen gradients, so a divergence
    // there fingers an optimizer. This pin verifies the full integrated
    // stack with live autograd, so a divergence here fingers autograd /
    // loss / dataloader integration *given* the optimizers already pass
    // their frozen-gradient gate. Companion files:
    //   * `scripts/pin_pretrained_training_trajectory.py`
    //   * `scripts/verify_training_trajectory.py`
    //   * `ferrotorch-train/examples/multi_epoch_train_dump.rs`
    //   * `ferrotorch-train/tests/conformance_multi_epoch_training.rs`
    //
    // `weights_url`/`weights_sha256` point at the tar bundle so this
    // registry entry has the same shape as the rest of the parity
    // bundles; the verify harness itself pulls individual files via
    // `hf_hub_download` and does not consume the tar. The
    // FerrotorchStateDict format tag indicates "not a HF safetensors
    // checkpoint" — the bundle is a single-file convenience archive
    // (the per-epoch state_dicts inside *are* safetensors files but
    // they are pulled separately, not via this registry entry's URL).
    ModelInfo {
        name: "training-trajectory-v1",
        description: "Phase E multi-epoch training-trajectory fixtures: 3-layer MLP (64-32-16-8) trained with Adam(lr=1e-3) + MSE on a deterministic 100-sample dataset for 5 epochs (25 batches/epoch, batch_size=4, sequential). Reference state_dicts from torch + autograd for verifying ferrotorch's full training stack (forward + loss + backward + optimizer + dataloader) against torch. Apache 2.0; real-artifact baseline for end-to-end training parity vs torch (#1161).",
        weights_url: "https://huggingface.co/ferrotorch/training-trajectory-v1/resolve/main/bundle.tar",
        weights_sha256: "0bf88f958b75f0bf4d5b8806bc3cf55f3563c8d46114c966b5f1e30acd661bb7",
        format: WeightsFormat::FerrotorchStateDict,
        num_parameters: 0,
    },
    // #1163: sd-v1-5-generation-trajectory — Phase F end-to-end SD-1.5
    // text-to-image generation parity fixtures. The mirror is a
    // fixture *bundle* (`bundle.tar`) plus per-stage individual files:
    //   * prompt_input_ids.bin   — `[1, 77]` CLIP-BPE tokenized prompt
    //                              "a photograph of an astronaut riding a horse"
    //   * uncond_input_ids.bin   — `[1, 77]` CLIP-BPE tokenized empty prompt
    //   * cond_embeds.bin        — `[1, 77, 768]` CLIP last_hidden_state
    //                              for the prompt
    //   * uncond_embeds.bin      — `[1, 77, 768]` CLIP last_hidden_state
    //                              for the empty negative prompt
    //   * init_latent.bin        — `[1, 4, 64, 64]` Gaussian noise from
    //                              torch.Generator(device='cpu').manual_seed(42)
    //                              (the rust pipeline reads this back because
    //                              rust's `rand::StdRng` PRNG does not match
    //                              torch's; cf. #1156 sklearn shuffled
    //                              configs which take the same approach)
    //   * step_K_{noise_pred_uncond,noise_pred_cond,guided_noise,latent_after}.bin
    //                            — per-step UNet outputs (unconditional +
    //                              conditional), the CFG-blended noise
    //                              (uncond + 7.5 * (cond - uncond)), and the
    //                              latent emitted by the DDIM scheduler
    //                              step. K runs 0..3 for the 4-step recipe.
    //   * final_image.bin        — `[1, 3, 512, 512]` decoded image in
    //                              [-1, 1] from `vae.decode(latent / 0.18215)`
    //   * meta.json              — prompt / seed / steps / guidance scale /
    //                              timesteps list / scheduler config string
    //
    // The reference trajectory is produced by running
    // `diffusers.StableDiffusionPipeline.from_pretrained(
    //    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32,
    //    safety_checker=None,
    // )` with `DDIMScheduler.from_config(pipe.scheduler.config)` for
    // `num_inference_steps=4, guidance_scale=7.5` on the fixed prompt
    // above. The 4-step recipe yields `timesteps=[751, 501, 251, 1]`
    // (Leading spacing + `steps_offset=1`); 4 steps is a quick-iteration
    // setting that still exercises every component (CLIP encode * 2,
    // UNet * 8 [4 steps * CFG], scheduler * 4, VAE decode * 1).
    //
    // ferrotorch-side companion files:
    //   * `scripts/pin_pretrained_sd_pipeline.py`
    //   * `scripts/verify_sd_pipeline_inference.py`
    //   * `ferrotorch-diffusion/examples/sd_pipeline_dump.rs`
    //   * `ferrotorch-diffusion/src/scheduler.rs` (new DDIMScheduler)
    //   * `ferrotorch-diffusion/src/pipeline.rs` (new StableDiffusionPipeline)
    //   * `ferrotorch-diffusion/tests/conformance_sd_pipeline.rs`
    //
    // `weights_url`/`weights_sha256` point at the tar bundle so this
    // registry entry has the same shape as the rest of the parity
    // bundles; the verify harness itself pulls individual files via
    // `hf_hub_download` and does not consume the tar. The
    // FerrotorchStateDict format tag indicates "not a HF safetensors
    // checkpoint" — the bundle is a single-file convenience archive
    // and not consumed by the safetensors loader.
    ModelInfo {
        name: "sd-v1-5-generation-trajectory",
        description: "Phase F SD-1.5 end-to-end text-to-image generation trajectory fixtures: prompt=\"a photograph of an astronaut riding a horse\", seed=42, num_inference_steps=4, guidance_scale=7.5, DDIMScheduler(scaled_linear, scaled_linear, eps prediction, leading spacing, steps_offset=1). Captures CLIP cond/uncond embeddings, initial Gaussian noise, every per-step (noise_pred_uncond, noise_pred_cond, guided_noise, latent_after_step) record, and the final VAE-decoded image. CreativeML Open RAIL-M; real-artifact baseline for SD-1.5 text-to-image generation parity vs diffusers (#1163).",
        weights_url: "https://huggingface.co/ferrotorch/sd-v1-5-generation-trajectory/resolve/main/bundle.tar",
        weights_sha256: "5fa7bd809e3aaa120a79c744801de44342a2e22ab82137cd5fe0d43302924c6e",
        format: WeightsFormat::FerrotorchStateDict,
        num_parameters: 0,
    },
    // #1167: distributions-parity-v1 — Phase G.1 torch.distributions
    // parity fixtures. The mirror is a fixture *bundle* (`bundle.tar`)
    // plus per-config subfolders. There are 17 distribution configs
    // (normal_{standard,shifted}, beta_25, gamma_21, cauchy_standard,
    //  exponential_1p5, uniform_neg2_3, lognormal_0_p5, laplace_0_1,
    //  halfnormal_1, studentt_df5, bernoulli_p3, poisson_3,
    //  categorical_k4, dirichlet_k4, mvn_3d, multinomial_k3_n20)
    // and 8 KL pairs (kl_{normal,bernoulli,uniform,categorical,
    //  laplace,exponential,gamma,poisson}_*). Each distribution
    // subfolder ships:
    //   * params.json           — params + family + metadata
    //   * test_points.bin       — fixed test points where log_prob is evaluated
    //   * sample.bin            — [N, *event_shape] torch reference samples
    //                             (provenance only — moment-based compare)
    //   * log_prob.bin          — [M] reference log_prob at test_points
    //   * entropy.bin           — [1 or B] reference entropy
    //                             (absent when torch lacks entropy(),
    //                             e.g. Poisson, Multinomial)
    //   * ref_moments.json      — sample mean + variance of the torch sample
    // KL subfolders ship `params.json` + `kl.bin`.
    //
    // The reference fixtures are produced by `torch.distributions.*`
    // under `torch.manual_seed(42)` with N=10000 samples per config,
    // so the harness can verify ferrotorch-distributions's sample
    // moments, log_prob, entropy, and KL divergence against torch
    // without re-running torch at verification time
    // (`scripts/verify_distributions_inference.py` +
    //  `ferrotorch-distributions/examples/distributions_dump.rs` +
    //  `ferrotorch-distributions/tests/conformance_torch_parity.rs`).
    //
    // The harness compares **moments** (mean, var) for sample data —
    // not byte-level sample sequences — because ferrotorch_core's
    // `creation::rand`/`randn` use a time-seeded xorshift PRNG, not
    // torch's seeded Philox. Per-metric tolerances are:
    //   * sample mean : max_abs <= 0.05    (MC noise budget at N=10000)
    //   * sample var  : max_abs <= 0.10    (variance estimator noise)
    //   * log_prob    : max_abs <= 1e-4
    //   * entropy     : max_abs <= 1e-4
    //   * KL          : max_abs <= 1e-4
    // For three distributions (cauchy_standard, laplace_0_1,
    // multinomial_k3_n20) the per-axis sample variance has an MC
    // noise floor above the global 0.10 tolerance — these skip the
    // moment comparison but keep all analytical metrics under the
    // 1e-4 floor.
    //
    // `weights_url`/`weights_sha256` point at the tar bundle so this
    // registry entry has the same shape as the rest of the parity
    // bundles; the verify harness itself pulls individual files via
    // `hf_hub_download` and does not consume the tar. The
    // FerrotorchStateDict format tag indicates "not a HF safetensors
    // checkpoint" — the bundle is a single-file convenience archive.
    ModelInfo {
        name: "distributions-parity-v1",
        description: "Phase G.1 torch.distributions parity fixtures: 17 canonical distribution configs (Normal, Beta, Gamma, Cauchy, Dirichlet, Exponential, Bernoulli, Categorical, Uniform, LogNormal, Multinomial, Poisson, StudentT, MultivariateNormal, HalfNormal, Laplace) plus 8 KL pairs (Normal-Normal, Bernoulli-Bernoulli, Uniform-Uniform, Categorical-Categorical, Laplace-Laplace, Exponential-Exponential, Gamma-Gamma, Poisson-Poisson). Reference samples (N=10000 with torch.manual_seed(42)), log_prob at fixed test points, entropy, and KL divergence from torch.distributions. Apache 2.0; real-artifact baseline for ferrotorch-distributions parity vs torch.distributions (#1167).",
        weights_url: "https://huggingface.co/ferrotorch/distributions-parity-v1/resolve/main/bundle.tar",
        weights_sha256: "bae19e48e3a4c6b5040557298fe037d4e4fccc832237b7a3522bc76ba6bf9f9e",
        format: WeightsFormat::FerrotorchStateDict,
        num_parameters: 0,
    },
    // #1168: tokenizer-parity-v1 — Phase G.2 HF tokenizer parity
    // fixtures for `ferrotorch-tokenize`. The mirror is a fixture
    // *bundle* (`bundle.tar`) plus per-family subfolders. Five
    // canonical tokenizer families are pinned (Llama 3 Instruct,
    // CLIP, BERT, GPT-2, SmolLM Instruct) covering BPE + WordPiece
    // model types, ASCII / unicode / whitespace / code / template
    // control-token strings, with chat-template renders on the two
    // Instruct families.
    //
    // Each `<family>/` subfolder ships:
    //   * tokenizer.json (+ tokenizer_config.json, vocab.*, merges.txt
    //     when upstream provides them) — exactly the same files the
    //     rust side reads via `ferrotorch_tokenize::load_tokenizer`.
    //   * strings.json    — the 20-element fixed test corpus.
    //   * token_ids.json  — Python `tokenizers.Tokenizer` reference
    //                       encodings (encode_with_special +
    //                       encode_no_special, lists of u32 per
    //                       input). The reference is intentionally
    //                       `tokenizers.Tokenizer` (not
    //                       `transformers.AutoTokenizer`) because that
    //                       is exactly the library the rust crate
    //                       wraps; `transformers` adds
    //                       `clean_up_tokenization_spaces` and
    //                       slow-tokenizer-specific decoder layers
    //                       that the rust crate does not implement.
    //   * decoded.json    — Python reference decodes
    //                       (decode_with_special_keep,
    //                        decode_with_special_skip,
    //                        decode_no_special).
    //   * chat_template.json — for families with a chat template
    //                          (Llama 3 Instruct + SmolLM Instruct):
    //                          rendered system+user+assistant
    //                          conversation with and without
    //                          `add_generation_prompt`. Reference
    //                          renders come from
    //                          `transformers.AutoTokenizer.apply_chat_template`
    //                          (a transformers feature, reproduced by
    //                          rust via minijinja).
    //   * meta.json       — provenance: upstream repo, tokenizers /
    //                       transformers versions, vocab size, chat
    //                       template presence.
    //
    // The harness compares with **exact** equality on every list and
    // string (no float tolerance — tokenization is integer-domain).
    // Per-family verdict + first divergent string surface immediately
    // if a regression lands. `weights_url`/`weights_sha256` pin the
    // tar bundle so the registry has one SHA to track; the verify
    // harness itself pulls per-family files via `hf_hub_download`. The
    // FerrotorchStateDict format tag indicates "not a HF safetensors
    // checkpoint" — the bundle is a single-file convenience archive.
    ModelInfo {
        name: "tokenizer-parity-v1",
        description: "Phase G.2 HF tokenizer parity fixtures: 5 canonical tokenizer families (Llama 3 BPE Instruct, CLIP BPE, BERT WordPiece, GPT-2 BPE, SmolLM BPE Instruct). 20 fixed test strings per family + a fixed system+user+assistant chat conversation rendered with and without add_generation_prompt. Encode/decode references come from tokenizers.Tokenizer (the exact library the ferrotorch-tokenize rust crate wraps); chat-template renders come from transformers.AutoTokenizer (the rust side reproduces apply_chat_template via minijinja). Mixed upstream licenses; real-artifact baseline for ferrotorch-tokenize parity vs HF (#1168).",
        weights_url: "https://huggingface.co/ferrotorch/tokenizer-parity-v1/resolve/main/bundle.tar",
        weights_sha256: "8d949235bb5cfaaea8916dcce001d17fd4b4383c2d5e033272397cf9545d1ef6",
        format: WeightsFormat::FerrotorchStateDict,
        num_parameters: 0,
    },
    // #1169: serialize-parity-v1 — Phase G.3 ferrotorch-serialize
    // format-parity fixtures. The mirror is a fixture *bundle*
    // (`bundle.tar`) plus four per-target subfolders:
    //
    //   * resnet18-pth/
    //       resnet18-f37072fd.pth     — official torchvision ZIP-pickle
    //                                    checkpoint (modern format —
    //                                    NOT the legacy `5c106cde.pth`
    //                                    tar-pickle, which neither
    //                                    `torch.load(..., weights_only=True)`
    //                                    nor ferrotorch's
    //                                    `zip::ZipArchive` reader can
    //                                    parse).
    //       reference_state_dict/
    //         <key>.bin               — per-tensor f32 binaries
    //                                    `[u32 ndim][u32 shape...][f32]`
    //                                    produced by `torch.load`.
    //       keys.json                 — ordered tensor name list.
    //   * safetensors-rt/
    //       resnet18.safetensors      — same state_dict re-saved via
    //                                    `safetensors.torch.save_file`.
    //       reference_state_dict/<key>.bin  — same per-tensor f32 bins.
    //       keys.json
    //   * gguf/
    //       SmolLM2-135M-Instruct-Q8_0.gguf — upstream
    //                                    `unsloth/SmolLM2-135M-Instruct-GGUF`
    //                                    Q8_0 + F32 mix (the K-quant
    //                                    Q4_K_M variant uses GGML type
    //                                    14 which ferrotorch-serialize
    //                                    does not parse today).
    //       reference_dequant/<name>.bin    — `gguf.quants.dequantize`
    //                                          outputs for a stride-
    //                                          sampled subset of 12
    //                                          tensors covering Q8_0
    //                                          + F32.
    //       sampled_tensor_names.json
    //       meta.json
    //   * onnx-mlp/
    //       mlp_weights.bin           — fixed-seed
    //                                    (`torch.manual_seed(42)`)
    //                                    weights for a
    //                                    `Linear(4->8) + ReLU + Linear(8->2)`
    //                                    MLP. Layout (in order):
    //                                    fc1.weight [8,4], fc1.bias [8],
    //                                    fc2.weight [2,8], fc2.bias [2],
    //                                    each preceded by
    //                                    `[u32 ndim][u32 shape...]`.
    //       input_{zeros,ones,random}.bin
    //       torch_forward_{zeros,ones,random}.bin
    //       meta.json
    //
    // The rust harness
    // (`ferrotorch-serialize/examples/serialize_parity_dump.rs`) runs
    // one target per invocation and dumps either per-tensor f32
    // binaries (pth / safetensors / gguf) or the rust-emitted
    // `mlp.onnx` plus three rust-side ferrotorch forward outputs
    // (onnx). The python verifier
    // (`scripts/verify_serialize_inference.py`) compares per the
    // hard per-target tolerance:
    //
    //   * pth_load               : max_abs == 0       (byte-exact)
    //   * safetensors_round_trip : max_abs == 0       (byte-exact)
    //   * gguf_load              : max_abs <= 1e-4    (Q8_0 dequant
    //                                                  noise floor)
    //   * onnx_export            : max_abs <= 1e-5 AND
    //                              cosine_sim >= 0.9999 between
    //                              (rust-emitted ONNX run via
    //                               onnxruntime) and (ferrotorch's
    //                               own forward).
    //
    // The cargo-side gate
    // (`ferrotorch-serialize/tests/conformance_format_parity.rs`)
    // shells out to the python verifier so a `cargo test --workspace`
    // run cannot accidentally skip parity. All four wrappers are
    // `#[ignore]`-gated because the verifier downloads from HF and
    // invokes `onnxruntime`.
    //
    // Real bugs surfaced during #1169 implementation (all fixed
    // before this entry landed):
    //   * pytorch_import: `BINUNICODE` (4-byte length) opcode constant
    //     was `0x8d` instead of `0x58`, breaking every modern ZIP-pickle
    //     `.pth` immediately at the first `BINUNICODE` string.
    //   * pytorch_import: `BINBYTES` / `SHORT_BINBYTES` constants were
    //     mutually swapped (0x42/0x44 instead of 0x42/0x43).
    //   * pytorch_import: `REDUCE` on `collections.OrderedDict()` /
    //     `builtins.dict()` produced a `Reduce` value, which the
    //     subsequent `SETITEMS` couldn't fill — now collapsed to an
    //     empty `Dict([])`.
    //   * pytorch_import: `try_extract_tensor_info` only recognized
    //     `_rebuild_tensor_v2`, silently dropping every `nn.Parameter`
    //     (wrapped in `_rebuild_parameter(tensor, requires_grad, ...)`).
    //     Recursion through `args[0]` now picks them up; resnet18
    //     went from 40 -> 102 tensors loaded.
    //   * onnx_export: `TENSOR_RAW_DATA` field number was `13` (which
    //     is `external_data`, a `repeated message`, NOT raw bytes) —
    //     every initializer was malformed and ONNX Runtime rejected
    //     the whole file with "Error parsing message with type
    //     'onnx.TensorProto'". Corrected to `9`.
    //
    // `weights_url`/`weights_sha256` point at the tar bundle so this
    // registry entry has the same shape as the rest of the parity
    // bundles; the verify harness itself pulls per-target files via
    // `hf_hub_download`. The `FerrotorchStateDict` format tag
    // indicates "not a HF safetensors checkpoint" — the bundle is a
    // single-file convenience archive.
    ModelInfo {
        name: "serialize-parity-v1",
        description: "Phase G.3 ferrotorch-serialize format-parity fixtures: real torchvision resnet18 ZIP-pickle .pth (resnet18-f37072fd) + the same state_dict re-saved as SafeTensors + real unsloth SmolLM2-135M-Instruct Q8_0 GGUF + a fixed-seed Linear(4->8) + ReLU + Linear(8->2) MLP for ONNX export. Tolerances are byte-exact for pth + safetensors, max_abs<=1e-4 for GGUF Q8_0 dequant, max_abs<=1e-5 + cosine_sim>=0.9999 for the ONNX round-trip (rust-emitted ONNX run through onnxruntime vs ferrotorch's own forward). Mixed upstream licenses (BSD-3 for resnet18, Apache-2.0 for SmolLM2); real-artifact baseline for ferrotorch-serialize parity vs torch / safetensors / gguf / onnxruntime (#1169).",
        weights_url: "https://huggingface.co/ferrotorch/serialize-parity-v1/resolve/main/bundle.tar",
        weights_sha256: "7c20267db5706421e7367c4d275346114a43ff6d55e6ff1aa11069bc45562296",
        format: WeightsFormat::FerrotorchStateDict,
        num_parameters: 0,
    },
];

/// List all available pretrained models.
pub fn list_models() -> Vec<&'static ModelInfo> {
    MODELS.iter().collect()
}

/// Get info for a specific model by name.
pub fn get_model_info(name: &str) -> Option<&'static ModelInfo> {
    MODELS.iter().find(|m| m.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_models_non_empty() {
        let models = list_models();
        assert!(
            !models.is_empty(),
            "registry should contain at least one model"
        );
    }

    #[test]
    fn test_get_model_info_resnet50() {
        let info = get_model_info("resnet50");
        assert!(info.is_some(), "resnet50 should be in the registry");
        let info = info.unwrap();
        assert_eq!(info.name, "resnet50");
        assert_eq!(info.num_parameters, 25_557_032);
        assert_eq!(info.format, WeightsFormat::SafeTensors);
    }

    #[test]
    fn test_get_model_info_resnet18() {
        let info = get_model_info("resnet18");
        assert!(info.is_some());
        assert_eq!(info.unwrap().num_parameters, 11_689_512);
    }

    #[test]
    fn test_get_model_info_vgg16() {
        let info = get_model_info("vgg16");
        assert!(info.is_some());
        assert_eq!(info.unwrap().num_parameters, 138_357_544);
    }

    #[test]
    fn test_get_model_info_vit_b_16() {
        let info = get_model_info("vit_b_16");
        assert!(info.is_some());
        assert_eq!(info.unwrap().num_parameters, 86_567_656);
    }

    #[test]
    fn test_get_model_info_nonexistent() {
        assert!(get_model_info("nonexistent_model").is_none());
    }

    #[test]
    fn test_get_model_info_empty_string() {
        assert!(get_model_info("").is_none());
    }

    #[test]
    fn test_registry_includes_all_vision_architectures() {
        // CL-385: every architecture exposed by ferrotorch_vision::models
        // should have an entry here so the vision registry's
        // pretrained=true path can resolve a download URL.
        let expected = [
            "resnet18",
            "resnet34",
            "resnet50",
            "vgg11",
            "vgg16",
            "vit_b_16",
            "efficientnet_b0",
            "swin_tiny",
            "convnext_tiny",
            "unet",
            "yolo",
            "mobilenet_v2",
            "mobilenet_v3_small",
            "densenet121",
            "inception_v3",
            "fasterrcnn_resnet50_fpn",
            "maskrcnn_resnet50_fpn",
            "deeplabv3_resnet50",
            "fcn_resnet50",
            "retinanet_resnet50_fpn",
            "fcos_resnet50_fpn",
            "lraspp_mobilenet_v3_large",
        ];
        for name in expected {
            assert!(
                get_model_info(name).is_some(),
                "ferrotorch_hub registry is missing entry for vision arch '{name}'"
            );
        }
    }

    #[test]
    fn test_all_models_have_valid_fields() {
        for model in list_models() {
            assert!(!model.name.is_empty(), "model name must not be empty");
            assert!(
                !model.description.is_empty(),
                "model description must not be empty"
            );
            assert!(
                !model.weights_url.is_empty(),
                "model weights_url must not be empty"
            );
            assert_eq!(
                model.weights_sha256.len(),
                64,
                "SHA-256 hex digest must be 64 characters for model '{}'",
                model.name
            );
            assert!(
                model.num_parameters > 0,
                "model '{}' must have >0 parameters",
                model.name
            );
        }
    }
}
