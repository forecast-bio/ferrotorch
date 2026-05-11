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
