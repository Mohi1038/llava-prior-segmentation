"""Command-line entry point for single-image FSeg-LLaVA inference."""

import argparse
import json
from pathlib import Path

from PIL import Image

from .config import PriorSegmentationConfig
from .models import HuggingFaceLlavaBackend, SamPredictorBackend
from .pipeline import merge_class_masks, run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", type=Path)
    parser.add_argument("--classes", nargs="+", help="Candidate foreground class names")
    parser.add_argument("--llava-model", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--sam-checkpoint", type=Path)
    parser.add_argument("--sam-model-type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--quantize",
        default="none",
        choices=["none", "int8"],
        help="Quantize LLaVA's language model (int8 needs optimum-quanto; use when RAM < ~24 GB)",
    )
    parser.add_argument("--version", action="version", version="llava-prior-segmentation 0.1.0")
    args = parser.parse_args()
    if args.image is None or not args.classes or args.sam_checkpoint is None:
        parser.error("image, --classes, and --sam-checkpoint are required")

    image = Image.open(args.image).convert("RGB")
    config = PriorSegmentationConfig(candidate_classes=tuple(args.classes))
    llava = HuggingFaceLlavaBackend(
        model_id=args.llava_model, device_name=args.device, quantize=args.quantize
    )
    sam = SamPredictorBackend(args.sam_checkpoint, args.sam_model_type, args.device)
    predictions = run_pipeline(image, args.classes, llava, sam, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = []
    for prediction in predictions:
        class_name = prediction.observation.class_name
        mask_path = args.output_dir / f"{class_name.replace(' ', '_')}.npy"
        import numpy as np

        np.save(mask_path, prediction.sam_mask.detach().cpu().numpy())
        metadata.append({"class": class_name, "response": prediction.observation.response, "mask": str(mask_path)})
    import numpy as np

    semantic_mask_path = args.output_dir / "semantic_mask.npy"
    np.save(semantic_mask_path, merge_class_masks(predictions).numpy())
    metadata_path = args.output_dir / "predictions.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()