"""
Generador de datos de muestra para desarrollo y testing.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import CLASS_NAMES, PATIENTS_CSV_PATH, RAW_DATA_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)
CLASS_SEED_OFFSETS = {"COVID19": 1000, "Normal": 2000, "Pneumonia": 3000}


def generate_synthetic_xray(
    class_name: str,
    img_size: tuple = (224, 224),
    seed: int | None = None,
) -> Image.Image:
    if seed is not None:
        np.random.seed(seed)

    base = np.random.normal(loc=80, scale=15, size=(*img_size, 1))
    base = np.clip(base, 0, 255).astype(np.uint8)
    img = Image.fromarray(base.squeeze(), mode="L")
    draw = ImageDraw.Draw(img)

    cx, cy = img_size[0] // 2, img_size[1] // 2
    draw.ellipse([cx - 70, cy - 80, cx + 70, cy + 60], fill=120, outline=100)

    if class_name == "Normal":
        draw.ellipse([cx - 50, cy - 50, cx - 10, cy + 30], fill=140)
        draw.ellipse([cx + 10, cy - 50, cx + 50, cy + 30], fill=140)
    elif class_name == "Pneumonia":
        draw.ellipse([cx - 50, cy - 50, cx - 10, cy + 30], fill=140)
        draw.ellipse([cx + 10, cy - 50, cx + 50, cy + 30], fill=100)
        for _ in range(5):
            ox = np.random.randint(cx + 5, cx + 45)
            oy = np.random.randint(cy - 40, cy + 20)
            radius = np.random.randint(5, 15)
            draw.ellipse([ox - radius, oy - radius, ox + radius, oy + radius], fill=70)
    elif class_name == "COVID19":
        draw.ellipse([cx - 50, cy - 50, cx - 10, cy + 30], fill=110)
        draw.ellipse([cx + 10, cy - 50, cx + 50, cy + 30], fill=110)
        for _ in range(8):
            side = np.random.choice([-1, 1])
            ox = cx + side * np.random.randint(15, 45)
            oy = np.random.randint(cy - 40, cy + 20)
            radius = np.random.randint(8, 18)
            opacity = np.random.randint(60, 90)
            draw.ellipse([ox - radius, oy - radius, ox + radius, oy + radius], fill=opacity)

    img = img.filter(ImageFilter.GaussianBlur(radius=2))
    return Image.merge("RGB", [img, img, img])


def generate_sample_dataset(
    output_dir: Path,
    num_per_class: int = 30,
    seed: int = 42,
) -> Path:
    logger.info("Generando dataset de muestra en %s", output_dir)
    patient_rows = []

    for class_name in CLASS_NAMES:
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for index in range(num_per_class):
            image = generate_synthetic_xray(
                class_name=class_name,
                seed=seed + CLASS_SEED_OFFSETS[class_name] + index,
            )
            filepath = class_dir / f"{class_name.lower()}_{index:04d}.png"
            image.save(filepath)

            patient_rows.append(
                {
                    "patient_id": f"{class_name.lower()}_{index:04d}",
                    "image_name": filepath.name,
                    "image_path": str(filepath),
                    "diagnosis": class_name,
                    "age": 30 + (index % 35),
                    "sex": "F" if index % 2 == 0 else "M",
                    "admission_date": f"2026-01-{(index % 28) + 1:02d}",
                    "source": "sample_generator",
                }
            )

        logger.info("%s: %s imagenes generadas", class_name, num_per_class)

    PATIENTS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(patient_rows).to_csv(PATIENTS_CSV_PATH, index=False)
    logger.info("Metadata de pacientes generada en %s", PATIENTS_CSV_PATH)
    logger.info("Dataset de muestra completo: %s imagenes", len(patient_rows))
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Generar dataset de muestra para testing del pipeline"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RAW_DATA_DIR),
        help="Directorio de salida para las imagenes",
    )
    parser.add_argument(
        "--num-per-class",
        type=int,
        default=30,
        help="Numero de imagenes por clase",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para reproducibilidad",
    )
    args = parser.parse_args()

    generate_sample_dataset(
        output_dir=Path(args.output_dir),
        num_per_class=args.num_per_class,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
