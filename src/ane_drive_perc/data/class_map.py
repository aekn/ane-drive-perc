BDD100K_DET_CLASSES: tuple[str, ...] = (
    "pedestrian",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign",
)

BDD100K_CLASS_TO_IDX: dict[str, int] = {
    name: idx for idx, name in enumerate(BDD100K_DET_CLASSES)
}

BD100K_IDX_TO_CLASS: dict[int, str] = {
    idx: name for idx, name in enumerate(BDD100K_DET_CLASSES)
}


def get_bdd100k_detection_classes() -> tuple[str, ...]:
    return BDD100K_DET_CLASSES


def encode_bdd100k_label(label: str) -> int:
    try:
        return BDD100K_CLASS_TO_IDX[label]
    except KeyError as e:
        valid = ", ".join(BDD100K_DET_CLASSES)
        raise ValueError(
            f"unknown BDD100K label '{label}'. Valid labels: {valid}"
        ) from e


def decode_bdd100k_label(index: int) -> str:
    try:
        return BD100K_IDX_TO_CLASS[index]
    except KeyError as e:
        raise ValueError(f"unknown BDD100K class index {index}") from e
