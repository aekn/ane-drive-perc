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

BDD100K_IDX_TO_CLASS: dict[int, str] = {
    idx: name for idx, name in enumerate(BDD100K_DET_CLASSES)
}

BDD_SOURCE_TO_TARGET: dict[str, str] = {
    "person": "pedestrian",
    "pedestrian": "pedestrian",
    "rider": "rider",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "train": "train",
    "motor": "motorcycle",
    "motorcycle": "motorcycle",
    "bike": "bicycle",
    "bicycle": "bicycle",
    "traffic light": "traffic light",
    "traffic sign": "traffic sign",
}


def get_bdd100k_detection_classes() -> tuple[str, ...]:
    return BDD100K_DET_CLASSES


def encode_bdd100k_label(label: str) -> int:
    normalized = BDD_SOURCE_TO_TARGET.get(label, label)
    try:
        return BDD100K_CLASS_TO_IDX[normalized]
    except KeyError as exc:
        valid = ", ".join(BDD100K_DET_CLASSES)
        raise ValueError(
            f"Unknown BDD100K label {label!r}. Valid labels: {valid}"
        ) from exc


def decode_bdd100k_label(index: int) -> str:
    try:
        return BDD100K_IDX_TO_CLASS[index]
    except KeyError as exc:
        raise ValueError(f"Unknown BDD100K class index {index}") from exc
