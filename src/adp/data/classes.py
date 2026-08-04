from dataclasses import dataclass

from omegaconf import DictConfig


@dataclass(frozen=True)
class ClassMap:
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.names:
            raise ValueError("Class map cannot be empty.")

        if len(set(self.names)) != len(self.names):
            raise ValueError(f"Class names must be unique: {self.names}")

    @property
    def num_classes(self) -> int:
        return len(self.names)

    @property
    def id_to_name(self) -> dict[int, str]:
        return dict(enumerate(self.names))

    @property
    def name_to_id(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(self.names)}

    def id_for_name(self, name: str) -> int:
        try:
            return self.name_to_id[name]
        except KeyError as exc:
            raise KeyError(f"Unknown canonical class name: {name!r}") from exc

    def contains(self, name: str) -> bool:
        return name in self.name_to_id


def build_class_map(cfg: DictConfig) -> ClassMap:
    return ClassMap(tuple(str(name) for name in cfg.classes.names))


def build_source_category_map(cfg: DictConfig) -> dict[str, str]:
    """Return the BDD-native category to ADP-canonical category map."""
    return {str(k): str(v) for k, v in cfg.bdd.category_map.items()}
