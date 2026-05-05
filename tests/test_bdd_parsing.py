from ane_drive_perc.data.bdd import parse_bdd_metadata


def test_parse_project_bdd_metadata() -> None:
    metadata = {
        "id": "99267ac5-d1d37e06",
        "split": "train",
        "image": "100k/images/train/99267ac5-d1d37e06.jpg",
        "label": "100k/labels/train/99267ac5-d1d37e06.json",
        "width": 1280,
        "height": 720,
        "weather": "clear",
        "scene": "highway",
        "timeofday": "daytime",
        "boxes": [
            {
                "category": "car",
                "category_id": 2,
                "source_category": "car",
                "xyxy": [100.0, 200.0, 300.0, 400.0],
                "occluded": False,
                "truncated": False,
            }
        ],
    }

    parsed = parse_bdd_metadata(metadata, fallback_image_id="fallback")

    assert parsed.image_id == "99267ac5-d1d37e06"
    assert parsed.width == 1280
    assert parsed.height == 720
    assert parsed.objects[0].label == 2
    assert parsed.objects[0].xyxy == (100.0, 200.0, 300.0, 400.0)
    assert parsed.attributes["weather"] == "clear"


def test_parse_raw_bdd_person_to_pedestrian() -> None:
    metadata = {
        "name": "abc.jpg",
        "attributes": {"weather": "clear", "scene": "city", "timeofday": "daytime"},
        "frames": [
            {
                "objects": [
                    {
                        "category": "person",
                        "box2d": {"x1": 1, "y1": 2, "x2": 3, "y2": 4},
                        "attributes": {"occluded": False, "truncated": True},
                    }
                ]
            }
        ],
    }

    parsed = parse_bdd_metadata(metadata, fallback_image_id="fallback")

    assert parsed.image_id == "abc"
    assert parsed.objects[0].category == "pedestrian"
    assert parsed.objects[0].label == 0
    assert parsed.objects[0].truncated is True
