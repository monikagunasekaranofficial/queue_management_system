OBJECT_CLASSES = {
    "people": (0,),
    "vehicles": (2, 3, 5, 7),
}


def classes_for_mode(object_mode):
    """Return COCO class IDs for supported queue-monitoring modes."""
    try:
        return OBJECT_CLASSES[object_mode]
    except KeyError as error:
        raise ValueError(
            "object_mode must be either 'people' or 'vehicles'"
        ) from error
