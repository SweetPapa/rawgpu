"""Training recipes. Each module has META (what the Studio shows: a name, a description, the parameters
with defaults, the charts and tiles) and train(run); some also try_it(run, request) for the playground.

A parameter: {"key", "label", "type": int|float|str|choice|bool|path, "default", optional "min", "max",
"choices", "help", "advanced"}. A chart: {"title", "series": [{"key", "label"}], "y": "log"|None}. A
tile: {"key", "label", "fmt", "best": "min"|"max"|None}. "panels" name the event views the recipe
emits: samples, reliability, rollout, retrieval, probes, planning.
"""
import importlib

ORDER = ["gpt", "jepa", "probe", "critic", "eg2_contrastive"]

COMMON = [
    {"key": "max_temp_c", "label": "Pause at", "type": "int", "default": 85, "min": 60, "max": 90, "unit": "°C",
     "help": "Training pauses while the card is this hot, and resumes when it has cooled 8 °C.", "advanced": True},
    {"key": "seed", "label": "Random seed", "type": "int", "default": 0, "advanced": True},
]


def get(name):
    if name not in ORDER:
        raise KeyError(f"no recipe {name!r}")
    return importlib.import_module(f"recipes.{name}")


def catalog():
    out = []
    for name in ORDER:
        m = get(name)
        meta = dict(m.META)
        meta["id"] = name
        meta["params"] = list(meta["params"]) + COMMON
        meta["playground"] = hasattr(m, "try_it")
        out.append(meta)
    return out


def defaults(name):
    meta = next(m for m in catalog() if m["id"] == name)
    return {p["key"]: p["default"] for p in meta["params"]}
