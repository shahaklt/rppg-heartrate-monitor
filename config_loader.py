"""Central config loader. Every stage calls load_config() to read configs/config.yaml.

Kept dependency-free except pyyaml so it imports cleanly in any environment.
"""
from __future__ import annotations

import os
from functools import lru_cache

import yaml

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "configs", "config.yaml")


class Config(dict):
    """dict with attribute access and nested resolution helpers."""

    def __getattr__(self, name):
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return Config(value) if isinstance(value, dict) else value

    def path(self, key: str) -> str:
        """Resolve a configured relative path against the project root."""
        return os.path.join(PROJECT_ROOT, self["paths"][key])


@lru_cache(maxsize=None)
def load_config(path: str = DEFAULT_CONFIG) -> Config:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw)


def project_path(*parts: str) -> str:
    return os.path.join(PROJECT_ROOT, *parts)
