"""
统一配置加载 — 消除 Phase 1 / Phase 2 重复代码
"""
import yaml
from pathlib import Path


def load_config(config_path: str) -> dict:
    """加载并校验 YAML 配置文件"""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    for key in ("data", "model", "train"):
        if key not in config:
            raise KeyError(f"Config missing required section: '{key}'")
    return config
