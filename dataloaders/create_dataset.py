"""
兼容层：从 ssl_dataset / phase2_dataset 聚合导出，便于统一引用。
新代码请直接 from dataloaders.ssl_dataset / dataloaders.phase2_dataset 导入。
"""
from dataloaders.phase2_dataset import (
    create_phase2_dataloaders,
    get_num_phase2_classes,
)
from dataloaders.ssl_dataset import create_ssl_dataloaders

__all__ = [
    "create_ssl_dataloaders",
    "create_phase2_dataloaders",
    "get_num_phase2_classes",
]
