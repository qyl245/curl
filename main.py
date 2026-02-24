import argparse
import copy
import os
import random

import numpy as np
import torch

from utils.config import load_config
from utils.logging_utils import setup_logger

logger = setup_logger()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_encoder(config: dict, modality: str):
    from models.encoder import EMGEncoder, IMUEncoder

    mcfg = config["model"][modality]
    if modality == "emg":
        return EMGEncoder(embed_dim=mcfg["embed_dim"], nhead=mcfg["nhead"], num_layers=mcfg["num_layers"])
    return IMUEncoder(embed_dim=mcfg["embed_dim"], nhead=mcfg["nhead"], num_layers=mcfg["num_layers"])


def run_phase1(config: dict, modality: str):
    from data.create_dataset import create_ssl_dataloaders
    from models.ssl_model import SSLModel
    from training.ssl_trainer import SSLTrainer

    modalities = ["emg", "imu"] if modality == "both" else [modality]
    for mod in modalities:
        logger.info(f"===== Phase1 SSL ({mod.upper()}) =====")
        cfg_local = copy.deepcopy(config)
        if mod == "imu":
            cfg_local["train"]["phase1"]["num_epochs"] = cfg_local["train"]["phase1"].get(
                "imu_num_epochs",
                cfg_local["train"]["phase1"]["num_epochs"],
            )

        encoder = build_encoder(cfg_local, mod)
        ssl_model = SSLModel(encoder=encoder, config=cfg_local, modality=mod)
        train_loader, val_loader = create_ssl_dataloaders(cfg_local, mod)
        trainer = SSLTrainer(
            model=ssl_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=cfg_local,
            modality=mod,
        )
        trainer.train()


def _load_ckpt_if_exists(module: torch.nn.Module, path: str, name: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} checkpoint not found: {path}")
    module.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
    logger.info(f"Loaded {name} checkpoint: {path}")


def run_phase2(config: dict, auto_preprocess: bool):
    from data.create_dataset import create_phase2_dataloaders, get_num_phase2_classes
    from models.fusion import MultiModalModel
    from scripts.preprocess_jtom import preprocess_jtom_integrated
    from training.fusion_trainer import FusionTrainer

    phase2_pt = config["data"]["phase2"]["processed_pt_path"]
    if auto_preprocess and not os.path.exists(phase2_pt):
        logger.info("Phase2 样本不存在，自动运行 preprocess_jtom_integrated ...")
        preprocess_jtom_integrated(
            csv_path="dataset/j-tom03.csv",
            save_path=phase2_pt,
            target_length=200,
        )

    train_loader, val_loader, test_loader = create_phase2_dataloaders(config)
    num_classes = get_num_phase2_classes(config)

    emg_encoder = build_encoder(config, "emg")
    imu_encoder = build_encoder(config, "imu")
    _load_ckpt_if_exists(emg_encoder, config["paths"]["phase1_emg_ckpt"], "EMG encoder")
    _load_ckpt_if_exists(imu_encoder, config["paths"]["phase1_imu_ckpt"], "IMU encoder")

    fusion_model = MultiModalModel(
        emg_encoder=emg_encoder,
        imu_encoder=imu_encoder,
        model_cfg=config["model"],
        num_classes=num_classes,
    )
    trainer = FusionTrainer(
        model=fusion_model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )
    trainer.train()

    best_path = os.path.join(config["train"]["output_dir"], "models", "best_fusion_model.pt")
    if os.path.exists(best_path):
        fusion_model.load_state_dict(torch.load(best_path, map_location="cpu"), strict=True)
        fusion_model.to(trainer.device)
    test_metrics = trainer.evaluate(test_loader)
    trainer.dump_report(test_metrics, filename="phase2_test_eval.json")
    logger.info(f"Phase2 test metrics: {test_metrics}")


def parse_args():
    parser = argparse.ArgumentParser(description="Biceps coach POC pipeline")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)

    sub = parser.add_subparsers(dest="stage", required=True)
    p1 = sub.add_parser("phase1", help="Run Phase1 SSL training")
    p1.add_argument("--modality", choices=["emg", "imu", "both"], default="both")

    p2 = sub.add_parser("phase2", help="Run Phase2 fusion training")
    p2.add_argument("--auto_preprocess", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.device:
        config["train"]["device"] = args.device
    if args.seed is not None:
        config["train"]["seed"] = args.seed
    set_seed(config["train"].get("seed", 42))

    if args.stage == "phase1":
        run_phase1(config, args.modality)
    elif args.stage == "phase2":
        run_phase2(config, auto_preprocess=args.auto_preprocess)
    else:
        raise ValueError(f"Unknown stage: {args.stage}")


if __name__ == "__main__":
    main()
