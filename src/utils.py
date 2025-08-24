import omegaconf
from omegaconf import OmegaConf
import argparse


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/exp.yaml",
        help="Path to YAML config"
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Override as key=value (e.g., data.max_len=256 model.dropout=0.3)",
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    
    if args.overrides:
        dot = OmegaConf.from_dotlist(args.overrides)
        cfg = OmegaConf.merge(cfg, dot)
    
    return cfg
