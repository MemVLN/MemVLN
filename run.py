import os
from datetime import datetime

import hydra
from accelerate.logging import get_logger
from omegaconf import OmegaConf

from common.misc import rgetattr
from trainer.build import build_trainer

logger = get_logger(__name__)


@hydra.main(config_path='configs', config_name='default', version_base=None)
def main(cfg):
    naming_keys = [cfg.name]
    for name in cfg.naming_keywords:
        key = str(rgetattr(cfg, name))
        if key:
            naming_keys.append(key)
    exp_name = '_'.join(naming_keys)

    # Record the experiment
    cfg.exp_dir = os.path.join(
        cfg.base_dir, exp_name,
        f"{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}" if 'time' in cfg.naming_keywords else ""
    )
    os.makedirs(cfg.exp_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(cfg.exp_dir, 'config.yaml'), resolve=True)

    trainer = build_trainer(cfg)
    trainer.run()


if __name__ == '__main__':
    main()
