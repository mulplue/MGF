from yacs.config import CfgNode
from typing import Callable


def Build_Metrics(cfg: CfgNode) -> Callable:
    if cfg.DATA.TASK == "TP":
        from metrics.TP_metrics import TP_metrics

        return TP_metrics(cfg)
