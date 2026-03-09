import os
import sys

_SYSTEM_GLDISPATCH = "/usr/lib/x86_64-linux-gnu/libGLdispatch.so.0.0.0"
if os.path.exists(_SYSTEM_GLDISPATCH) and _SYSTEM_GLDISPATCH not in os.environ.get("LD_PRELOAD", ""):
    os.environ["LD_PRELOAD"] = _SYSTEM_GLDISPATCH + " " + os.environ.get("LD_PRELOAD", "")
    os.execvpe(sys.executable, [sys.executable, "-u", "-m", "ascent.run"] + sys.argv[1:], os.environ)

import hydra  # noqa
from habitat import get_config  # noqa
from habitat.config import read_write
from habitat.config.default import patch_config
from habitat.config.default_structured_configs import register_hydra_plugin
from habitat_baselines.run import execute_exp
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
from omegaconf import DictConfig
from omegaconf import OmegaConf
import sys
sys.path.insert(0, "third_party/frontier_exploration")
sys.path.insert(0, "third_party/depth_camera_filtering")
sys.path.insert(0, "third_party/vlfm")
import vlfm.measurements.traveled_stairs  # noqa: F401
import vlfm.obs_transformers.resize  # noqa: F401
import vlfm.policy.action_replay_policy  # noqa: F401
import vlfm.policy.habitat_policies  # noqa: F401
from habitat_baselines.config.default_structured_configs import (
    HabitatBaselinesConfigPlugin,
)

from ascent import ascent_policy
from ascent import ascent_trainer 

class HabitatConfigPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(provider="ascent", path="config/")


# 只注册一次，在这里
register_hydra_plugin(HabitatConfigPlugin)

@hydra.main(
    version_base=None,
    config_path="../experiments",  
    config_name="eval_ascent_hm3d.yaml",  # 修改：文件名不带 .yaml 后缀
)
def main(cfg: DictConfig) -> None:
    cfg = patch_config(cfg)
    execute_exp(cfg, "eval" if cfg.habitat_baselines.evaluate else "train")


if __name__ == "__main__":
    register_hydra_plugin(HabitatBaselinesConfigPlugin)  # 如果需要的话保留
    main()