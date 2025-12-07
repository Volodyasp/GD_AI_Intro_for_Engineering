import os
from envyaml import EnvYAML

_current_dir = os.path.abspath(os.path.dirname(__file__))
_config_name = "config.yaml"

CONFIG = EnvYAML(os.path.join(_current_dir, _config_name))
