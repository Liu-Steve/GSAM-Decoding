from dataclasses import dataclass, field

from model.samd.samd_config import SamdConfig as BaseSamdConfig


@dataclass
class SamdConfig(BaseSamdConfig):
    use_gsam: bool = field(default=True)
    use_small_dict: bool = field(default=True)
    map_type: str = field(default="")
    lazy_threshold: int = field(default=1)
