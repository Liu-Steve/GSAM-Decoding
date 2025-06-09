import gc
from model.shotgun.lru_cache import ShotgunCache, ShotgunCacheConfig

shotgun_cache = ShotgunCache([
    ShotgunCacheConfig(
        prefix_capacity=2**20,
        followup_capacity=128,
        prefix_len=5,
        followup_len=5,
        file_path="openwebtext_sample100/openwebtext_prefix5_followup5_lru_cache.pkl"
    )
])

gc.collect()

print("Ready")

while True:
    pass
