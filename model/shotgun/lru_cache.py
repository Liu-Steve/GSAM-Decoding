import numpy as np
from collections import OrderedDict
from typing import Optional

type Tokens = tuple[int, ...]

class TwoLevelLRUCache:
    """
    Two-level LRU cache.

    - Top level: up to `prefix_capacity` distinct keys (each a tuple[int, ...]).
      Most-recently-used (MRU) key is on the right; least-recently-used (LRU) key on the left.
    - Second level: for every key, up to `followup_capacity` values
      (also tuples[int, ...]), kept in their own per-key LRU list.

    All public ops below are O(1):
    - put(key, value):       insert / update and mark MRU
    - get(key):              return all values for key and mark key MRU
    - get_value(key, value): check specific value, mark both levels MRU
    - __contains__(key):     membership test
    - __len__():             number of keys currently held
    """

    def __init__(self, prefix_capacity: int, followup_capacity: int) -> None:
        if prefix_capacity <= 0 or followup_capacity <= 0:
            raise ValueError("Capacities must be positive integers")
        self._prefix_capacity = prefix_capacity
        self._followup_capacity = followup_capacity

        # OrderedDict[key, OrderedDict[value, None]]
        self._cache: OrderedDict[Tokens, OrderedDict[Tokens, None]] = OrderedDict()

    def _touch_key(self, key: Tokens) -> None:
        """Mark `key` as most-recently used."""
        self._cache.move_to_end(key, last=True)

    def _touch_value(self, key: Tokens, value: Tokens) -> None:
        """Mark `value` (under `key`) as most-recently used."""
        self._cache[key].move_to_end(value, last=True)

    def put(self, key: Tokens, value: Tokens) -> None:
        """
        Insert `value` for `key` (or refresh its recency if already present).

        Handles both key- and value-level eviction when capacity limits are exceeded.
        """
        if key in self._cache:
            self._touch_key(key)
            vcache = self._cache[key]
            if value in vcache:
                self._touch_value(key, value)
            else:
                if len(vcache) >= self._followup_capacity:          # evict LRU value
                    vcache.popitem(last=False)
                vcache[value] = None
                self._touch_value(key, value)
        else:
            if len(self._cache) >= self._prefix_capacity:        # evict LRU key (and its values)
                self._cache.popitem(last=False)
            self._cache[key] = OrderedDict({value: None})

    def get(self, key: Tokens) -> Optional[list[Tokens]]:
        """
        Return all values associated with `key` (MRU to LRU order) and
        mark `key` as most-recently used.  Returns None on a miss.
        """
        if key not in self._cache:
            return None
        self._touch_key(key)
        return list(self._cache[key].keys())

    def get_value(self, key: Tokens, value: Tokens) -> Optional[Tokens]:
        """
        Access a specific (key, value) pair.

        Touches both key and value on a hit; returns the `value` or None on a miss.
        """
        if key not in self._cache:
            return None
        vcache = self._cache[key]
        if value not in vcache:
            self._touch_key(key)                   # still update key recency
            return None
        self._touch_key(key)
        self._touch_value(key, value)
        return value

    def __contains__(self, key: Tokens) -> bool:
        return key in self._cache

    def __len__(self) -> int:
        """Number of keys currently stored."""
        return len(self._cache)


class ShotgunCacheConfig:
    def __init__(
        self,
        prefix_capacity: int,
        followup_capacity: int,
        key_token_len: int,
        value_token_len: int,
    ) -> None:
        self._prefix_capacity = prefix_capacity
        self._followup_capacity = followup_capacity
        self._key_token_len = key_token_len
        self._value_token_len = value_token_len


class ShotgunCache:
    def __init__(self, configs: list[ShotgunCacheConfig]) -> None:
        if not configs:
            raise ValueError("At least one cache config is required")

        self._caches = [
            TwoLevelLRUCache(
                config._prefix_capacity,
                config._followup_capacity,
            )
            for config in configs
        ]

        self._key_lens = [config._key_token_len for config in configs]
        self._value_lens = [config._value_token_len for config in configs]
        self.max_key_len = max(self._key_lens)
        self.max_value_len = max(self._value_lens)
        self.max_key_value_len = self.max_key_len + self.max_value_len

    def get_draft_tokens(self, key: np.ndarray) -> list[Tokens]:
        """
        Retrieves draft tokens from all caches for the given `key`.
        
        Args:
        - key: The input token IDs to use as a key for cache lookup. If a cache table requires
          a key of length `k`, then the last `k` tokens of `key` will be used as the key.
            
        Returns:
            A list of draft tokens if any cache has a match, or an empty list if no matches found.
        """
        drafts_list = [
            drafts
            for cache, key_len in zip(self._caches, self._key_lens)
            if (drafts := cache.get(tuple(key[-key_len:]))) is not None
        ]

        return [
            draft
            for drafts in drafts_list
            for draft in drafts
        ]

    def update_cache(self, token_ids: np.ndarray) -> None:
        """
        Updates the cache with the given token IDs.

        Args:
        - token_ids: The token IDs to update the cache with.
        """
        for cache, key_len, value_len in zip(self._caches, self._key_lens, self._value_lens):
            total_len = key_len + value_len
            if len(token_ids) < total_len:
                continue
            offset_limit = len(token_ids) - total_len + 1
            for offset in range(offset_limit):
                key = token_ids[offset:offset+key_len]
                value = token_ids[offset+key_len:offset+total_len]
                cache.put(tuple(key), tuple(value))
