from collections import OrderedDict
from typing import Optional

type Tokens = tuple[int, ...]

class TwoLevelLRUCache:
    """
    Two-level LRU cache.

    - Top level: up to `key_capacity` distinct keys (each a tuple[int, ...]).
      Most-recently-used (MRU) key is on the right; least-recently-used (LRU) key on the left.
    - Second level: for every key, up to `value_capacity` values
      (also tuples[int, ...]), kept in their own per-key LRU list.

    All public ops below are O(1):
    - put(key, value):       insert / update and mark MRU
    - get(key):              return all values for key and mark key MRU
    - get_value(key, value): check specific value, mark both levels MRU
    - __contains__(key):     membership test
    - __len__():             number of keys currently held
    """

    def __init__(self, key_capacity: int, value_capacity: int) -> None:
        if key_capacity <= 0 or value_capacity <= 0:
            raise ValueError("Capacities must be positive integers")
        self._key_capacity = key_capacity
        self._value_capacity = value_capacity

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
                if len(vcache) >= self._value_capacity:          # evict LRU value
                    vcache.popitem(last=False)
                vcache[value] = None
                self._touch_value(key, value)
        else:
            if len(self._cache) >= self._key_capacity:        # evict LRU key (and its values)
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
        key_capacity: int,
        value_capacity: int,
        key_token_len: int,
        value_token_len: int,
    ) -> None:
        self._key_capacity = key_capacity
        self._value_capacity = value_capacity
        self._key_token_len = key_token_len
        self._value_token_len = value_token_len


class ShotgunCache:
    def __init__(self, configs: list[ShotgunCacheConfig]) -> None:
        if not configs:
            raise ValueError("At least one cache config is required")

        self._caches = [
            TwoLevelLRUCache(
                config._key_capacity,
                config._value_capacity,
            )
            for config in configs
        ]
        self._key_token_lens = [config._key_token_len for config in configs]

        self.max_key_token_len = max(self._key_token_lens)

    def get_draft_tokens(self, key: Tokens) -> list[Tokens]:
        """
        Retrieves draft tokens from all caches for the given `key`.
        
        Args:
            key: The input token IDs to use as a key for cache lookup.
            
        Returns:
            A list of draft tokens if any cache has a match, or an empty list if no matches found.
        """
        drafts_list = [
            drafts
            for cache, key_token_len in zip(self._caches, self._key_token_lens)
            if (drafts := cache.get(tuple(key[-key_token_len:]))) is not None
        ]

        return [
            draft
            for drafts in drafts_list
            for draft in drafts
        ]
