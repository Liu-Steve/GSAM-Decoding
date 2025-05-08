import types
import numpy as np
from collections import OrderedDict
from typing import Optional
import pickle

type Tokens = tuple[int, ...]

class TwoLevelLRUCache:
    """
    Two-level LRU cache.

    - Top level: up to `prefix_capacity` distinct prefixes (each a tuple[int, ...]).
      Most-recently-used (MRU) prefix is on the right; least-recently-used (LRU) prefix on the left.
    - Second level: for every prefix, up to `followup_capacity` followups
      (also tuples[int, ...]), kept in their own per-prefix LRU list.

    All public ops below are O(1):
    - put(prefix, followup):       insert / update and mark MRU
    - get(prefix):              return all followups for prefix and mark prefix MRU
    - get_followup(prefix, followup): check specific followup, mark both levels MRU
    - __contains__(prefix):     membership test
    - __len__():             number of prefixes currently held
    """

    def __init__(
        self,
        prefix_capacity: int,
        followup_capacity: int,
        prefix_len: int,
        followup_len: int,
    ) -> None:
        if prefix_capacity <= 0 or followup_capacity <= 0:
            raise ValueError("Capacities must be positive integers")
        self._prefix_capacity = prefix_capacity
        self._followup_capacity = followup_capacity
        self._prefix_len = prefix_len
        self._followup_len = followup_len

        # OrderedDict[prefix, OrderedDict[followup, None]]
        self._cache: OrderedDict[Tokens, OrderedDict[Tokens, None]] = OrderedDict()

    def _touch_prefix(self, prefix: Tokens) -> None:
        """Mark `prefix` as most-recently used."""
        self._cache.move_to_end(prefix, last=True)

    def _touch_followup(self, prefix: Tokens, followup: Tokens) -> None:
        """Mark `followup` (under `prefix`) as most-recently used."""
        self._cache[prefix].move_to_end(followup, last=True)

    def put(self, prefix: Tokens, followup: Tokens) -> None:
        """
        Insert `followup` for `prefix` (or refresh its recency if already present).

        Handles both prefix- and followup-level eviction when capacity limits are exceeded.
        """

        if prefix in self._cache:
            self._touch_prefix(prefix)
            vcache = self._cache[prefix]
            if followup in vcache:
                self._touch_followup(prefix, followup)
            else:
                if len(vcache) >= self._followup_capacity:          # evict LRU followup
                    vcache.popitem(last=False)
                vcache[followup] = None
                self._touch_followup(prefix, followup)
        else:
            if len(self._cache) >= self._prefix_capacity:        # evict LRU prefix (and its followups)
                self._cache.popitem(last=False)
            self._cache[prefix] = OrderedDict({followup: None})

    def resize(self, prefix_capacity: int, followup_capacity: int) -> None:
        """
        Resize the cache to new capacity limits.
        
        If new capacities are smaller, least recently used entries will be discarded.
        
        Args:
        - `prefix_capacity`: New capacity for prefixes
        - `followup_capacity`: New capacity for followups per prefix
        
        Raises:
        - `ValueError`: If either capacity is not positive
        """

        if prefix_capacity <= 0 or followup_capacity <= 0:
            raise ValueError("Capacities must be positive integers")
        
        # Remove least recently used prefixes until we're within the new capacity
        if prefix_capacity < self._prefix_capacity:
            while len(self._cache) > prefix_capacity:
                self._cache.popitem(last=False)

        # Remove least recently used followups until we're within the new capacity
        if followup_capacity < self._followup_capacity:
            for followups in self._cache.values():
                while len(followups) > followup_capacity:
                    followups.popitem(last=False)

        # Update capacity values
        self._prefix_capacity = prefix_capacity
        self._followup_capacity = followup_capacity

    def get(self, prefix: Tokens) -> Optional[list[Tokens]]:
        """
        Return all followups associated with `prefix` (MRU to LRU order) and
        mark `prefix` as most-recently used.  Returns None on a miss.
        """
        if prefix not in self._cache:
            return None
        self._touch_prefix(prefix)
        return list(self._cache[prefix].keys())

    def get_followup(self, prefix: Tokens, followup: Tokens) -> Optional[Tokens]:
        """
        Access a specific (prefix, followup) pair.

        Touches both prefix and followup on a hit; returns the `followup` or None on a miss.
        """
        if prefix not in self._cache:
            return None
        vcache = self._cache[prefix]
        if followup not in vcache:
            self._touch_prefix(prefix)                   # still update prefix recency
            return None
        self._touch_prefix(prefix)
        self._touch_followup(prefix, followup)
        return followup

    def __contains__(self, prefix: Tokens) -> bool:
        return prefix in self._cache

    def __len__(self) -> int:
        """Number of prefixes currently stored."""
        return len(self._cache)
        
    @staticmethod
    def load_from_file(path: str, frozen: bool = False) -> 'TwoLevelLRUCache':
        """
        Load a `TwoLevelLRUCache` instance from a pickle file.
        
        Args:
        - `path`: Path to the pickle file containing a `TwoLevelLRUCache` instance.
        - `frozen`: If True, the cache will be frozen, meaning its recency tracking
                   will be disabled and items won't change position in the LRU order.
        
        Returns:
        - A `TwoLevelLRUCache` instance.
        
        Raises:
        - ValueError: If the pickle file does not contain a `TwoLevelLRUCache` instance.
        """
        with open(path, 'rb') as f:
            cache = pickle.load(f)
        
        if not isinstance(cache, TwoLevelLRUCache):
            raise ValueError(f"The pickle file does not contain a TwoLevelLRUCache instance. Found {type(cache).__name__} instead.")

        if frozen:
            # Define no-op methods
            def noop_touch_prefix(self, prefix):
                pass
            def noop_touch_followup(self, prefix, followup):
                pass
            def noop_put(self, prefix, followup):
                pass
            
            # Replace methods with no-ops
            cache._touch_prefix = types.MethodType(noop_touch_prefix, cache)
            cache._touch_followup = types.MethodType(noop_touch_followup, cache)
            cache.put = types.MethodType(noop_put, cache)
        
        return cache


class ShotgunCacheConfig:
    def __init__(
        self,
        prefix_capacity: int,
        followup_capacity: int,
        prefix_len: int,
        followup_len: int,
        file_path: Optional[str] = None,
        frozen: bool = False,
    ) -> None:
        self._prefix_capacity = prefix_capacity
        self._followup_capacity = followup_capacity
        self._prefix_len = prefix_len
        self._followup_len = followup_len
        self._file_path = file_path
        self._frozen = frozen


class ShotgunCache:
    def __init__(self, configs: list[ShotgunCacheConfig]) -> None:
        if not configs:
            raise ValueError("At least one cache config is required")

        self._caches = []
        for config in configs:
            if config._file_path:
                cache = TwoLevelLRUCache.load_from_file(config._file_path, config._frozen)
                if cache._prefix_len != config._prefix_len:
                    raise ValueError(
                        f"Cache prefix length mismatch. "
                        f"Expected {config._prefix_len}, "
                        f"got {cache._prefix_len} "
                        f"from file {config._file_path}"
                    )
                if cache._followup_len != config._followup_len:
                    raise ValueError(
                        f"Cache followup length mismatch. "
                        f"Expected {config._followup_len}, "
                        f"got {cache._followup_len} "
                        f"from file {config._file_path}"
                    )
                cache.resize(config._prefix_capacity, config._followup_capacity)
                self._caches.append(cache)
            else:
                if config._frozen:
                    raise ValueError("Fresh cache table must not be frozen")
                self._caches.append(
                    TwoLevelLRUCache(
                        config._prefix_capacity,
                        config._followup_capacity,
                        config._prefix_len,
                        config._followup_len,
                    )
                )

        self._prefix_lens = [config._prefix_len for config in configs]
        self._followup_lens = [config._followup_len for config in configs]
        self.max_prefix_len = max(self._prefix_lens)
        self.max_followup_len = max(self._followup_lens)
        self.max_prefix_followup_len = self.max_prefix_len + self.max_followup_len

    def get_draft_tokens(self, prefix: np.ndarray) -> list[Tokens]:
        """
        Retrieves draft tokens from all caches for the given `prefix`.
        
        Args:
        - prefix: The input token IDs to use as a prefix for cache lookup. If a cache table requires
          a prefix of length `k`, then the last `k` tokens of `prefix` will be used as the prefix.
            
        Returns:
            A list of draft tokens if any cache has a match, or an empty list if no matches found.
        """
        drafts_list = [
            drafts
            for cache, prefix_len in zip(self._caches, self._prefix_lens)
            if (drafts := cache.get(tuple(prefix[-prefix_len:]))) is not None
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
        for cache, prefix_len, followup_len in zip(self._caches, self._prefix_lens, self._followup_lens):
            total_len = prefix_len + followup_len
            if len(token_ids) < total_len:
                continue
            offset_limit = len(token_ids) - total_len + 1
            for offset in range(offset_limit):
                prefix = token_ids[offset:offset+prefix_len]
                followup = token_ids[offset+prefix_len:offset+total_len]
                cache.put(tuple(prefix), tuple(followup))
