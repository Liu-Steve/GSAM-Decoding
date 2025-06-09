import types
import numpy as np
from collections import OrderedDict
from typing import Optional, Any
import pickle

type Tokens = tuple[int, ...]

class TwoLevelLRUCache:
    """
    Two-level LRU cache.

    - Top level: up to `leader_capacity` distinct leaders (each a tuple[int, ...]).
      Most-recently-used (MRU) leader is on the right; least-recently-used (LRU) leader on the left.
    - Second level: for every leader, up to `follower_capacity` followers
      (also tuples[int, ...]), kept in their own per-leader LRU list.

    All public ops below are O(1):
    - put(leader, follower):       insert / update and mark MRU
    - get(leader):              return all followers for leader and mark leader MRU
    - get_follower(leader, follower): check specific follower, mark both levels MRU
    - __contains__(leader):     membership test
    - __len__():             number of leaders currently held
    """

    def __init__(
        self,
        leader_capacity: int,
        follower_capacity: int,
        leader_len: int,
        follower_len: int,
    ) -> None:
        if leader_capacity <= 0 or follower_capacity <= 0:
            raise ValueError("Capacities must be positive integers")
        self._leader_capacity = leader_capacity
        self._follower_capacity = follower_capacity
        self._leader_len = leader_len
        self._follower_len = follower_len

        # OrderedDict[leader, OrderedDict[follower, None]]
        self._cache: OrderedDict[Tokens, OrderedDict[Tokens, None]] = OrderedDict()

    def _touch_leader(self, leader: Tokens) -> None:
        """Mark `leader` as most-recently used."""
        self._cache.move_to_end(leader, last=True)

    def _touch_follower(self, leader: Tokens, follower: Tokens) -> None:
        """Mark `follower` (under `leader`) as most-recently used."""
        self._cache[leader].move_to_end(follower, last=True)

    def put(self, leader: Tokens, follower: Tokens) -> None:
        """
        Insert `follower` for `leader` (or refresh its recency if already present).

        Handles both leader- and follower-level eviction when capacity limits are exceeded.
        """

        if leader in self._cache:
            self._touch_leader(leader)
            vcache = self._cache[leader]
            if follower in vcache:
                self._touch_follower(leader, follower)
            else:
                if len(vcache) >= self._follower_capacity:          # evict LRU follower
                    vcache.popitem(last=False)
                vcache[follower] = None
                self._touch_follower(leader, follower)
        else:
            if len(self._cache) >= self._leader_capacity:        # evict LRU leader (and its followers)
                self._cache.popitem(last=False)
            self._cache[leader] = OrderedDict({follower: None})

    def clear(self) -> None:
        """
        Clear the cache.
        """

        self._cache.clear()

    def resize(self, leader_capacity: int, follower_capacity: int) -> None:
        """
        Resize the cache to new capacity limits.
        
        If new capacities are smaller, least recently used entries will be discarded.
        
        Args:
        - `leader_capacity`: New capacity for leaders
        - `follower_capacity`: New capacity for followers per leader
        
        Raises:
        - `ValueError`: If either capacity is not positive
        """

        if leader_capacity <= 0 or follower_capacity <= 0:
            raise ValueError("Capacities must be positive integers")
        
        # Remove least recently used leaders until we're within the new capacity
        if leader_capacity < self._leader_capacity:
            while len(self._cache) > leader_capacity:
                self._cache.popitem(last=False)

        # Remove least recently used followers until we're within the new capacity
        if follower_capacity < self._follower_capacity:
            for followers in self._cache.values():
                while len(followers) > follower_capacity:
                    followers.popitem(last=False)

        # Update capacity values
        self._leader_capacity = leader_capacity
        self._follower_capacity = follower_capacity

    def get(self, leader: Tokens) -> Optional[list[Tokens]]:
        """
        Return all followers associated with `leader` (MRU to LRU order) and
        mark `leader` as most-recently used.  Returns None on a miss.
        """
        if leader not in self._cache:
            return None
        self._touch_leader(leader)
        return list(self._cache[leader].keys())

    def get_follower(self, leader: Tokens, follower: Tokens) -> Optional[Tokens]:
        """
        Access a specific (leader, follower) pair.

        Touches both leader and follower on a hit; returns the `follower` or None on a miss.
        """
        if leader not in self._cache:
            return None
        vcache = self._cache[leader]
        if follower not in vcache:
            self._touch_leader(leader)                   # still update leader recency
            return None
        self._touch_leader(leader)
        self._touch_follower(leader, follower)
        return follower

    def __contains__(self, leader: Tokens) -> bool:
        return leader in self._cache

    def __len__(self) -> int:
        """Number of leaders currently stored."""
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
            def noop_touch_leader(self, leader):
                pass
            def noop_touch_follower(self, leader, follower):
                pass
            def noop_put(self, leader, follower):
                pass
            def noop_clear(self):   
                pass
            
            # Replace methods with no-ops
            cache._touch_leader = types.MethodType(noop_touch_leader, cache)
            cache._touch_follower = types.MethodType(noop_touch_follower, cache)
            cache.put = types.MethodType(noop_put, cache)
            cache.clear = types.MethodType(noop_clear, cache)
        
        return cache


class CachebackCacheConfig:
    def __init__(
        self,
        leader_capacity: int,
        follower_capacity: int,
        leader_len: int,
        follower_len: int,
        file_path: Optional[str] = None,
        frozen: bool = False,
    ) -> None:
        self._leader_capacity = leader_capacity
        self._follower_capacity = follower_capacity
        self._leader_len = leader_len
        self._follower_len = follower_len
        self._file_path = file_path
        self._frozen = frozen


class CachebackCache:
    def __init__(self, configs: list[CachebackCacheConfig]) -> None:
        if not configs:
            raise ValueError("At least one cache config is required")

        self._caches = []
        for config in configs:
            if config._file_path:
                cache = TwoLevelLRUCache.load_from_file(config._file_path, config._frozen)
                if cache._leader_len != config._leader_len:
                    raise ValueError(
                        f"Cache leader length mismatch. "
                        f"Expected {config._leader_len}, "
                        f"got {cache._leader_len} "
                        f"from file {config._file_path}"
                    )
                if cache._follower_len != config._follower_len:
                    raise ValueError(
                        f"Cache follower length mismatch. "
                        f"Expected {config._follower_len}, "
                        f"got {cache._follower_len} "
                        f"from file {config._file_path}"
                    )
                cache.resize(config._leader_capacity, config._follower_capacity)
                self._caches.append(cache)
            else:
                if config._frozen:
                    raise ValueError("Fresh cache table must not be frozen")
                self._caches.append(
                    TwoLevelLRUCache(
                        config._leader_capacity,
                        config._follower_capacity,
                        config._leader_len,
                        config._follower_len,
                    )
                )

        self._leader_lens = [config._leader_len for config in configs]
        self._follower_lens = [config._follower_len for config in configs]
        self.max_leader_len = max(self._leader_lens)
        self.max_follower_len = max(self._follower_lens)
        self.max_leader_follower_len = self.max_leader_len + self.max_follower_len

    def get_draft_tokens(
            self,
            leader: list[int]
        ) -> tuple[list[tuple[Tokens, list[Any]]], list[int]]:
        """
        Retrieves draft tokens from all caches for the given `leader`.
        
        Args:
        - leader: The input token IDs to use as a leader for cache lookup. If a cache table requires
          a leader of length `k`, then the last `k` tokens of `leader` will be used as the leader.
            
        Returns:
        - A list of draft tokens if any cache has a match, or an empty list if no matches found.
        - A list of the lengths of the draft tokens.
        """

        drafts_list = []
        drafts_lens = []
        for cache, leader_len, follower_len in zip(self._caches, self._leader_lens, self._follower_lens):
            drafts = cache.get(tuple(leader[-leader_len:]))
            if drafts is not None:
                drafts_list.extend([(draft, []) for draft in reversed(drafts)])
                drafts_lens.extend([follower_len for _ in range(len(drafts))])
        
        return drafts_list, drafts_lens

    def update_cache(self, token_ids: list[int]) -> None:
        """
        Updates the cache with the given token IDs.

        Args:
        - token_ids: The token IDs to update the cache with.
        """
        for cache, leader_len, follower_len in zip(self._caches, self._leader_lens, self._follower_lens):
            total_len = leader_len + follower_len
            if len(token_ids) < total_len:
                continue
            offset_limit = len(token_ids) - total_len + 1
            for offset in range(offset_limit):
                leader = token_ids[offset:offset+leader_len]
                follower = token_ids[offset+leader_len:offset+total_len]
                cache.put(tuple(leader), tuple(follower))

    def clear(self) -> None:
        """
        Clear the cache.
        """
        for cache in self._caches:
            cache.clear()
