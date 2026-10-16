from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key not in self.cache:
            return None
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            # Update existing key and move to end
            self.cache[key] = value
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                # Evict least recently used (first item)
                self.cache.popitem(last=False)
            self.cache[key] = value


# Concise assertions
if __name__ == "__main__":
    # Test eviction
    cache = LRUCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    cache.put(3, 3)  # Evicts key 1
    assert cache.get(1) is None, "Eviction failed: key 1 should be gone"
    assert cache.get(2) == 2, "Key 2 should still exist"

    # Test updating an existing key
    cache.put(2, 20)
    assert cache.get(2) == 20, "Update failed: key 2 should have value 20"

    # Test promotion after get
    cache2 = LRUCache(2)
    cache2.put(1, 1)
    cache2.put(2, 2)
    cache2.get(1)  # Promotes key 1 to most recently used
    cache2.put(3, 3)  # Should evict key 2 (least recently used)
    assert cache2.get(2) is None, "Promotion failed: key 2 should be evicted"
    assert cache2.get(1) == 1, "Key 1 should still exist after promotion"

    print("All assertions passed.")
