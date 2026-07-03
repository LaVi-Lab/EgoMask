import json
import logging
import fasteners
import contextlib
from abc import abstractmethod
from dataclasses import dataclass
from sqlitedict import SqliteDict
from typing import (
    Dict,
    Callable,
    Generator,
    Mapping,
    Optional,
    Tuple,
    Iterable,
)


def request_to_key(request: Mapping) -> str:
    """Normalize a `request` into a `key` so that we can hash using it."""
    return json.dumps(request, sort_keys=True)

class KeyValueStore(contextlib.AbstractContextManager):
    """Key value store that persists writes."""

    @abstractmethod
    def contains(self, key: Dict) -> bool:
        pass

    @abstractmethod
    def get(self, key: Dict) -> Optional[Dict]:
        pass

    @abstractmethod
    def get_all(self) -> Generator[Tuple[Dict, Dict], None, None]:
        pass

    @abstractmethod
    def put(self, key: Mapping, value: Dict) -> None:
        pass

    @abstractmethod
    def multi_put(self, pairs: Iterable[Tuple[Dict, Dict]]) -> None:
        pass

    @abstractmethod
    def remove(self, key: Dict) -> None:
        pass


class SqliteKeyValueStore(KeyValueStore):
    """Key value store backed by a SQLite file."""

    def __init__(self, path: str, tablename: str = None):
        super().__init__()
        if tablename is not None:
            self._sqlite_dict = SqliteDict(path, tablename)
        else:
            self._sqlite_dict = SqliteDict(path)
        self.lock = fasteners.InterProcessLock(path)

    def __enter__(self) -> "SqliteKeyValueStore":
        self._sqlite_dict.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._sqlite_dict.__exit__(exc_type, exc_value, traceback)

    def contains(self, key: Dict) -> bool:
        return request_to_key(key) in self._sqlite_dict

    def get(self, key: Dict) -> Optional[Dict]:
        key_string = request_to_key(key)
        result = self._sqlite_dict.get(key_string)
        if result is not None:
            return result
        return None

    def get_all(self) -> Generator[Tuple[Dict, Dict], None, None]:
        for key, value in self._sqlite_dict.items():
            yield (key, value)

    def put(self, key: Mapping, value: Dict) -> None:
        key_string = request_to_key(key)
        with self.lock:
            self._sqlite_dict[key_string] = value
            self._sqlite_dict.commit()

    def multi_put(self, pairs: Iterable[Tuple[Dict, Dict]]) -> None:
        for key, value in pairs:
            self.put(key, value)

    def remove(self, key: Dict) -> None:
        del self._sqlite_dict[key]
        self._sqlite_dict.commit()



def write_to_key_value_store(
    key_value_store: KeyValueStore, key: Mapping, response: Dict
) -> bool:
    """
    Write to the key value store with retry. Returns boolean indicating whether the write was successful or not.
    """
    try:
        key_value_store.put(key, response)
        return True
    except Exception as e:
        print(f"[INFO] Error when writing to cache: {str(e)}")
        return False

class Cache(object):
    """
    Reference: https://github.com/RaRe-Technologies/sqlitedict
    """

    def __init__(self, cache_path: str):
        print(f"[INFO] Created cache with config: {cache_path}")
        self.config_path = cache_path

    def get(
        self, request: Mapping, compute: Callable[[], Dict], force: bool = False
    ) -> Tuple[Dict, bool]:

        with SqliteKeyValueStore(self.cache_path,"prepare") as key_value_store:
            response = key_value_store.get(request)
            if response:
                cached = True
            else:
                cached = False

            if force or not cached:
                # Compute and commit the request/response to SQLite
                response = compute()

                write_to_key_value_store(key_value_store, request, response)

        return response, cached
