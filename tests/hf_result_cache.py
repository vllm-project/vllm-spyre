import json
import os
from pathlib import Path


class HFResultCache:
    """
    A simple cache for storing and retrieving results from Hugging Face models.
    The cache is stored in a JSON file named 'hf_cache.json' in the same
    directory as this script.

    This cache can be (re)populated by running all tests and committing the
    changes to the .json file.
    """

    def __init__(self):
        """
        Initialize the HFResultCache. Load existing cached results from
        'hf_cache.json'. If the file does not exist, an empty cache dictionary
        is created.
        """
        current_dir = Path(os.path.abspath(os.path.dirname(__file__)))
        self.cached_results_file_path = current_dir / "hf_cache.json"

        if not self.cached_results_file_path.exists():
            self.cached_results = {}
            # Start with empty file
            with open(self.cached_results_file_path, 'w') as f:
                json.dump(self.cached_results, f)
        else:
            with open(self.cached_results_file_path) as f:
                self.cached_results = json.load(f)

        self.dirty = False

    def write_cache(self):
        """
        Write the current cache to 'hf_cache.json' if it has been modified.
        """
        if self.dirty:
            with open(self.cached_results_file_path, 'w') as f:
                json.dump(self.cached_results, f)
            self.dirty = False

    def get_cached_result(self, model: str, prompt: str | list[int],
                          max_tokens: int) -> dict:
        """
        Retrieve a cached result for the given model, prompt, and max_tokens.
        Returns an empty dictionary if no cache entry is found.
        """
        if isinstance(prompt, list):
            prompt = self._token_ids_to_string(prompt)
        max_tokens = str(max_tokens)

        return self.cached_results.get(model, {}).get(prompt,
                                                      {}).get(max_tokens, {})

    def add_to_cache(self, model: str, prompt: str | list[int],
                     max_tokens: int, result: dict):
        """
        Add a new result to the cache for the given model, prompt, and
        max_tokens. Marks the cache as 'dirty' to indicate that it needs to be
        written to disk.
        """
        if isinstance(prompt, list):
            prompt = self._token_ids_to_string(prompt)
        max_tokens = str(max_tokens)

        self.cached_results.setdefault(model,
                                       {}).setdefault(prompt, {}).setdefault(
                                           max_tokens, result)
        self.dirty = True

    def _token_ids_to_string(self, token_ids: list[int]) -> str:
        """Use a string to represent a list of token ids, so that it can be
        hashed and used as a json key."""

        return "__tokens__" + "_".join(str(token_id) for token_id in token_ids)
