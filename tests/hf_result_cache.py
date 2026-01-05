import json
import os
import re
from pathlib import Path

from spyre_util import ModelInfo


class HFResultCache:
    """
    A simple cache for storing and retrieving results from Hugging Face models.
    The cache is stored in a JSON file named 'hf_cache.json' in the same
    directory as this script.

    This cache can be (re)populated by running all tests and committing the
    changes to the .json file.
    """

    NO_REVISION_KEY = "no-revision"

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
            with open(self.cached_results_file_path, "w") as f:
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
            json_string = json.dumps(self.cached_results, indent=4)
            json_string = self._remove_newlines_in_json_lists(json_string)
            with open(self.cached_results_file_path, "w") as f:
                f.write(json_string)
            self.dirty = False

    def get_cached_result(
        self, model: str | ModelInfo, prompt: str | list[int], max_tokens: int
    ) -> dict:
        """
        Retrieve a cached result for the given model, prompt, and max_tokens.
        Returns an empty dictionary if no cache entry is found.
        """
        if isinstance(prompt, list):
            prompt = self._token_ids_to_string(prompt)
        max_tokens = str(max_tokens)

        if isinstance(model, ModelInfo):
            revision = model.revision if model.revision else self.NO_REVISION_KEY
            model_name = model.name
        else:
            revision = self.NO_REVISION_KEY
            model_name = model

        return (
            self.cached_results.get(model_name, {})
            .get(revision, {})
            .get(prompt, {})
            .get(max_tokens, {})
        )

    def add_to_cache(
        self, model: str | ModelInfo, prompt: str | list[int], max_tokens: int, result: dict
    ):
        """
        Add a new result to the cache for the given model, prompt, and
        max_tokens. Marks the cache as 'dirty' to indicate that it needs to be
        written to disk.
        """
        if isinstance(prompt, list):
            prompt = self._token_ids_to_string(prompt)
        max_tokens = str(max_tokens)

        if isinstance(model, ModelInfo):
            revision = model.revision if model.revision else self.NO_REVISION_KEY
            model_name = model.name
        else:
            revision = self.NO_REVISION_KEY
            model_name = model

        self.cached_results.setdefault(model_name, {}).setdefault(revision, {}).setdefault(
            prompt, {}
        ).setdefault(max_tokens, result)
        self.dirty = True

    def _token_ids_to_string(self, token_ids: list[int]) -> str:
        """Use a string to represent a list of token ids, so that it can be
        hashed and used as a json key."""

        return "__tokens__" + "_".join(str(token_id) for token_id in token_ids)

    def _remove_newlines_in_json_lists(self, json_string):
        """
        Removes newline characters only within JSON list structures.
        """
        # Regex to find content inside square brackets (JSON lists)
        # It captures the content within the brackets, including newlines.
        # The 're.DOTALL' flag allows '.' to match newlines.
        pattern = r"\[(.*?)\]"

        def replace_newlines(match):
            # Get the captured content (the list items)
            list_content = match.group(1)
            # Strip leading indentation, leaving one space between elements
            cleaned_content = re.sub(r"\n\s+", "\n ", list_content)
            # Delete all newline characters
            cleaned_content = cleaned_content.replace("\n", "").replace("\r", "")
            # Return the content wrapped in square brackets again
            return f"[{cleaned_content}]"

        # Apply the regex and replacement function
        modified_json_string = re.sub(pattern, replace_newlines, json_string, flags=re.DOTALL)
        return modified_json_string
