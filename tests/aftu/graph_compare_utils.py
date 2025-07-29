import difflib
import os
import re
import shutil
import tempfile
from collections.abc import Iterator
from glob import iglob
from os import path
from subprocess import PIPE, STDOUT, CalledProcessError, TimeoutExpired, run
from typing import Optional

from vllm.model_executor.model_loader.weight_utils import (
    download_weights_from_hf)


def load_graph_to_compare(file_path):
    with open(file_path) as file:
        content = file.read()

    # Replace id: <number> with id: ###
    content = re.sub(r'id: \d+', 'id: ###', content)

    # Replace ptr: <pointer> with ptr: xxxx
    content = re.sub(r'ptr: 0x[0-9a-fA-F]{12}', 'ptr: xxxx', content)

    # Replace value
    content = re.sub(r'values: ([0-9a-fA-F]{2}\s*)+', 'values: $$', content)

    # Regex to find all 's#' patterns surrounded by spaces,
    # or starting with a space and ending with a comma.
    # Examples: ' s1 ', ' s1,', ' s1 s2 '
    matched_symbols = re.findall(r'\s*(s\d+)[\s|,]', content)

    symbols_set = set([m for m in matched_symbols])

    # reindex symbols, considering the sorted indices

    sorted_symbols = sorted(list(symbols_set))
    symbol_map = {i: s for i, s in enumerate(sorted_symbols)}

    for i, s in symbol_map.items():
        content = content.replace(s, f'S#{i}')

    return content


def collect_graph_files(input_dir: str) -> dict[str, tuple[str, str]]:
    # Get G1 graphs.
    # Assumes the 'input_dir' contains 'export_dtcompiler' with the files.

    filepaths = iglob(path.join(input_dir, "export_dtcompiler", "*/*.ops"))

    # Filter out G2 files
    filepaths = [f for f in filepaths if not f.endswith(".g2.ops")]

    # NOTE: f.split("dump")[-1], split the filename by using dump,
    # to get numeric part which is the last one
    filemap = { f.split("dump")[-1]: (f, load_graph_to_compare(f)) \
        for f in filepaths}

    return filemap


def diff_graph(a_filepath, a_file, b_filepath, b_file) -> Iterator[str]:
    return difflib.unified_diff(a_file.split("\n"),
                                b_file.split("\n"),
                                fromfile=a_filepath,
                                tofile=b_filepath)


def get_aftu_script_dir() -> str:
    # TODO: since AFTU is not a lib yet, this function does the best
    # effort to get the scripts dir with inference.py to run the tests
    # for graph comparison.
    try:

        # Let's look for it... assuming it is installed as local,
        # i.e. git clone ... && uv pip install -e . [--no-deps]
        import aiu_fms_testing_utils
        module_dir = path.dirname(aiu_fms_testing_utils.__file__)
        repo_dir = path.dirname(module_dir)

        # Make sure it is the repo dir name
        if path.basename(repo_dir) != "aiu-fms-testing-utils":
            return None

        script_dir = os.path.join(repo_dir, "scripts")
        assert os.path.exists(script_dir)

        return script_dir
    except ImportError:
        return None


def compare_graphs(a_map: dict[str, tuple[str, str]],
                   b_map: dict[str, tuple[str, str]]) -> bool:

    are_graphs_similar = True
    for k, a_graph in a_map.items():
        a_filename, a_filedata = a_graph
        b_filename, b_filedata = b_map[k]

        diff = diff_graph(a_filename, a_filedata, b_filename, b_filedata)
        diff = list(diff)
        if diff:
            print("Found difference!", a_filename, b_filename)
            lines_count = len(diff)
            for line in diff[:20]:
                print(line)
            if (lines_count > 20):
                print(f"[...] Omitted {lines_count - 20} lines")
            are_graphs_similar = False

    return are_graphs_similar


def run_inference_py_and_get_graphs(
        inference_py_args: list[str],
        script_dir: str,
        extra_env: Optional[dict[str,
                                 str]] = None) -> dict[str, tuple[str, str]]:
    with tempfile.TemporaryDirectory() as tmpdir:

        env = os.environ.copy()
        env.update({
            "DEE_DUMP_GRAPHS": "aftu",
            "TORCH_SENDNN_CACHE_ENABLE": "0"
        })
        if extra_env:
            env.update(extra_env)
        # Copy scripts
        shutil.copytree(script_dir, os.path.join(tmpdir, "scripts"))

        try:
            run(inference_py_args,
                stdout=PIPE,
                stderr=STDOUT,
                text=True,
                check=True,
                env=env,
                cwd=tmpdir,
                timeout=600)
        except TimeoutExpired as e:
            print("`inference.py` process timeout!")
            if e.stdout:
                print(e.stdout)
            raise e

        except CalledProcessError as e:
            print(f"`inference.py` Process finished with code {e.returncode}")
            if e.stdout:
                print(e.stdout)
            raise e

        aftu_graphs = collect_graph_files(tmpdir)

    return aftu_graphs


def get_model_path(model_name_or_path):
    is_local = os.path.isdir(model_name_or_path)
    model_path = model_name_or_path
    # Get location of model from HF cache.
    if not is_local:
        model_path = download_weights_from_hf(
            model_name_or_path=model_path,
            cache_dir=None,
            allow_patterns=["*.safetensors", "*.bin", "*.pt"])

    return model_path
