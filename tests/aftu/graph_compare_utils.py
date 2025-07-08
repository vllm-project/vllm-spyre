import difflib
import os
import re
import shutil
import tempfile
from collections.abc import Iterator
from glob import iglob
from os import path
from subprocess import PIPE, Popen
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

    return content


def collect_graph_files(input_dir) -> dict[str, tuple[str, str]]:
    filepaths = iglob(path.join(input_dir, "export_dtcompiler", "*/*.ops"))

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
    # TODO: while AFTU is not a lib yet, this function does the best
    # effort to get the scripts dir with inference.py to run the tests
    # for graph comparison

    script_dir = os.environ.get("VLLM_SPYRE_TEST_AFTU_SCRIPTS_DIR", "")

    if script_dir:
        return script_dir

    # Let's look for it... assuming it is installed as local install
    import aiu_fms_testing_utils
    module_dir = path.dirname(aiu_fms_testing_utils.__file__)
    repo_dir = path.dirname(module_dir)

    # Make sure it is the repo dir name
    assert path.basename(repo_dir) == "aiu-fms-testing-utils"

    return os.path.join(repo_dir, "scripts")


def compare_graphs(a_map: dict[str, str], b_map: dict[str, str]) -> bool:
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


def get_aftu_graphs(
        inference_py_args: list[str],
        extra_env: Optional[dict[str, str]] = None) -> dict[str, str]:
    with tempfile.TemporaryDirectory() as tmpdir:

        env = os.environ.copy()
        env.update({
            "DEE_DUMP_GRAPHS": "aftu",
            "TORCH_SENDNN_CACHE_ENABLE": "0"
        })
        if extra_env:
            env.update(extra_env)
        # Copy scripts
        script_dir = get_aftu_script_dir()
        shutil.copytree(script_dir, os.path.join(tmpdir, "scripts"))

        process = Popen(inference_py_args,
                        stdout=PIPE,
                        stderr=PIPE,
                        env=env,
                        cwd=tmpdir)

        process.communicate()

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
