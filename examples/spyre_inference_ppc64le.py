#!/usr/bin/env python3
import argparse
import asyncio
import atexit
import gc
import glob
import io
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time

import numpy
from openai import APIConnectionError, OpenAI

# Bump if you make changes to the log file.
VERSION = 1


# Log the interesting blips to logbuf, and tee to stdout
def log(string):
    global logbuf
    print(string)
    print(string, file=logbuf)


def cleanup_pserver():
    pserver.terminate()
    pserver.wait()


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--debug', action='store_true')
parser.add_argument('-l', '--online', action='store_true')
parser.add_argument('-r', '--remote', action='store_true')
parser.add_argument('-m',
                    '--model',
                    default="/models/granite-3.0-8b-instruct-r241014a/")
parser.add_argument('-t', '--maxtoken', type=int, default=64)
parser.add_argument('-n', '--promptlen', type=int, default=256)
parser.add_argument(
    '-i',
    '--prompts',
    default=
    "/home/senuser/aiu-fms-testing-utils/tests/resources/prompts/granite/input"
)
parser.add_argument('-p', '--port', type=int, default=8000)
parser.add_argument('-o', '--output', default="./")
parser.add_argument('-b', '--batch_size', type=int, default=1)
parser.add_argument('-k', '--keepoutput', action='store_true')
parser.add_argument('-c', '--cards', type=int, default=0)
parser.add_argument('--mintoken', type=int, default=0)
args = parser.parse_args()
debug = args.debug
online = args.online
fms_input_dir = args.prompts
tmpdir_output_dir = args.output
batch_size = args.batch_size
max_tokens = args.maxtoken
promptlen = args.promptlen
logbuf = io.StringIO()
keepoutput = args.keepoutput
num_cards = args.cards
min_tokens = args.mintoken

# For offline inferencing.
os.environ["VLLM_SPYRE_WARMUP_NEW_TOKENS"] = str(max_tokens)
os.environ["VLLM_SPYRE_WARMUP_BATCH_SIZES"] = str(batch_size)
os.environ["VLLM_SPYRE_WARMUP_PROMPT_LENS"] = str(promptlen)

# Verify setup is consistent with number of aius
if not args.remote:
    if "VLLM_AIU_PCIE_IDS" not in os.environ or os.environ[
            "VLLM_AIU_PCIE_IDS"] == "":
        print("source vllm_start.sh before running scripti."
              "VLLM_AIU_PCIE_IDS invalid.")
        exit(1)

    ids = os.environ["VLLM_AIU_PCIE_IDS"].split()
    if len(ids) != int(os.environ["AIU_WORLD_SIZE"]):
        print("VLLM_AIU_PCIE_IDS length does not match AIU_WORLD_SIZE."
              "Rerun vllm_start.sh?")
        exit(1)
else:
    ids = []

# If the user wants to use a specific number of cards, verify it.
# Otherwise use all available.
if num_cards > 0 and num_cards > len(ids):
    print(f"number of cards requested greater than number available: "
          f"{args.cards} > {len(ids)}")
    exit(1)
else:
    num_cards = len(ids)

tstart = time.perf_counter()

# Verify input directory has input
if len(glob.glob(os.path.join(fms_input_dir, "*.txt"))) <= 0:
    print(f"prompt directory does not contain any prompt files (.txt): "
          f"{fms_input_dir}")
    exit(1)

log(f"version, {VERSION}")

# Setup vllm client/service
if not debug:
    # Modify OpenAI's API key and API base to use vLLM's API server.
    # For now, assume it's running locally without access tokens.
    if online:
        openai_api_key = "EMPTY"
        openai_api_base = f"http://localhost:{args.port}/v1"
        client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

        # kick off server in a subprocess to avoid cross-contamination.
        # Or just connect.
        stime = time.perf_counter()
        if not args.remote:
            pserver = subprocess.Popen(
                [
                    "python3", "-m", "vllm.entrypoints.openai.api_server",
                    "--model", args.model, "-tp",
                    str(num_cards)
                ],
                stdout=sys.stdout,
            )
            atexit.register(cleanup_pserver)

        # wait on server to connect, wait for it to start
        model = None
        while not model:
            try:
                models = client.models.list()
                model = models.data[0].id
            except APIConnectionError:
                time.sleep(1)
        stime = time.perf_counter() - stime
        log(f"{stime:.2f}s, connected to server")

    else:
        import vllm
        llm = vllm.LLM(model=args.model, tensor_parallel_size=num_cards)
        llm_sparms = vllm.SamplingParams(max_tokens=max_tokens,
                                         temperature=0.0,
                                         min_tokens=min_tokens)

total_gentok = []
total_time = []
ttfirsttok = []
lock = threading.Lock()


async def runone(files, file_i):
    global total_gentok
    global total_time
    global lock
    global ttfirsttok
    global tstart
    file = files[file_i]
    # read out the next batch_size files for prompting.
    req_len = len(files[file_i:file_i + batch_size])
    prompt = [
        pathlib.Path(files[p]).read_text()
        for p in range(file_i, file_i + req_len)
    ]
    comp_tok = 0
    test_output = [""] * req_len
    tt = 0.0
    end_ft = 0.0

    if not debug and online:
        end_ft = -1.0
        start = time.perf_counter()
        try:
            chat_completion = client.completions.create(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                #echo=True, streaming mode breaks with this option.
                stream=True,
                temperature=0.0,
                extra_body=dict(min_tokens=min_tokens))
            # Not sure if always true.
            # But each iteration is one token of one request.
            for c in chat_completion:
                if end_ft < 0:
                    end_ft = time.perf_counter() - start
                comp_tok += 1
                test_output[c.choices[0].index] += c.choices[0].text
        except Exception:
            tnow = time.perf_counter() - tstart
            log(f"{tnow:.2f}s, error, prompt generation returned an exception. "
                f"Prompt too long?")

        tt = time.perf_counter() - start
        # Check for failed prompts
        for i in range(len(test_output)):
            if len(test_output[i]) == 0:
                tnow = time.perf_counter() - tstart
                log(f"{tnow:.2f}s, error, prompt: "
                    f"{os.path.basename(files[file_i+i])}"
                    f" did not generate any tokens. prompt likely too long")
        test_output = [prompt[i] + test_output[i] for i in range(req_len)]
    elif not debug and not online:
        start = time.perf_counter()
        output = llm.generate(prompts=prompt,
                              sampling_params=llm_sparms,
                              use_tqdm=False)
        tt = time.perf_counter() - start

        test_output = [
            prompt[i] + output[i].outputs[0].text for i in range(len(output))
        ]

        # From the innards of vllm, hopefully accurate
        end_ft = output[0].metrics.first_token_time - output[
            0].metrics.first_scheduled_time

        for o in output:
            comp_tok += len(o.outputs[0].token_ids)
    else:
        comp_tok += 1
        test_output = ["test"] * req_len

    if comp_tok != 0:
        avg_toktime = tt / comp_tok
    else:
        avg_toktime = 0.0
        end_ft = 0.0

    # Collect statistics and print
    lock.acquire()
    ttfirsttok.append(end_ft)
    total_time.append(tt)
    total_gentok.append(comp_tok)
    tnow = time.perf_counter() - tstart
    test_name = os.path.basename(file)
    if req_len > 1:
        test_name += "-" + os.path.basename(files[file_i + req_len - 1])
    log(f"{tnow:.2f}s, test, {test_name}, request time, {tt:.2f}s,"
        f" average tokentime, {avg_toktime:.2f}s, time-to-first,"
        f" {end_ft:.2f}s, total_tok, {comp_tok}")
    lock.release()

    # Name the files based on the input.
    # Not the test which contained them within a batch.
    if keepoutput:
        for x in range(req_len):
            prompt_file_name = os.path.basename(files[file_i + x])
            pathlib.Path(os.path.join(tmpdir_output_dir,
                                      prompt_file_name)).write_text(
                                          test_output[x])


async def runall():
    # TODO: how to best simulate "users" with online mode?
    # Multiple concurrent requests?
    #
    # Run all as separate requests, simultaneously
    #async with asyncio.TaskGroup() as tg:
    #    for file in glob.glob(os.path.join(fms_input_dir,"*.txt")):
    #        tg.create_task(runone(file))
    global tstart
    files = [x for x in glob.glob(os.path.join(fms_input_dir, "*.txt"))]
    for i in range(0, len(files), batch_size):
        await runone(files, i)


asyncio.run(runall())
tnow = time.perf_counter() - tstart
if not args.remote:
    log(f"aius={ids}")
    mode = "online" if online else "offline"
    log(f"{tnow:.2f}s, {mode} {batch_size}/{promptlen}/{max_tokens}/{num_cards}"
        )
    logname_pfx = (f"log_{mode}_{batch_size}-"
                   f"{promptlen}-{max_tokens}_"
                   f"{num_cards}aiu_")
else:
    log(f"{tnow:.2f}s, online (remote) {batch_size}/{promptlen}/{max_tokens}")
    logname_pfx = f"log_online_{batch_size}-{promptlen}-{max_tokens}_"

ttfirsttok = numpy.array(ttfirsttok)
total_time = numpy.array(total_time)
total_gentok = numpy.array(total_gentok)

total_inftime = numpy.sum(total_time)
log(f"{tnow:.2f}s, done, total inference time, {total_inftime:.2f}s")

# Average tok latency/job
toklatency = numpy.divide(total_time,
                          total_gentok,
                          out=numpy.zeros_like(total_time),
                          where=total_gentok != 0)

# Cumulative stats
avg_toklatency = numpy.average(toklatency)
std_toklatency = numpy.std(toklatency)
avg_ttft = numpy.average(ttfirsttok)
std_ttft = numpy.std(ttfirsttok)

log(f"average token latency, {avg_toklatency:.2f}s, std token latency,"
    f" {std_toklatency:.2f}s, average ttft, {avg_ttft:.2f}s, std ttft,"
    f" {std_ttft:.2f}s")

nextid = 0
logname = logname_pfx + str(nextid) + ".txt"
while pathlib.Path(logname).exists():
    nextid += 1
    logname = logname_pfx + str(nextid) + ".txt"

pathlib.Path(logname).write_text(logbuf.getvalue())

# TODO: multi-aiu doesn't exit. See aiu-app-sw-tracker#246
# From examples, child processes will get sigterm and make an ugly stack dump
if not debug and not online:
    del llm
    # TODO: send ourselves sigkill since vllm can't seem to clean itself up
    os.kill(os.getpid(), signal.SIGKILL)
    gc.collect()
