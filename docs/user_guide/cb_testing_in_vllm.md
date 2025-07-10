# Continuous Batching tests / inference scripts in vLLM

Brief overview of what has been implemented so far in VLLM to test / debug continuous batching

Date: 10th July 2025

## Inference script

* **File path:** `examples/offline_inference/cb_spyre_inference.py`
* **Purpose:** Debugging (ie. using manual execution)

### Description
* Runs inference on a set of prompts with continuous batching enabled (number of prompts is parametrizable)
* Prints the generated text for each sequence. 
* All the requested sequences are defined in the beginning, there is no requests joining the waiting queue while the decoding of some other request has already started.
* If `--compare-with-CPU` is set, then the output text is compared to the one of hugging face, running on CPU. Note that here the logprobs are not compared, only tokens. 

### Parametrization
* `--model`: the model
* `--max_model_len`: maximum length of a sequence (padded prompt plus decoded tokens) (cannot exceed model size)
* `--max_num_seqs`: Max number of sequences processed in a single iteration (decode batch size)
* `--tp`: Tensor parallelism (number of Spyre cards)
* `--num-prompts`: Total number of requested prompts
* `--max-tokens`: Number of tokens generated for each requested sequence
* `--compare-with-CPU`: If set, compare the text output with CPU version running with Hugging Face instead of vLLM

## CB tests through unit tests

* **File path (tests targetting CB specifically):** `vllm-spyre/tests/e2e/test_spyre_cb.py`
* **Purpose:** Automated execution to verify that a specific behaviour acts as expected (passing/failing)
* **Usage (when running locally):** `python -m pytest -sv -m "spyre and cb" --forked tests`
    * `-s` option: show all the print statements in the code
    * `-v` option: verbose mode, make the test output more detailed: show name of each test function and whether it passed, failed or was skipped
    * `--forked` option: isolates the tests and avoid having one test crashing impacting the other tests
    * `-m "spyre and cb"`: runs the tests with configurations marked as "spyre" and "cb" only

### Description

Unit tests are designed for automated and systematic execution to verify that CB behaves as expected for different scenarios. For each scenario (i.e. configuration of parameters), the test either passes or fails. When a test suite fails, identifying which specific test case failed is often more informative than the failure message itself. Below is a brief description of the different unit tests targetting CB. The description can also be found in the docstring of the different test functions:

> All the applicable unit tests in vLLM will eventually also execute with CB enabled in addition to SB, but two test functions specifically target continuous batching correctness: `test_cb_output` and `test_scheduler_cb_steps_tkv`. The other functions found in that files are mostly helper methods, or functions that test CB in aspects more specific to vLLM (such as scheduling constraints). Still it can be interesting to have a look in the code, but their description is skipped here.

#### `test_cb_output`
`test_cb_output` checks the correctness of the output of CB on a set of prompts (4 hardcoded prompts for that test). The output from vllm is compared to this of Hugging Face on CPU. 

* **The test passes if:** the logprobs of HF on CPU and vLLM (on Spyre or CPU depending on the backend) are compared, and the test passes only if the pairwise relative differences of the values are all below a threshold: `math.isclose(hf_logprob, vllm_logprob, rel_tol=0.35)`. Otherwise it fails.
> The above applies for sendnn backend, on CPU the tokens need to additionally be exactly the same for the test to pass

* **parametrization:**
    * `model`: the model
    * `backend`: if the test is running with `eager` backend for CPU or `sendnn` for Spyre
    * `max_num_seqs`: Max number of sequences processed in a single iteration (decode batch size)

* **Note on parametrization of prompts and max_tokens**: so far both the prompts and max tokens are hardcoded in that test. There are 4 prompts about chicken soup, and the `max_tokens` is set to 20 for all the prompts. It could be a good idea to add a `max-tokens` and `num-prompts` parametrization to create prompts in the same way as it is done in the inference script. Or we could even create prompts of specified length by parametrizing the number of tokens in the prompt, as it is done in the `test_scheduler_cb_steps_tkv` where artificial prompts are created by setting the parameter `prompts_lengths`. To be discussed.

#### `test_scheduler_cb_steps_tkv` (For this test the final output is not checked)

> **Note 1: since we are now testing more than only the tkv value at each step, I plan to rename that test, because the name is now a bit misleading**

> **Note 2: the final output is not checked because for a lot of parametrized scenarios they are only relevant for testing scheduling implementation, this saves some computation time.**

Checking the final output correctness alone is not enough to ensure that CB is correctly implemented (otherwise how can we differentiate with static batching for example). So `test_scheduler_cb_steps_tkv` is meant to check the correctness of the step-by-step execution of continuous batching. It does so by comparing, at every engine step (i.e. prefill or decode iteration), a bunch of attributes. This is allows a finer testing of the padding and scheduling implementation.

* **Checked attributes at each step:**
    * `tkv`: after each step, the tkv is compared against the expected tkv value for that step
    * `waiting`, `running`, `request_outputs`, `finished_requests` not really relevant in a compiler point of view, but after each iteration, we check that the list of running and waiting requests and those that have finished are correct. this tests the scheduler correctness.
    * (waiting to be merged, PR #261): `n_reserved_blocks` and `n_used_blocks`

* **Parametrization:**
    * `model`: the model
    * `backend`: if the test is running with `eager` backend for CPU or `sendnn` for Spyre
    * `seqs_max_tokens`: Number of tokens generated for each requested sequence prompt
    * `prompts_lengths`: Number of tokens for each prompt. Prompts are artificially generated given that parameter
    * `steps_add_reqs`: Steps where the requests prompts are joining, this helps simulating a online setup where server receives requests from users at different times
    * `checked_steps`: a list of reference values containing the reference values for the attributes described above at each step

* **Parametrization functions:** Because there are a lot of different scenarios and edge cases that we want to test through `test_scheduler_cb_steps_tkv`, the set of parameters described in the previous point are provided through different functions. Each function provide a different set of parameters and expected values at each step for the case that it is testing. This improves readability because the function names gives a brief idea of the tested scenario. For example:
    * `get_params_test_blocks_borders_aligned_prompts`: parametrization for the situation where the prompts are by chance already aligned with the blocks boundaries (no **right** padding required)
    * `get_params_test_blocks_borders_misaligned_prompts`: parametrization for the situation where the prompts are misaligned with the block boundaries, and thus **right** padding is required
    * ... additional special cases

