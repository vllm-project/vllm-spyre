# Continuous Batching tests / inference scripts in vLLM

Brief overview of what has been implemented so far in VLLM to test / debug continuous batching

## Inference script

* **File paths:**
    * `examples/offline_inference/cb_spyre_inference.py`
    * `examples/offline_inference/long_context.py`
* **Purpose:** Debugging (i.e. using manual execution)

### Description

* Runs inference on a set of prompts with continuous batching enabled (number of prompts is parametrizable)
* Prints the generated text for each sequence.
* All the requested sequences are defined in the beginning, there is no requests joining the waiting queue while the decoding of some other request has already started.
* The exact sequence of prefill and decode steps depends on the parameter values `max_num_seqs`, `num-prompts`, `max-tokens`.
* If `--compare-with-CPU` is set, then the output text is compared to the one of hugging face, running on CPU. Note that here the logprobs are not compared, only tokens.

### Parametrization

For `cb_spyre_inference.py`

* `--model`: the model
* `--max_model_len`: maximum length of a sequence (padded prompt plus decoded tokens) (cannot exceed model size)
* `--max_num_seqs`: Max number of sequences processed in a single iteration (decode batch size)
* `--tp`: Tensor parallelism (number of Spyre cards)
* `--num-prompts`: Total number of requested prompts
* `--max-tokens`: Number of tokens generated for each requested sequence
* `--compare-with-CPU`: If set, compare the text output with CPU version running with Hugging Face instead of vLLM

For `long_context.py`: the same parameters, but with some differences:

* `--max_prompt_len`: max lengths of prompts (prompts will have length up to `max_prompt_len`)
* doesn't allow to specify `--max-tokens`: number of tokens set automatically given `max_model_len` and prompts lengths

## CB tests through unit tests

!!! abstract "In Short"
    See the detailed description of the individual unit tests for continuous batching in their respective files directly.

    * [Output Tests](tests/output_tests.md): Check the correctness of end output logits/tokens of sequences ran with continuous batching enabled
    * [Scheduler Steps Tests](tests/scheduler_steps_tests.md): Check the correctness of the step-by-step execution of continuous batching for different scenarios of prompt lengths and requested tokens
    * [Other Tests](tests/other_tests.md): Other tests verifying the various behaviours of vLLM, when running with continuous batching enabled

* **Purpose:** Automated execution to verify that a specific behaviour acts as expected (passing/failing)

* **Files paths:**
    * Output Tests: `vllm-spyre/tests/e2e/test_spyre_basic.py`
    * Scheduler Steps Tests: `vllm-spyre/tests/e2e/test_spyre_cb_scheduler_steps.py`
    * Other Tests: various files including `vllm-spyre/tests/e2e/test_spyre_cb.py`

<!-- markdownlint-disable MD031 MD046 -->

### Usage (when running locally)

#### Commands

    # Runs all the tests
    python -m pytest -svx -m "spyre and cb" --forked tests
    
    # Runs specific test file
    python -m pytest -svx -m "spyre and cb" --forked tests/e2e/test_spyre_cb_scheduler_steps.py
    
    # Runs specific test function
    python -m pytest -svx -m "spyre and cb" --forked tests/e2e/test_spyre_basic.py::test_output

<!-- markdownlint-enable MD031 MD046 -->

#### Parameters description

* `-x` option: stops the execution as soon as a test fails
* `-s` option: show all the print statements in the code
* `-v` option: verbose mode, make the test output more detailed: show name of each test function and whether it passed, failed or was skipped
* `--forked` option: isolates the tests and avoid having one test crashing impacting the other tests
* `-m "spyre and cb"`: runs the tests with configurations marked as "spyre" and "cb" only

!!! tip
    To run a test with a different model than the default `ibm-ai-platform/micro-g3.3-8b-instruct-1b`, you can run the test with `VLLM_SPYRE_TEST_MODEL_LIST` environment variable set to the target model, for example:
    ```bash
    VLLM_SPYRE_TEST_MODEL_LIST='tiny-granite-3.2-8b' python -m pytest -svx -m "spyre and cb" --forked tests/e2e/test_spyre_cb.py
    ```

<!-- markdownlint-disable MD024 no-duplicate-heading -->

### Description

<!-- markdownlint-enable MD024 -->

Unit tests are designed for automated and systematic execution to verify that CB behaves as expected for different scenarios. For each scenario (i.e. configuration of parameters), the test either passes or fails. When a test suite fails, identifying which specific test case failed is often more informative than the failure message itself. Below is a brief description of the different unit tests targeting CB. The description can also be found in the docstring of the different test functions:

!!! caution
    When adding new parametrization to a test, the parameters are typically combinatorial and number of executed tests can increase really fast. For example the following test function will run 2 x 2 = 4 different scenarios in total:
    ```python
    @pytest.mark.parametrize("model", ["micro-g3.3-8b-instruct-1b", "granite-3.3-8b-instruct"])
    @pytest.mark.parametrize("max_tokens", [[10, 20], [60, 78]])
    def test_function(model: str, max_tokens: list[int]):
       ...
    ```

#### Output Tests

See [Output Tests](tests/output_tests.md)

Output tests checks the correctness of the output of CB on a set of prompts. For now, the number of prompts and the prompts themself are hardcoded, as well as the max requested tokens per prompt (constant and set to 20). The output from vllm is compared to this of Hugging Face on CPU.

!!! note inline end
    This applies for sendnn backend, on CPU the tokens need to additionally be exactly the same for the test to pass

* The test passes if: the logprobs of HF on CPU and vLLM (on Spyre or CPU depending on the backend) are compared, and the test passes only if the pairwise relative differences of the values are all below a threshold: `math.isclose(hf_logprob, vllm_logprob, rel_tol=0.35)`. Otherwise it fails. There is no logic that takes into account the fact that the tokens might becomes different at some point, making the logits diverging.

#### Scheduler Steps Tests

See [Scheduler Steps Tests](tests/scheduler_steps_tests.md)

!!! Question
    For these tests, the final output is not checked, only the step-by-step execution correctness. Would it make sense to have output validation though?

Checking the final output correctness alone is not enough to ensure that CB is correctly implemented (otherwise how can we differentiate with static batching for example). So the scheduler steps tests are meant to check the correctness of the step-by-step execution of continuous batching. It does so by comparing, at every engine step (i.e. prefill or decode iteration), a bunch of attributes. This is allows a finer testing of the padding and scheduling implementation.

* **Checked attributes at each step:**
    * `tkv`: after each step, the tkv is compared against the expected tkv value for that step
    * `waiting`, `running`, `request_outputs`, `finished_requests` not really relevant in a compiler point of view, but after each iteration, we check that the list of running and waiting requests and those that have finished are correct. this tests the scheduler correctness.
    * (waiting to be merged, PR #261): `n_reserved_blocks` and `n_used_blocks`

#### Other Tests

See [Other Tests](tests/other_tests.md)

Most of the other tests primarily verify the correctness of various vLLM Spyre's plugin behaviors, such as launching the online server or enforcing scheduler constraints. While they don't always directly target the correctness of continuous batching, they ensure that the system functions as expected when continuous batching is enabled.
