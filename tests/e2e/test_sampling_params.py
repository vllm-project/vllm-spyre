import pytest
from spyre_util import get_spyre_backend_list
from vllm import LLM, SamplingParams


@pytest.fixture(scope="module")
def spyre_model() -> LLM:
    return LLM(model="ibm-granite/granite-3.1-2b-instruct")


@pytest.fixture(scope="function", autouse=True)
def setenv(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_SPYRE_DYNAMO_BACKEND", get_spyre_backend_list())
    monkeypatch.setenv("VLLM_SPYRE_OVERRIDE_SIGNALS_HANDLER", "1")


def test_spyre_batch1_temperature(spyre_model: LLM):
    prompt = "The capital of the United Kingdom is"
    params1 = SamplingParams(temperature=0.0, seed=8780, max_tokens=5)
    params2 = SamplingParams(temperature=0.5, seed=8780, max_tokens=5)
    params3 = SamplingParams(temperature=1.0, seed=8780, max_tokens=5)

    output1 = spyre_model.generate(prompt, params1)[0]
    output2 = spyre_model.generate(prompt, params2)[0]
    output3 = spyre_model.generate(prompt, params3)[0]

    assert output1.outputs[0].text != output2.outputs[0].text
    assert output2.outputs[0].text != output3.outputs[0].text


def test_spyre_batch1_max_tokens(spyre_model: LLM):
    prompt = "Count to twenty"
    params1 = SamplingParams(temperature=0, seed=8780, max_tokens=15)
    params2 = SamplingParams(temperature=0, seed=8780)

    output1 = spyre_model.generate(prompt, params1)[0]
    output2 = spyre_model.generate(prompt, params2)[0]

    assert len(output1.outputs[0].token_ids) == 15
    assert len(output2.outputs[0].token_ids) > 15


def test_spyre_batch1_stop_sequence(spyre_model: LLM):
    tokenizer = spyre_model.get_tokenizer()
    stop_str = "train"
    stop_word_id = tokenizer.encode(stop_str)
    prompt = "The best way to travel from Paris to Berlim is by "

    params1 = SamplingParams(stop=[stop_str],
                             max_tokens=20,
                             logit_bias={stop_word_id[0]: 100})
    params2 = SamplingParams(max_tokens=20, logit_bias={stop_word_id[0]: 100})

    output1 = spyre_model.generate(prompt, params1)[0]
    output2 = spyre_model.generate(prompt, params2)[0]

    assert stop_str not in output1.outputs[0].text
    assert output1.outputs[0].finish_reason == 'stop'
    assert output2.outputs[0].finish_reason != 'stop'


def test_spyre_batch1_presence_penalty(spyre_model: LLM):
    prompt = "Repeat over and over again: one one"
    word = " one"

    param1 = SamplingParams(presence_penalty=2.0, max_tokens=20)
    param2 = SamplingParams(presence_penalty=0.0, max_tokens=20)

    output = spyre_model.generate(prompt, param1)[0]
    no_penalty = spyre_model.generate(prompt, param2)[0]

    no_penalty_count = no_penalty.outputs[0].text.lower().count(word)

    assert output.outputs[0].text.lower().count(word) < no_penalty_count


def test_spyre_batch1_frequency_penalty(spyre_model: LLM):
    prompt = "Repeat just that word and nothing else \
                over and over again: one one"

    word = " one"

    param1 = SamplingParams(frequency_penalty=2.0, max_tokens=20)
    param2 = SamplingParams(frequency_penalty=0.0, max_tokens=20)

    output = spyre_model.generate(prompt, param1)[0]
    no_penalty = spyre_model.generate(prompt, param2)[0]

    word_count = no_penalty.outputs[0].text.lower().count(word)

    assert output.outputs[0].text.lower().count(word) < word_count


def test_spyre_batch1_n_generations(spyre_model: LLM):
    prompt = "The three most popular sports in the world are: "
    params = SamplingParams(n=3, max_tokens=20)

    output = spyre_model.generate(prompt, params)[0]

    assert len(output.outputs) == 3
    assert output.outputs[0].text != output.outputs[1].text
    assert output.outputs[1].text != output.outputs[2].text


def test_spyre_batch1_top_p(spyre_model: LLM):
    prompt = "The first three letters of the alphabet are"
    params1 = SamplingParams(top_p=0.01, temperature=2, max_tokens=10)
    params2 = SamplingParams(temperature=2, max_tokens=10)

    output1 = spyre_model.generate(prompt, params1)[0]
    output2 = spyre_model.generate(prompt, params2)[0]

    assert "'A', 'B', and 'C'" in output1.outputs[0].text
    assert output1.outputs[0].text != output2.outputs[0].text


def test_spyre_batch1_top_k(spyre_model: LLM):
    prompt = "The opposite of hot is"
    params1 = SamplingParams(top_k=1, seed=42, max_tokens=5)
    params2 = SamplingParams(seed=42, max_tokens=5)

    output1 = spyre_model.generate(prompt, params1)[0]
    output2 = spyre_model.generate(prompt, params2)[0]

    assert "cold" in output1.outputs[0].text.lower()
    assert output1.outputs[0].text != output2.outputs[0].text


def test_spyre_batch1_logit_bias(spyre_model: LLM):
    tokenizer = spyre_model.get_tokenizer()
    banned_word = "train"
    forced_word = "plane"

    banned_ids = tokenizer.encode(banned_word, add_special_tokens=False)
    forced_ids = tokenizer.encode(forced_word, add_special_tokens=False)

    banned_word_id = banned_ids[0]
    forced_word_id = forced_ids[0]

    prompt = "The fastest way to travel between continents is by "
    params1 = SamplingParams(temperature=0,
                             max_tokens=5,
                             logit_bias={
                                 banned_word_id: -100,
                                 forced_word_id: 100,
                             })
    params2 = SamplingParams(temperature=0, max_tokens=5)

    output1 = spyre_model.generate(prompt, params1)[0]
    output2 = spyre_model.generate(prompt, params2)[0]

    assert banned_word not in output1.outputs[0].text.lower()
    assert forced_word in output1.outputs[0].text.lower()

    assert output1.outputs[0].text != output2.outputs[0].text


def test_spyre_batch1_min_tokens(spyre_model: LLM):
    prompt = "Answer only yes or no and do not explain further:\
                 can computers count?"

    params1 = SamplingParams(seed=42, min_tokens=48, max_tokens=50)
    params2 = SamplingParams(seed=42, max_tokens=50)

    output1 = spyre_model.generate(prompt, params1)[0]
    output2 = spyre_model.generate(prompt, params2)[0]

    assert len(output1.outputs[0].token_ids) >= 48
    assert len(output2.outputs[0].token_ids) < 48


def test_spyre_batch1_ignore_eos(spyre_model: LLM):
    prompt = "Answer only yes or no and do not explain \
                further: can computers count?"

    params1 = SamplingParams(ignore_eos=True, max_tokens=50)
    params2 = SamplingParams(ignore_eos=False, max_tokens=50)

    output1 = spyre_model.generate(prompt, params1)[0]
    output2 = spyre_model.generate(prompt, params2)[0]

    assert len(output1.outputs[0].token_ids) == 50
    assert len(output2.outputs[0].token_ids) != len(
        output1.outputs[0].token_ids)

    assert output1.outputs[0].finish_reason == 'length'
    assert output2.outputs[0].finish_reason != 'length'


def test_spyre_batch1_skip_special_tokens(spyre_model: LLM):
    tokenizer = spyre_model.get_tokenizer()
    especial_token_str = tokenizer.eos_token
    especial_token_id = tokenizer.eos_token_id
    prompt = "Hello"

    params1 = SamplingParams(skip_special_tokens=True, max_tokens=5)
    params2 = SamplingParams(skip_special_tokens=False, max_tokens=5)

    output1 = spyre_model.generate(prompt, params1)[0]
    output2 = spyre_model.generate(prompt, params2)[0]

    assert especial_token_id not in output1.outputs[0].token_ids
    assert especial_token_str not in output1.outputs[0].text

    assert especial_token_id in output2.outputs[0].token_ids
    assert especial_token_str in output2.outputs[0].text


def test_spyre_batch1_min_p(spyre_model: LLM):
    prompt = "The opposite of black is"
    params1 = SamplingParams(min_p=0.5, temperature=2, max_tokens=5)
    params2 = SamplingParams(temperature=2, max_tokens=5)

    output1 = spyre_model.generate(prompt, params1)[0]
    output2 = spyre_model.generate(prompt, params2)[0]

    assert "white" in output1.outputs[0].text.lower()
    assert output1.outputs[0].text != output2.outputs[0].text


def test_spyre_batch1_bad_words(spyre_model: LLM):
    prompt = "The capital of France is"
    params1 = SamplingParams(max_tokens=5,
                             temperature=0,
                             bad_words=["Paris", "France"])
    params2 = SamplingParams(max_tokens=5, temperature=0)

    output1 = spyre_model.generate(prompt, params1)[0]
    output2 = spyre_model.generate(prompt, params2)[0]

    assert "Paris" not in output1.outputs[0].text
    assert "France" not in output1.outputs[0].text
    assert output1.outputs[0].text != output2.outputs[0].text


def test_spyre_batch1_detokenize(spyre_model: LLM):
    prompt = "Hello, world!"
    params = SamplingParams(max_tokens=5, temperature=0, detokenize=False)
    output = spyre_model.generate(prompt, params)[0]

    assert output.outputs[0].text == ""
    assert len(output.outputs[0].token_ids) > 0


def test_spyre_batch1_logprobs(spyre_model: LLM):
    num_logprobs = 5
    prompt = "The sky is"
    params = SamplingParams(max_tokens=5, temperature=0, logprobs=num_logprobs)
    output = spyre_model.generate(prompt, params)[0]

    completion_output = output.outputs[0]

    assert completion_output.logprobs is not None
    assert len(completion_output.logprobs) == len(completion_output.token_ids)
    assert len(completion_output.logprobs[0]) == num_logprobs


@pytest.mark.parametrize("backend", "eager")
def test_spyre_dynamic_batch_request_isolation(spyre_model: LLM):

    prompts = [
        "Write an essay on artificial intelligence.",
        "The primary colors are ",
        "Repeat this word once: test",
        "Write a 100-word essay on LLM reasoning capabilities.",
        "Teach me about the history of Isaac Newton",
        "Who is the greatest mathematician of the 21st century?",
    ]

    sampling_params = [
        SamplingParams(temperature=0.7, max_tokens=20, seed=8780),
        SamplingParams(temperature=0.0, max_tokens=2, seed=8780),
        SamplingParams(max_tokens=2, ignore_eos=True, seed=8780),
        SamplingParams(max_tokens=20, ignore_eos=True, seed=8780),
        SamplingParams(max_tokens=20, presence_penalty=2.0),
        SamplingParams(max_tokens=20, min_tokens=9, seed=8780),
    ]

    expected_out = []
    for prompt, param in zip(prompts, sampling_params):
        output = spyre_model.generate(prompt, param)
        expected_out.append(output[0])

    vllm_outputs = spyre_model.generate(prompts, sampling_params)

    # checks isolation
    assert expected_out[0].outputs[0].text == vllm_outputs[0].outputs[0].text
    assert expected_out[1].outputs[0].text == vllm_outputs[0].outputs[1].text
    assert expected_out[2].outputs[0].text == vllm_outputs[0].outputs[2].text
    assert expected_out[3].outputs[0].text == vllm_outputs[0].outputs[3].text
    assert expected_out[4].outputs[0].text == vllm_outputs[0].outputs[4].text
    assert expected_out[5].outputs[0].text == vllm_outputs[0].outputs[5].text
