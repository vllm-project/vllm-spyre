import pytest
from spyre_util import get_spyre_backend_list, get_spyre_model_list
from vllm import LLM, SamplingParams


@pytest.fixture(scope="module")
@pytest.mark.parametrize("model", get_spyre_model_list())
def spyre_model(model: str) -> LLM:
    return LLM(model=model)


@pytest.mark.parametrize("backend", get_spyre_backend_list())
def test_spyre_batch1_temperature(spyre_model: LLM, backend: str,
                                  monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_SPYRE_DYNAMO_BACKEND", backend)
    monkeypatch.setenv("VLLM_SPYRE_OVERRIDE_SIGNALS_HANDLER", "1")

    prompt = "The capital of the United Kingdom is"
    params = SamplingParams(temperature=0.0, seed=8780, max_tokens=5)

    output1 = spyre_model.generate(prompt, params, request_id="1")[0]
    output2 = spyre_model.generate(prompt, params, request_id="2")[0]

    assert output1.outputs[0].text == output2.outputs[0].text
    assert "London" in output1.outputs[0].text


@pytest.mark.parametrize("backend", get_spyre_backend_list())
def test_spyre_batch1_max_tokens(spyre_model: LLM, backend: str,
                                 monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_SPYRE_DYNAMO_BACKEND", backend)
    monkeypatch.setenv("VLLM_SPYRE_OVERRIDE_SIGNALS_HANDLER", "1")

    prompt = "Count to twenty"
    params = SamplingParams(temperature=0, seed=8780, max_tokens=15)

    output = spyre_model.generate(prompt, params, request_id="1")[0]

    assert len(output.outputs[0].token_ids) == 15


@pytest.mark.parametrize("backend", get_spyre_backend_list())
def test_spyre_batch1_stop_sequence(spyre_model: LLM, backend: str,
                                    monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_SPYRE_DYNAMO_BACKEND", backend)
    monkeypatch.setenv("VLLM_SPYRE_OVERRIDE_SIGNALS_HANDLER", "1")

    stop_str = "London"
    prompt = f"The best way to travel from Paris to {stop_str} is by train."
    params = SamplingParams(stop=[stop_str], max_tokens=50)

    output = spyre_model.generate(prompt, params, request_id="1")[0]

    assert stop_str not in output.outputs[0].text
    assert output.outputs[0].finish_reason == 'stop'


@pytest.mark.parametrize("backend", get_spyre_backend_list())
def test_spyre_batch1_presence_penalty(spyre_model: LLM, backend: str,
                                       monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_SPYRE_DYNAMO_BACKEND", backend)
    monkeypatch.setenv("VLLM_SPYRE_OVERRIDE_SIGNALS_HANDLER", "1")

    prompt = "Repeat over and over again: a new day. Repeat: a new day."

    param1 = SamplingParams(presence_penalty=2.0, max_tokens=50)
    output = spyre_model.generate(prompt, param1, request_id="1")[0]

    param2 = SamplingParams(presence_penalty=0.0, max_tokens=50)
    no_penalty = spyre_model.generate(prompt, param2, request_id="2")[0]
    no_penalty_count = no_penalty.outputs[0].text.lower().count("a new day")

    assert output.outputs[0].text.lower().count("a new day") < no_penalty_count


@pytest.mark.parametrize("backend", get_spyre_backend_list())
def test_spyre_batch1_frequency_penalty(spyre_model: LLM, backend: str,
                                        monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_SPYRE_DYNAMO_BACKEND", backend)
    monkeypatch.setenv("VLLM_SPYRE_OVERRIDE_SIGNALS_HANDLER", "1")

    prompt = "Add fruits to the list: apple, banana, apple, banana"

    param1 = SamplingParams(frequency_penalty=2.0, max_tokens=50)
    output = spyre_model.generate(prompt, param1, request_id="1")[0]

    param2 = SamplingParams(presence_penalty=0.0, max_tokens=50)
    no_penalty = spyre_model.generate(prompt, param2, request_id="2")[0]

    first_word_count = no_penalty.outputs[0].text.lower().count("banana")
    second_word_count = no_penalty.outputs[0].text.lower().count("apple")

    assert output.outputs[0].text.lower().count("apple") < first_word_count
    assert output.outputs[0].text.lower().count("banana") < second_word_count


@pytest.mark.parametrize("backend", get_spyre_backend_list())
def test_spyre_batch1_n_generations(spyre_model: LLM, backend: str,
                                    monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_SPYRE_DYNAMO_BACKEND", backend)
    monkeypatch.setenv("VLLM_SPYRE_OVERRIDE_SIGNALS_HANDLER", "1")

    prompt = "The three most popular sports in the world are: "

    params = SamplingParams(n=3, temperature=0.8, seed=8780, max_tokens=50)
    output = spyre_model.generate(prompt, params, request_id="1")[0]

    assert len(output.outputs) == 3
    assert output.outputs[0].text != output.outputs[1].text
    assert output.outputs[1].text != output.outputs[2].text


@pytest.mark.parametrize("backend", get_spyre_backend_list())
def test_spyre_batch1_top_p(spyre_model: LLM, backend: str,
                            monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_SPYRE_DYNAMO_BACKEND", backend)
    monkeypatch.setenv("VLLM_SPYRE_OVERRIDE_SIGNALS_HANDLER", "1")

    prompt = "The first three letters of the alphabet are "

    params = SamplingParams(top_p=0.01, temperature=0.5, max_tokens=10)
    output = spyre_model.generate(prompt, params, request_id="1")[0]

    assert "A, B and C" in output.outputs[0].text


@pytest.mark.parametrize("backend", get_spyre_backend_list())
def test_spyre_batch1_top_k(spyre_model: LLM, backend: str,
                            monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_SPYRE_DYNAMO_BACKEND", backend)
    monkeypatch.setenv("VLLM_SPYRE_OVERRIDE_SIGNALS_HANDLER", "1")

    prompt = "The opposite of hot is "

    params = SamplingParams(top_k=1, max_tokens=5)
    output = spyre_model.generate(prompt, params, request_id="1")[0]

    assert "cold" in output.outputs[0].text.lower()


@pytest.mark.parametrize("backend", get_spyre_backend_list())
def test_spyre_batch1_logit_bias(spyre_model: LLM, backend: str,
                                 monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_SPYRE_DYNAMO_BACKEND", backend)
    monkeypatch.setenv("VLLM_SPYRE_OVERRIDE_SIGNALS_HANDLER", "1")

    tokenizer = spyre_model.get_tokenizer()
    banned_word = "train"
    forced_word = "plane"

    banned_ids = tokenizer.encode(banned_word, add_special_tokens=False)
    forced_ids = tokenizer.encode(forced_word, add_special_tokens=False)

    banned_word_id = banned_ids[0]
    forced_word_id = forced_ids[0]

    prompt = "The fastest way to travel between continents is by "

    params = SamplingParams(temperature=0,
                            max_tokens=5,
                            logit_bias={
                                banned_word_id: -100,
                                forced_word_id: 100,
                            })

    output = spyre_model.generate(prompt, params, request_id="1")[0]

    assert banned_word not in output.outputs[0].text.lower()
    assert forced_word in output.outputs[0].text.lower()


@pytest.mark.parametrize("backend", get_spyre_backend_list())
def test_spyre_batch1_min_tokens(spyre_model: LLM, backend: str,
                                 monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_SPYRE_DYNAMO_BACKEND", backend)
    monkeypatch.setenv("VLLM_SPYRE_OVERRIDE_SIGNALS_HANDLER", "1")

    prompt = "Hello."
    params = SamplingParams(temperature=0, min_tokens=20, max_tokens=25)

    output = spyre_model.generate(prompt, params, request_id="1")[0]

    assert len(output.outputs[0].tokens_ids) >= 20


@pytest.mark.parametrize("backend", get_spyre_backend_list())
def test_spyre_batch1_ignore_eos(spyre_model: LLM, backend: str,
                                 monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_SPYRE_DYNAMO_BACKEND", backend)
    monkeypatch.setenv("VLLM_SPYRE_OVERRIDE_SIGNALS_HANDLER", "1")

    prompt = "One plus one equals two."
    params = SamplingParams(temperature=0, ignore_eos=True, max_tokens=100)

    output = spyre_model.generate(prompt, params, request_id="1")[0]

    assert output.outputs[0].finish_reason == 'length'
    assert len(output.outputs[0].tokens_ids) == 100


@pytest.mark.parametrize("backend", get_spyre_backend_list())
def test_spyre_dynamic_batch_isolation(spyre_model: LLM, backend: str,
                                       monkeypatch: pytest.MonkeyPatch):

    monkeypatch.setenv("VLLM_SPYRE_DYNAMO_BACKEND", backend)
    monkeypatch.setenv("VLLM_SPYRE_OVERRIDE_SIGNALS_HANDLER", "1")

    prompts = [
        "Write an essay on artificial intelligence.",
        "The primary colors are ",
        "Repeat this word once: test",
        "Write a 100-word essay on LLM reasoning capabilities.",
        "Teach me about the history of Isaac Newton",
        "Who is the greatest mathematician of the 21st century?",
    ]

    sampling_params = [
        SamplingParams(temperature=0.7, max_tokens=100, seed=8780),
        SamplingParams(temperature=0.0, max_tokens=10, seed=8780),
        SamplingParams(max_tokens=20, ignore_eos=True, seed=8780),
        SamplingParams(max_tokens=100, ignore_eos=True, seed=8780),
        SamplingParams(max_tokens=100, presence_penalty=2.0),
        SamplingParams(max_tokens=100, min_tokens=90, seed=8780),
    ]

    expected_out = []
    for prompt, param in zip(prompts, sampling_params):
        output = spyre_model.generate(prompt, param)
        expected_out.append(output[0])

    vllm_outputs = spyre_model.generate(prompts=prompts,
                                        sampling_params=sampling_params)

    # checks isolation
    assert expected_out[0].outputs[0].text == vllm_outputs[0].outputs[0].text
    assert expected_out[1].outputs[0].text == vllm_outputs[0].outputs[1].text
    assert expected_out[2].outputs[0].text == vllm_outputs[0].outputs[2].text
    assert expected_out[3].outputs[0].text == vllm_outputs[0].outputs[3].text
    assert expected_out[4].outputs[0].text == vllm_outputs[0].outputs[4].text
    assert expected_out[5].outputs[0].text == vllm_outputs[0].outputs[5].text
