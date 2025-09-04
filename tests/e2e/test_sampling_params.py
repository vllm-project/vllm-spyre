import pytest
from spyre_util import get_spyre_backend_list
from vllm import SamplingParams

pytestmark = pytest.mark.full_model


@pytest.fixture(scope="function", autouse=True)
@pytest.mark.parametrize("backend", get_spyre_backend_list())
def setenv(backend, monkeypatch):
    monkeypatch.setenv("VLLM_SPYRE_DYNAMO_BACKEND", backend)


def test_spyre_batch1_temperature(model, setenv, monkeypatch, use_llm_cache):
    prompt = "The capital of the United Kingdom is"
    params1 = SamplingParams(temperature=0.0, seed=8780, max_tokens=5)
    params2 = SamplingParams(temperature=0.5, seed=8780, max_tokens=5)
    params3 = SamplingParams(temperature=1.0, seed=8780, max_tokens=5)

    output1 = model.generate(prompt, params1)[0]
    output2 = model.generate(prompt, params2)[0]
    output3 = model.generate(prompt, params3)[0]

    assert output1.outputs[0].text != output2.outputs[0].text
    assert output2.outputs[0].text != output3.outputs[0].text


def test_spyre_batch1_max_tokens(model, setenv, monkeypatch, use_llm_cache):
    prompt = "Count to twenty"
    params1 = SamplingParams(temperature=0, seed=8780, max_tokens=15)
    params2 = SamplingParams(temperature=0, seed=8780)

    output1 = model.generate(prompt, params1)[0]
    output2 = model.generate(prompt, params2)[0]

    assert len(output1.outputs[0].token_ids) == 15
    assert len(output2.outputs[0].token_ids) > 15


def test_spyre_batch1_stop_sequence(model, setenv, monkeypatch, use_llm_cache):
    tokenizer = model.get_tokenizer()
    stop_str = "train"
    stop_word_id = tokenizer.encode(stop_str)
    prompt = "The best way to travel from Paris to Berlim is by "

    params1 = SamplingParams(stop=[stop_str],
                             max_tokens=20,
                             logit_bias={stop_word_id[0]: 100})
    params2 = SamplingParams(max_tokens=20, logit_bias={stop_word_id[0]: 100})

    output1 = model.generate(prompt, params1)[0]
    output2 = model.generate(prompt, params2)[0]

    assert stop_str not in output1.outputs[0].text
    assert output1.outputs[0].finish_reason == 'stop'
    assert output2.outputs[0].finish_reason != 'stop'


def test_spyre_batch1_presence_penalty(model, setenv, monkeypatch,
                                       use_llm_cache):
    prompt = "Repeat over and over again: one one"
    word = " one"

    param1 = SamplingParams(presence_penalty=2.0, max_tokens=20)
    param2 = SamplingParams(presence_penalty=0.0, max_tokens=20)

    with_penalty = model.generate(prompt, param1)[0]
    no_penalty = model.generate(prompt, param2)[0]

    assert with_penalty.outputs[0].text.lower().count(word) < \
                              no_penalty.outputs[0].text.lower().count(word)


def test_spyre_batch1_frequency_penalty(model, setenv, monkeypatch,
                                        use_llm_cache):
    prompt = "Repeat just that word and nothing else \
                over and over again: one one"

    word = " one"

    param1 = SamplingParams(frequency_penalty=2.0, max_tokens=20)
    param2 = SamplingParams(frequency_penalty=0.0, max_tokens=20)

    output = model.generate(prompt, param1)[0]
    no_penalty = model.generate(prompt, param2)[0]

    word_count = no_penalty.outputs[0].text.lower().count(word)

    assert output.outputs[0].text.lower().count(word) < word_count


def test_spyre_batch1_n_generations(model, setenv, monkeypatch, use_llm_cache):
    prompt = "The three most popular sports in the world are: "
    params = SamplingParams(n=3, max_tokens=20)

    output = model.generate(prompt, params)[0]

    assert len(output.outputs) == 3
    assert output.outputs[0].text != output.outputs[1].text
    assert output.outputs[1].text != output.outputs[2].text


def test_spyre_batch1_top_p(model, setenv, monkeypatch, use_llm_cache):
    prompt = "The first three letters of the alphabet are"
    params1 = SamplingParams(top_p=0.01, temperature=2, max_tokens=10)
    params2 = SamplingParams(temperature=2, max_tokens=10)

    output1 = model.generate(prompt, params1)[0]
    output2 = model.generate(prompt, params2)[0]

    assert "'A', 'B', and 'C'" in output1.outputs[0].text
    assert output1.outputs[0].text != output2.outputs[0].text


def test_spyre_batch1_top_k(model, setenv, monkeypatch, use_llm_cache):
    prompt = "The opposite of hot is"
    params1 = SamplingParams(top_k=1, seed=42, max_tokens=5)
    params2 = SamplingParams(seed=42, max_tokens=5)

    output1 = model.generate(prompt, params1)[0]
    output2 = model.generate(prompt, params2)[0]

    assert "cold" in output1.outputs[0].text.lower()
    assert output1.outputs[0].text != output2.outputs[0].text


def test_spyre_batch1_logit_bias(model, setenv, monkeypatch, use_llm_cache):
    tokenizer = model.get_tokenizer()
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

    output1 = model.generate(prompt, params1)[0]
    output2 = model.generate(prompt, params2)[0]

    assert banned_word not in output1.outputs[0].text.lower()
    assert forced_word in output1.outputs[0].text.lower()

    assert output1.outputs[0].text != output2.outputs[0].text


def test_spyre_batch1_min_tokens(model, setenv, monkeypatch, use_llm_cache):
    prompt = "Answer only yes or no and do not explain further:\
                 can computers count?"

    params1 = SamplingParams(seed=42, min_tokens=19, max_tokens=20)
    params2 = SamplingParams(seed=42, max_tokens=20)

    output1 = model.generate(prompt, params1)[0]
    output2 = model.generate(prompt, params2)[0]

    assert len(output1.outputs[0].token_ids) >= 19
    assert len(output2.outputs[0].token_ids) < 19


def test_spyre_batch1_ignore_eos(model, setenv, monkeypatch, use_llm_cache):
    prompt = "Answer only yes or no and do not explain \
                further: can computers count?"

    params1 = SamplingParams(ignore_eos=True, max_tokens=20)
    params2 = SamplingParams(ignore_eos=False, max_tokens=20)

    output1 = model.generate(prompt, params1)[0]
    output2 = model.generate(prompt, params2)[0]

    assert len(output1.outputs[0].token_ids) == 20
    assert len(output2.outputs[0].token_ids) != len(
        output1.outputs[0].token_ids)

    assert output1.outputs[0].finish_reason == 'length'
    assert output2.outputs[0].finish_reason != 'length'


def test_spyre_batch1_min_p(model, setenv, monkeypatch, use_llm_cache):
    prompt = "The opposite of black is"
    params1 = SamplingParams(min_p=0.5, temperature=2, max_tokens=5)
    params2 = SamplingParams(temperature=2, max_tokens=5)

    output1 = model.generate(prompt, params1)[0]
    output2 = model.generate(prompt, params2)[0]

    assert "white" in output1.outputs[0].text.lower()
    assert output1.outputs[0].text != output2.outputs[0].text


def test_spyre_batch1_bad_words(model, setenv, monkeypatch, use_llm_cache):
    prompt = "The capital of France is"
    params1 = SamplingParams(max_tokens=5,
                             temperature=0,
                             bad_words=["Paris", "France"])
    params2 = SamplingParams(max_tokens=5, temperature=0)

    output1 = model.generate(prompt, params1)[0]
    output2 = model.generate(prompt, params2)[0]

    assert "Paris" not in output1.outputs[0].text
    assert "France" not in output1.outputs[0].text
    assert output1.outputs[0].text != output2.outputs[0].text


def test_spyre_batch1_detokenize(model, setenv, monkeypatch, use_llm_cache):
    prompt = "Hello, world!"
    params = SamplingParams(max_tokens=5, temperature=0, detokenize=False)
    output = model.generate(prompt, params)[0]

    assert output.outputs[0].text == ""
    assert len(output.outputs[0].token_ids) > 0


def test_spyre_batch1_logprobs(model, setenv, monkeypatch, use_llm_cache):
    num_logprobs = 5
    prompt = "The sky is"
    params = SamplingParams(max_tokens=5, temperature=0, logprobs=num_logprobs)
    output = model.generate(prompt, params)[0]

    completion_output = output.outputs[0]

    assert completion_output.logprobs is not None
    assert len(completion_output.logprobs) == len(completion_output.token_ids)
    assert len(completion_output.logprobs[0]) == num_logprobs
