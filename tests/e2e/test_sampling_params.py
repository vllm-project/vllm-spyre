from collections import defaultdict

import pytest
from llm_cache import get_cached_llm
from spyre_util import ModelInfo
from vllm import SamplingParams

pytestmark = [pytest.mark.full_model, pytest.mark.other_e2e]

sb_mark = pytest.param("sb", marks=pytest.mark.sb, id="sb")
cb_mark = pytest.param("cb", marks=pytest.mark.cb, id="cb")
cp_mark = pytest.param("cp", marks=pytest.mark.chunked_prefill, id="cp")


def test_spyre_batch1_temperature(
    model: ModelInfo, backend, monkeypatch, use_llm_cache, warmup_shapes
):
    spyre_model = get_cached_llm(
        model=model,
        max_model_len=128,
        tensor_parallel_size=1,
        backend=backend,
        monkeypatch=monkeypatch,
        warmup_shapes=warmup_shapes,
    )

    prompt = "The capital of the United Kingdom is"
    params1 = SamplingParams(temperature=0.0, seed=8780, max_tokens=20)
    params2 = SamplingParams(temperature=0.5, seed=8780, max_tokens=20)
    params3 = SamplingParams(temperature=1.0, seed=8780, max_tokens=20)

    output1 = spyre_model.generate(prompt, params1)[0]
    output2 = spyre_model.generate(prompt, params2)[0]
    output3 = spyre_model.generate(prompt, params3)[0]

    assert output1.outputs[0].text != output2.outputs[0].text
    assert output2.outputs[0].text != output3.outputs[0].text


def test_spyre_batch1_max_tokens(
    model: ModelInfo, backend, monkeypatch, use_llm_cache, warmup_shapes
):
    spyre_model = get_cached_llm(
        model=model,
        max_model_len=128,
        tensor_parallel_size=1,
        backend=backend,
        monkeypatch=monkeypatch,
        warmup_shapes=warmup_shapes,
    )

    prompt = "Count to twenty"
    params1 = SamplingParams(temperature=0, seed=8780, max_tokens=15)
    params2 = SamplingParams(temperature=0, seed=8780)

    output1 = spyre_model.generate(prompt, params1)[0]
    output2 = spyre_model.generate(prompt, params2)[0]

    assert len(output1.outputs[0].token_ids) == 15
    assert len(output2.outputs[0].token_ids) > 15


@pytest.mark.xfail(reason="Failing currently because of output mismatch")
def test_spyre_batch1_stop_sequence(
    model: ModelInfo, backend, monkeypatch, use_llm_cache, warmup_shapes
):
    spyre_model = get_cached_llm(
        model=model,
        max_model_len=128,
        tensor_parallel_size=1,
        backend=backend,
        monkeypatch=monkeypatch,
        warmup_shapes=warmup_shapes,
    )
    stop_str = "train"
    prompt = "The best way to travel from Paris to Berlim is by "

    params1 = SamplingParams(stop=[stop_str], max_tokens=20, seed=8780)
    params2 = SamplingParams(max_tokens=20, seed=8780)

    output1 = spyre_model.generate(prompt, params1)[0]
    output2 = spyre_model.generate(prompt, params2)[0]

    assert stop_str not in output1.outputs[0].text
    assert output1.outputs[0].finish_reason == "stop"
    assert output2.outputs[0].finish_reason != "stop"


def max_repetitions(output):
    histo = defaultdict(int)
    for token in output.outputs[0].token_ids:
        histo[token] += 1

    return max(histo.values())


def test_spyre_batch1_presence_penalty(
    model: ModelInfo, backend, monkeypatch, use_llm_cache, warmup_shapes
):
    spyre_model = get_cached_llm(
        model=model,
        max_model_len=128,
        tensor_parallel_size=1,
        backend=backend,
        monkeypatch=monkeypatch,
        warmup_shapes=warmup_shapes,
    )
    prompt = "REPEAT OVER AND OVER AGAIN THE MINIMUM TIMES POSSIBLE: one one one one one"

    param1 = SamplingParams(presence_penalty=2.0, seed=8780, max_tokens=20)
    param2 = SamplingParams(presence_penalty=-2.0, seed=8780, max_tokens=20)

    with_penalty = spyre_model.generate(prompt, param1)[0]
    no_penalty = spyre_model.generate(prompt, param2)[0]

    with_penalty_max = max_repetitions(with_penalty)
    no_penalty_max = max_repetitions(no_penalty)

    assert no_penalty_max > 1
    assert no_penalty_max > with_penalty_max


def test_spyre_batch1_frequency_penalty(
    model: ModelInfo, backend, monkeypatch, use_llm_cache, warmup_shapes
):
    spyre_model = get_cached_llm(
        model=model,
        max_model_len=128,
        tensor_parallel_size=1,
        backend=backend,
        monkeypatch=monkeypatch,
        warmup_shapes=warmup_shapes,
    )

    prompt = "repeat the word hi ten times:"

    param1 = SamplingParams(frequency_penalty=2.0, seed=8780, max_tokens=20)
    param2 = SamplingParams(frequency_penalty=-2.0, seed=8780, max_tokens=20)

    with_penalty = spyre_model.generate(prompt, param1)[0]
    no_penalty = spyre_model.generate(prompt, param2)[0]

    with_penalty_max = max_repetitions(with_penalty)
    no_penalty_max = max_repetitions(no_penalty)
    assert no_penalty_max > 1
    assert no_penalty_max > with_penalty_max


def test_spyre_batch1_n_generations(
    model: ModelInfo, backend, monkeypatch, use_llm_cache, warmup_shapes
):
    spyre_model = get_cached_llm(
        model=model,
        max_model_len=128,
        tensor_parallel_size=1,
        backend=backend,
        monkeypatch=monkeypatch,
        warmup_shapes=warmup_shapes,
    )
    prompt = "The three most popular sports in the world are: "

    params = SamplingParams(n=3, seed=8780, max_tokens=20)

    output = spyre_model.generate(prompt, params)[0]

    assert len(output.outputs) == 3
    assert output.outputs[0].text != output.outputs[1].text
    assert output.outputs[1].text != output.outputs[2].text


def token_diversity(spyre_model, prompt, params, n_experiments):
    tokens = []

    outputs = spyre_model.generate([prompt] * n_experiments, params, use_tqdm=False)
    for output in outputs:
        tokens.extend(output.outputs[0].token_ids)

    return len(set(tokens))


def test_spyre_batch1_top_p(model: ModelInfo, backend, monkeypatch, use_llm_cache, warmup_shapes):
    spyre_model = get_cached_llm(
        model=model,
        max_model_len=128,
        tensor_parallel_size=1,
        backend=backend,
        monkeypatch=monkeypatch,
        warmup_shapes=warmup_shapes,
    )
    prompt = "The first three letters of the alphabet are"
    params1 = SamplingParams(top_p=0.01, temperature=1, max_tokens=10)
    params2 = SamplingParams(temperature=1, max_tokens=10)

    token_div1 = token_diversity(spyre_model, prompt, params1, 10)
    token_div2 = token_diversity(spyre_model, prompt, params2, 10)
    assert token_div1 < token_div2


def test_spyre_batch1_top_k(model: ModelInfo, backend, monkeypatch, use_llm_cache, warmup_shapes):
    spyre_model = get_cached_llm(
        model=model,
        max_model_len=128,
        tensor_parallel_size=1,
        backend=backend,
        monkeypatch=monkeypatch,
        warmup_shapes=warmup_shapes,
    )
    prompt = "The opposite of hot is"
    params1 = SamplingParams(temperature=1, top_k=1, max_tokens=5)
    params2 = SamplingParams(temperature=1, max_tokens=5)

    token_div1 = token_diversity(spyre_model, prompt, params1, 10)
    token_div2 = token_diversity(spyre_model, prompt, params2, 10)
    assert token_div1 < token_div2


@pytest.mark.parametrize("mode", [sb_mark, cb_mark, cp_mark])
def test_spyre_batch1_logit_bias(
    model: ModelInfo,
    backend,
    monkeypatch,
    use_llm_cache,
    warmup_shapes,
    max_model_len,
    max_num_seqs,
    mode: str,
):
    spyre_model = get_cached_llm(
        model=model,
        max_model_len=max_model_len,
        tensor_parallel_size=1,
        backend=backend,
        monkeypatch=monkeypatch,
        warmup_shapes=warmup_shapes if mode == "sb" else None,
        max_num_seqs=max_num_seqs if mode == "cb" or mode == "cp" else None,
        max_num_batched_tokens=128 if mode == "cp" else None,
        use_cb=mode == "cb" or mode == "cp",
    )
    tokenizer = spyre_model.get_tokenizer()
    banned_word = "train"
    forced_word = "plane"

    banned_ids = tokenizer.encode(banned_word, add_special_tokens=False)
    forced_ids = tokenizer.encode(forced_word, add_special_tokens=False)

    banned_word_id = banned_ids[0]
    forced_word_id = forced_ids[0]

    prompt = "The fastest way to travel between continents is by "
    params1 = SamplingParams(
        temperature=0,
        max_tokens=5,
        seed=8780,
        logit_bias={
            banned_word_id: -100,
            forced_word_id: 100,
        },
    )
    params2 = SamplingParams(temperature=0, seed=8780, max_tokens=5)

    output = spyre_model.generate([prompt, prompt], [params1, params2])

    assert banned_word not in output[0].outputs[0].text.lower()
    assert forced_word in output[0].outputs[0].text.lower()

    assert output[0].outputs[0].text != output[1].outputs[0].text


@pytest.mark.parametrize("mode", [sb_mark, cb_mark, cp_mark])
def test_spyre_batch1_min_tokens(
    model: ModelInfo,
    backend,
    monkeypatch,
    use_llm_cache,
    max_model_len,
    max_num_seqs,
    warmup_shapes,
    mode: str,
):
    spyre_model = get_cached_llm(
        model=model,
        max_model_len=max_model_len,
        tensor_parallel_size=1,
        backend=backend,
        monkeypatch=monkeypatch,
        warmup_shapes=warmup_shapes if mode == "sb" else None,
        max_num_seqs=max_num_seqs if mode == "cb" or mode == "cp" else None,
        max_num_batched_tokens=128 if mode == "cp" else None,
        use_cb=mode == "cb" or mode == "cp",
    )
    prompt = "What is the capital of the USA?"
    tokenizer = spyre_model.get_tokenizer()
    eos_id = tokenizer.eos_token_id

    params1 = SamplingParams(min_tokens=10, logit_bias={eos_id: 1000}, seed=8780, max_tokens=20)
    params2 = SamplingParams(seed=8780, logit_bias={eos_id: 1000}, max_tokens=20)

    output = spyre_model.generate([prompt] * 2, [params1, params2])

    # Logits bias should force eos token appears, then we check if
    # after min tokens reached the logits processor is properly
    # cleared. Therefore token count shall be 10 + 1
    # (min_tokens + eos_token_id)
    assert len(output[0].outputs[0].token_ids) == 11
    assert len(output[1].outputs[0].token_ids) == 1


def test_spyre_batch1_ignore_eos(
    model: ModelInfo, backend, monkeypatch, use_llm_cache, warmup_shapes
):
    spyre_model = get_cached_llm(
        model=model,
        max_model_len=128,
        tensor_parallel_size=1,
        backend=backend,
        monkeypatch=monkeypatch,
        warmup_shapes=warmup_shapes,
    )
    tokenizer = spyre_model.get_tokenizer()
    eos_id = tokenizer.eos_token_id
    prompt = "This is the end of the story"

    params1 = SamplingParams(ignore_eos=True, logit_bias={eos_id: 50}, seed=8780, max_tokens=20)
    params2 = SamplingParams(ignore_eos=False, logit_bias={eos_id: 50}, seed=8780, max_tokens=20)

    output1 = spyre_model.generate(prompt, params1)[0]
    output2 = spyre_model.generate(prompt, params2)[0]

    assert len(output1.outputs[0].token_ids) == 20
    assert len(output2.outputs[0].token_ids) != len(output1.outputs[0].token_ids)

    assert output1.outputs[0].finish_reason == "length"
    assert output2.outputs[0].finish_reason != "length"


@pytest.mark.parametrize("mode", [sb_mark, cb_mark, cp_mark])
def test_spyre_batch1_min_p(
    model: ModelInfo,
    backend,
    monkeypatch,
    use_llm_cache,
    max_model_len,
    max_num_seqs,
    warmup_shapes,
    mode: str,
):
    spyre_model = get_cached_llm(
        model=model,
        max_model_len=max_model_len,
        tensor_parallel_size=1,
        backend=backend,
        monkeypatch=monkeypatch,
        warmup_shapes=warmup_shapes if mode == "sb" else None,
        max_num_seqs=max_num_seqs if mode == "cb" or mode == "cp" else None,
        max_num_batched_tokens=128 if mode == "cp" else None,
        use_cb=mode == "cb" or mode == "cp",
    )
    prompt = "The opposite of black is"
    params1 = SamplingParams(min_p=0.5, temperature=1, max_tokens=5)
    params2 = SamplingParams(temperature=1, max_tokens=5)

    token_div1 = token_diversity(spyre_model, prompt, params1, 10)
    token_div2 = token_diversity(spyre_model, prompt, params2, 10)

    assert token_div1 < token_div2


@pytest.mark.xfail(reason="Failing currently because of output mismatch")
def test_spyre_batch1_bad_words(
    model: ModelInfo, backend, monkeypatch, use_llm_cache, warmup_shapes
):
    spyre_model = get_cached_llm(
        model=model,
        max_model_len=128,
        tensor_parallel_size=1,
        backend=backend,
        monkeypatch=monkeypatch,
        warmup_shapes=warmup_shapes,
    )
    prompt = "The capital of France is"
    params1 = SamplingParams(
        max_tokens=5, temperature=0, seed=8780, bad_words=[" Paris", " Parisi", " France"]
    )
    params2 = SamplingParams(max_tokens=5, seed=8780, temperature=0)

    output1 = spyre_model.generate(prompt, params1)[0]
    output2 = spyre_model.generate(prompt, params2)[0]

    assert "Paris" not in output1.outputs[0].text
    assert "France" not in output1.outputs[0].text
    assert output1.outputs[0].text != output2.outputs[0].text


def test_spyre_batch1_detokenize(
    model: ModelInfo, backend, monkeypatch, use_llm_cache, warmup_shapes
):
    spyre_model = get_cached_llm(
        model=model,
        max_model_len=128,
        tensor_parallel_size=1,
        backend=backend,
        monkeypatch=monkeypatch,
        warmup_shapes=warmup_shapes,
    )
    prompt = "Hello, world!"
    params = SamplingParams(max_tokens=5, seed=8780, temperature=0, detokenize=False)
    output = spyre_model.generate(prompt, params)[0]

    assert output.outputs[0].text == ""
    assert len(output.outputs[0].token_ids) > 0


def test_spyre_batch1_logprobs(
    model: ModelInfo, backend, monkeypatch, use_llm_cache, warmup_shapes
):
    spyre_model = get_cached_llm(
        model=model,
        max_model_len=128,
        tensor_parallel_size=1,
        backend=backend,
        monkeypatch=monkeypatch,
        warmup_shapes=warmup_shapes,
    )
    num_logprobs = 5
    prompt = "The sky is"
    params = SamplingParams(max_tokens=5, seed=8780, temperature=0, logprobs=num_logprobs)
    output = spyre_model.generate(prompt, params)[0]

    completion_output = output.outputs[0]

    assert completion_output.logprobs is not None
    assert len(completion_output.logprobs) == len(completion_output.token_ids)
    assert len(completion_output.logprobs[0]) == num_logprobs
