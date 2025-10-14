from vllm_spyre.v1.sample.spyre_logits_processor import SpyreMinPLogitsProcessor, SpyreMinTokensLogitsProcessor, SpyreLogitsProcessor

import math
from vllm.config import VllmConfig
from vllm.v1.sample.logits_processor import BatchUpdate
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import BatchUpdateBuilder
import pytest
import torch

EOS_TOKEN = 3
VOCAB_SIZE = 8

class DummyVllmConfig:

    def __init__(self):
        self.scheduler_config = DummySchedulerConfig()

class DummySchedulerConfig:
    
    def __init__(self, max_num_seqs=1):
        self.max_num_seqs = max_num_seqs


def generate_logits(batch_size: int =1):

    return torch.tensor(list(range(VOCAB_SIZE)) * batch_size, 
                        dtype=torch.float32).reshape((batch_size, VOCAB_SIZE) )


def prefill(params: SamplingParams, 
            batch_update_builder : BatchUpdateBuilder,
            lp : SpyreLogitsProcessor,
            logits : torch.Tensor,
            ouput_tokens : list[int],
            req_idx : int,
            num_reqs: int):
    params._all_stop_token_ids = set([EOS_TOKEN]) # 
    prompt_tokens = [0] * 8
    batch_update_builder.added.append((req_idx, params, prompt_tokens, ouput_tokens))
    batch_update = batch_update_builder.get_and_reset(num_reqs)
    lp.update_state(batch_update)

    lp.set_prefill_index(req_idx)
    out_logits = lp.apply(logits.clone())
    ouput_tokens.append(0) # just append a random token

    return out_logits

def decode(batch_update_builder : BatchUpdateBuilder,
            lp : SpyreLogitsProcessor,
            logits : torch.Tensor,
            batch_ouput_tokens : list[list[int]]):
    
    assert logits.shape[0] == len(batch_ouput_tokens)

    # This is called at each execute model in spyre model runner update_states
    lp.update_state(None) 

    out_logits = lp.apply(logits.clone())

    for output_tokens in batch_ouput_tokens:
        output_tokens.append(0) # just append a random token

    return out_logits
    
@pytest.mark.cpu
@pytest.mark.worker
def test_mintokens_logits_processor():
    '''
    Tests the builtin SpyreMinTokensLogitsProcessor, 
    '''

    device = torch.device('cpu')
    
    dummy_config = DummyVllmConfig()
    lp = SpyreMinTokensLogitsProcessor(dummy_config, device, False)

    batch_update_builder = BatchUpdateBuilder()

    batch_output_tokens = [[], [], []]


    # Step #0 Prefill req_id #0 (no min tokens)
    logits = generate_logits(1)
    out_logits = prefill(SamplingParams(), 
                         batch_update_builder,
                         lp, 
                         logits, 
                         batch_output_tokens[0],
                         req_idx=0,
                         num_reqs=1)

    assert torch.allclose(logits, out_logits) # Do nothing

    # Step #1 Prefill req_id #1 (with min tokens)
    params = SamplingParams(min_tokens=4)
    
    logits = generate_logits(1)
    out_logits = prefill(params, 
                         batch_update_builder,
                         lp, 
                         logits, 
                         batch_output_tokens[1],
                         req_idx=1,
                         num_reqs=2)

    assert out_logits[0][EOS_TOKEN].item() == -math.inf

    # Step #2 Prefill req_id #1
    logits = generate_logits(1)
    out_logits = prefill(SamplingParams(), 
                         batch_update_builder,
                         lp, 
                         logits, 
                         batch_output_tokens[2],
                         req_idx=2,
                         num_reqs=3)

    assert torch.allclose(logits, out_logits) # Do nothing

    # Step #3 - #6 Decoding, eos_token for req #1 must be -inf
    for _ in range(3):
        logits = generate_logits(3)
        out_logits = decode(batch_update_builder, 
                            lp, 
                            logits, 
                            batch_output_tokens,
                            )
        
        assert torch.allclose(logits[0], out_logits[0])
        assert torch.allclose(logits[2], out_logits[2])
        assert out_logits[1][EOS_TOKEN].item() == -math.inf


    # Step #7, min tokens reached, no changes in logits anymore
    logits = generate_logits(3)
    out_logits = decode(batch_update_builder, 
                        lp, 
                        logits, 
                        batch_output_tokens,
                        )
    
    assert torch.allclose(logits, out_logits)
    
