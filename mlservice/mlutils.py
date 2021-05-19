import os
import sys
sys.path.append('..')

import dotenv

import onnxruntime
from onnxruntime.transformers.gpt2_helper import Gpt2Helper
import numpy
import torch
from transformers import AutoTokenizer, AutoConfig

import config

dotenv.load_dotenv()

config_vars = {'dev': config.DevelopmentConfig, 'prod': config.ProductionConfig}[os.environ.get('CONFIGURATION_SETUP')]
MODEL_PATH = 'models/' + config_vars.MODEL_PATH
MODEL_OPTIMIZED_PATH = 'models/' + config_vars.MODEL_OPTIMIZED_PATH

print(os.listdir('models/' + config_vars.MODEL_PATH))

CACHE_DIR = config_vars.CACHE_DIR
CACHE_DIR = os.path.join(".", CACHE_DIR)
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

MODEL_CONFIG = AutoConfig.from_pretrained(MODEL_PATH, cache_dir=CACHE_DIR)
print('Loaded model config')
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, config=MODEL_CONFIG, cache_dir=CACHE_DIR)
print('loaded tokenizer')

NUM_ATTENTION_HEADS = MODEL_CONFIG.n_head
HIDDEN_SIZE = MODEL_CONFIG.n_embd
NUM_LAYER = MODEL_CONFIG.n_layer

DEVICE = torch.device("cpu")


# def get_tokenizer():
#     tokenizer = AutoTokenizer.from_pretrained('alienspaceman/rus_dreamgen_fulltext_medium', cache_dir=CACHE_DIR,use_fast=True)
#     tokenizer.padding_side = "left"
#     tokenizer.pad_token = tokenizer.eos_token
#     #okenizer.add_special_tokens({'pad_token': '[PAD]'})
#     return tokenizer
#
#
def start_session(file_path):
    session_int8 = onnxruntime.InferenceSession(file_path)
    return session_int8


# TOKENIZER = get_tokenizer()
SESSION = start_session(MODEL_OPTIMIZED_PATH)


def get_example_inputs(prompt_text=''):
    tokenizer = get_tokenizer()
    encodings_dict = tokenizer.batch_encode_plus(prompt_text, padding=True)

    input_ids = torch.tensor(encodings_dict['input_ids'], dtype=torch.int64)
    attention_mask = torch.tensor(encodings_dict['attention_mask'], dtype=torch.float32)
    position_ids = (attention_mask.long().cumsum(-1) - 1)
    position_ids.masked_fill_(position_ids < 0, 0)

    # Empty Past State for generating first word
    empty_past = []
    batch_size = input_ids.size(0)
    sequence_length = input_ids.size(1)
    past_shape = [2, batch_size, NUM_ATTENTION_HEADS, 0, HIDDEN_SIZE // NUM_ATTENTION_HEADS]
    for i in range(NUM_LAYER):
        empty_past.append(torch.empty(past_shape).type(torch.float32).to(DEVICE))

    return input_ids, attention_mask, position_ids, empty_past


def inference_with_io_binding(input_ids, position_ids, attention_mask, past):
    output_shapes = Gpt2Helper.get_output_shapes(batch_size=input_ids.size(0),
                                                 past_sequence_length=past[0].size(3),
                                                 sequence_length=input_ids.size(1),
                                                 config=MODEL_CONFIG)
    output_buffers = Gpt2Helper.get_output_buffers(output_shapes, DEVICE)

    io_binding = Gpt2Helper.prepare_io_binding(SESSION, input_ids, position_ids, attention_mask, past,
                                               output_buffers, output_shapes)
    SESSION.run_with_iobinding(io_binding)

    outputs = Gpt2Helper.get_outputs_from_io_binding_buffer(SESSION, output_buffers, output_shapes,
                                                            return_numpy=False)
    return outputs


def test_generation(input_text, num_tokens_to_produce=30):

    eos_token_id = TOKENIZER.eos_token_id

    input_ids, attention_mask, position_ids, past = get_example_inputs(input_text)
    batch_size = input_ids.size(0)

    has_eos = torch.zeros(batch_size, dtype=torch.bool)

    all_token_ids = input_ids.clone()

    for step in range(num_tokens_to_produce):
        outputs = inference_with_io_binding(input_ids, position_ids, attention_mask, past)

        next_token_logits = outputs[0][:, -1, :]
        # Greedy approach is used here. You can easily extend it to use beam search and sampling to pick next tokens.
        next_tokens = torch.argmax(next_token_logits, dim=-1)

        has_eos = has_eos | (next_tokens == eos_token_id)
        tokens_to_add = next_tokens.masked_fill(has_eos, eos_token_id)
        all_token_ids = torch.cat([all_token_ids, tokens_to_add.unsqueeze(-1)], dim=-1)

        # Update input_ids, attention_mask, position_ids and past
        input_ids = tokens_to_add.clone().detach().reshape([batch_size, 1]).to(DEVICE)
        position_ids = (position_ids[:, -1] + 1).reshape(batch_size, 1)
        attention_mask = torch.cat([attention_mask, torch.ones([batch_size, 1]).type_as(attention_mask)], 1).to(DEVICE)

        past = []
        for i in range(NUM_LAYER):
            past_i = torch.from_numpy(outputs[i + 1]) if isinstance(outputs[i + 1], numpy.ndarray) else outputs[
                i + 1].clone().detach()
            past.append(past_i.to(DEVICE))

        if torch.all(has_eos):
            break

    return "".join([TOKENIZER.decode(output, skip_special_tokens=True) for i, output in enumerate(all_token_ids)])
