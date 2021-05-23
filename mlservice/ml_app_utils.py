import re
import time

import onnxruntime
from onnxruntime.transformers.gpt2_helper import Gpt2Helper
import numpy
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig

from config import get_logger
from mlservice.ml_sampling_utils import TopKLogitsWarper, TemperatureLogitsWarper, TopPLogitsWarper, NoRepeatNGramLogitsProcessor


_logger = get_logger(logger_name=__name__)


class ModelConfig:
    def __init__(self,
                 model_path,
                 cache_dir,
                 model_optimized_path,
                 device='cpu',
                 tokenizer=None,
                 config=None,
                 session=None,
                 threads=None
                 ):
        self.model_path = model_path
        self.cache_dir = cache_dir
        self.model_optimized_path = model_optimized_path
        self.device = device
        self.tokenizer = tokenizer
        self.config = config
        self.session = session
        self.threads = threads

        if self.tokenizer is None:
            self.get_tokenizer()

        if self.config is None:
            self.get_config()

        if self.session is None:
            self.start_session()

    def get_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, cache_dir=self.cache_dir)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        #tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        _logger.info('Tokenizer is loaded')

    def get_config(self):
        self.config = AutoConfig.from_pretrained(self.model_path, cache_dir=self.cache_dir)
        _logger.info('Config is loaded')

    def start_session(self):
        self.session = onnxruntime.InferenceSession(self.model_optimized_path,
                                                    providers=['CUDAExecutionProvider' if self.device == 'cuda'
                                                               else 'CPUExecutionProvider']
                                                    )
        _logger.info('Session is created')

    def get_inputs(self, prompt_text):
        encodings_dict = self.tokenizer.batch_encode_plus([prompt_text], padding=True)
        input_ids = torch.tensor(encodings_dict['input_ids'], dtype=torch.int64)
        attention_mask = torch.tensor(encodings_dict['attention_mask'], dtype=torch.float32)
        position_ids = (attention_mask.long().cumsum(-1) - 1)
        position_ids.masked_fill_(position_ids < 0, 0)

        # Empty Past State for generating first word
        empty_past = []
        batch_size = input_ids.size(0)
        past_shape = [2, batch_size, self.config.n_head, 0, self.config.n_embd // self.config.n_head]
        for i in range(self.config.n_layer):
            empty_past.append(torch.empty(past_shape).type(torch.float32).to(self.device))
        return input_ids, attention_mask, position_ids, empty_past

    def inference_with_io_binding(self, input_ids, position_ids, attention_mask, past):
        """
        For performing gpt2 inference using our persistant session.
        """

        output_shapes = Gpt2Helper.get_output_shapes(
            batch_size=input_ids.size(0), past_sequence_length=past[0].size(3),
            sequence_length=input_ids.size(1), config=self.config
        )
        # _logger.info('Get output buffers')
        output_buffers = Gpt2Helper.get_output_buffers(output_shapes, self.device)
        # _logger.info('Prepare io binding')
        io_binding = Gpt2Helper.prepare_io_binding(
            self.session, input_ids, position_ids, attention_mask,
            past, output_buffers, output_shapes
        )
        # _logger.info(f'Output shape: {output_shapes}')
        # _logger.info(f'input_ids: {input_ids}')
        # _logger.info(f'past: {len(past)}')
        # _logger.info('Run session')
        # start = time.time()
        self.session.run_with_iobinding(io_binding)
        # _logger.info('Get outputs')
        # _logger.info('{:.2f}'.format(time.time() - start))
        outputs = Gpt2Helper.get_outputs_from_io_binding_buffer(
            self.session, output_buffers, output_shapes,
            return_numpy=False
        )
        return outputs

    def generate_text(self, input_text, temperature, top_k, top_p, no_repeat_ngram_size, num_tokens_to_produce=100):
        input_ids, attention_mask, position_ids, past = self.get_inputs(input_text)
        batch_size = input_ids.size(0)
        has_eos = torch.zeros(batch_size, dtype=torch.bool)
        all_token_ids = input_ids.clone()

        total_time = 0
        inference_time = 0
        custom_time = 0
        other_op_time = 0
        for _ in range(num_tokens_to_produce):
            start = time.time()

            outputs = self.inference_with_io_binding(input_ids, position_ids, attention_mask, past)
            inference_time += time.time() - start
            next_token_logits = outputs[0][:, -1, :]

            start_custom = time.time()

            next_token_logits = NoRepeatNGramLogitsProcessor(no_repeat_ngram_size)(all_token_ids, next_token_logits)
            next_token_logits = TemperatureLogitsWarper(temperature)(next_token_logits)
            next_token_logits = TopKLogitsWarper(top_k)(next_token_logits)
            next_token_logits = TopPLogitsWarper(top_p)(next_token_logits)

            custom_time += time.time() - start_custom
            start_other_time = time.time()
            probs = F.softmax(next_token_logits, dim=-1)

            next_tokens = torch.distributions.categorical.Categorical(probs).sample()
            # next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            # _logger.info(next_tokens)
            other_op_time += time.time() - start_other_time
            #logger.info(f'next tokens softmax: {next_tokens}')
            has_eos = has_eos | (next_tokens == self.tokenizer.eos_token_id)
            # has_eos = has_eos | (next_tokens == 2)
            tokens_to_add = next_tokens.masked_fill(has_eos, self.tokenizer.eos_token_id)
            all_token_ids = torch.cat([all_token_ids, tokens_to_add.unsqueeze(-1)], dim=-1)


            # Update input_ids, attention_mask, position_ids and past
            input_ids = tokens_to_add.clone().detach().reshape([batch_size, 1]).to(self.device)
            position_ids = (position_ids[:, -1] + 1).reshape(batch_size, 1)
            attention_mask = torch.cat([attention_mask, torch.ones([batch_size, 1]).type_as(attention_mask)], 1).to(
                self.device)

            past = []
            for i in range(self.config.n_layer):
                past_i = torch.from_numpy(outputs[i + 1]) if isinstance(outputs[i + 1], numpy.ndarray) else outputs[
                    i + 1].clone().detach()
                past.append(past_i.to(self.device))

            if torch.all(has_eos):
                break

            total_time += time.time() - start
            # _logger.info('Next step')

        _logger.info('Inference time {:.2f}'.format(inference_time) + 'sec')
        _logger.info('Custom time {:.2f}'.format(custom_time) + 'sec')
        _logger.info('Other op time {:.2f}'.format(other_op_time) + 'sec')
        _logger.info('Total time {:.2f}'.format(total_time) + 'sec')
        return "".join(
            [self.tokenizer.decode(output, skip_special_tokens=True) for i, output in enumerate(all_token_ids)])

    def test_generation(self, input_text, num_tokens_to_produce=30):
        _logger.info('Start generation')

        eos_token_id = self.tokenizer.eos_token_id

        input_ids, attention_mask, position_ids, past = self.get_inputs(input_text)
        batch_size = input_ids.size(0)
        _logger.info('Start inference_with_io_binding')

        has_eos = torch.zeros(batch_size, dtype=torch.bool)

        all_token_ids = input_ids.clone()

        total_time = 0
        inference_time = 0

        for step in range(num_tokens_to_produce):
            start = time.time()

            outputs = self.inference_with_io_binding(input_ids, position_ids, attention_mask, past)
            inference_time += time.time() - start

            next_token_logits = outputs[0][:, -1, :]
            # Greedy approach is used here. You can easily extend it to use beam search and sampling to pick next tokens.
            next_tokens = torch.argmax(next_token_logits, dim=-1)

            has_eos = has_eos | (next_tokens == eos_token_id)
            tokens_to_add = next_tokens.masked_fill(has_eos, eos_token_id)
            all_token_ids = torch.cat([all_token_ids, tokens_to_add.unsqueeze(-1)], dim=-1)

            # Update input_ids, attention_mask, position_ids and past
            input_ids = tokens_to_add.clone().detach().reshape([batch_size, 1]).to(self.device)
            position_ids = (position_ids[:, -1] + 1).reshape(batch_size, 1)
            attention_mask = torch.cat([attention_mask, torch.ones([batch_size, 1]).type_as(attention_mask)], 1).to(
                self.device)

            past = []
            for i in range(self.config.n_layer):
                past_i = torch.from_numpy(outputs[i + 1]) if isinstance(outputs[i + 1], numpy.ndarray) else outputs[
                    i + 1].clone().detach()
                past.append(past_i.to(self.device))

            if torch.all(has_eos):
                break
            total_time += time.time() - start

        _logger.info('Inference time {:.2f}'.format(inference_time) + 'sec')
        _logger.info('Total time {:.2f}'.format(total_time) + 'sec')
        return "".join(
            [self.tokenizer.decode(output, skip_special_tokens=True) for i, output in enumerate(all_token_ids)])

    @staticmethod
    def postprocess_text(inference):

        output = re.sub('^<s>', '', inference)
        output = re.sub('</s>.*$', '', output)
        # text_ext = re.sub('<.*$', '', text_ext)
        output = output.rstrip().lstrip()
        output = output.split('. ')

        return ". ".join(list(dict.fromkeys(output))[:4])
