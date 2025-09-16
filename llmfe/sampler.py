""" Class for sampling new programs. """
from __future__ import annotations
from abc import ABC, abstractmethod

from typing import Collection, Sequence, Type
import numpy as np
import time

import torch
from transformers import (
    pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
)

from llmfe import evaluator
from llmfe import buffer
from llmfe import config as config_lib
import requests
import json
import http.client
import os

class LLM(ABC):
    def __init__(self, samples_per_prompt: int) -> None:
        self._samples_per_prompt = samples_per_prompt

    def _draw_sample(self, prompt: str) -> str:
        """ Return a predicted continuation of `prompt`."""
        raise NotImplementedError('Must provide a language model.')

    @abstractmethod
    def draw_samples(self, prompt: str) -> Collection[str]:
        """ Return multiple predicted continuations of `prompt`. """
        return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]



class Sampler:
    """ Node that samples program skeleton continuations and sends them for analysis. """
    _global_samples_nums: int = 1

    def __init__(
            self,
            database: buffer.ExperienceBuffer,
            evaluators: Sequence[evaluator.Evaluator],
            samples_per_prompt: int,
            meta_data: dict,
            config: config_lib.Config,
            max_sample_nums: int | None = None,
            llm_class: Type[LLM] = LLM,
    ):
        self._samples_per_prompt = samples_per_prompt
        self._database = database
        self._evaluators = evaluators
        self._meta_data = meta_data
        self._llm = llm_class(samples_per_prompt)
        self._max_sample_nums = max_sample_nums
        self.config = config
        self.__class__._global_samples_nums = 1

    
    def sample(self, **kwargs):
        """ Continuously gets prompts, samples programs, sends them for analysis. """
        while True:
            # stop the search process if hit global max sample nums
            if self._max_sample_nums//5 and self.__class__._global_samples_nums >= self._max_sample_nums//5:
                break
            
            prompt = self._database.get_prompt()
            reset_time = time.time()
            samples = self._llm.draw_samples(prompt.code,self.config)
            sample_time = (time.time() - reset_time) / self._samples_per_prompt

            # This loop can be executed in parallel on remote evaluator machines.
            for sample in samples:
                sample = "\n    import pandas as pd\n    import numpy as np\n" + sample
                self._global_sample_nums_plus_one()
                cur_global_sample_nums = self._get_global_sample_nums()
                chosen_evaluator: evaluator.Evaluator = np.random.choice(self._evaluators)
                chosen_evaluator.analyse(
                    sample,
                    prompt.island_id,
                    prompt.data_input,
                    prompt.data_output,
                    prompt.version_generated,
                    **kwargs,
                    global_sample_nums=cur_global_sample_nums,
                    sample_time=sample_time
                )

    def _get_global_sample_nums(self) -> int:
        return self.__class__._global_samples_nums

    def set_global_sample_nums(self, num):
        self.__class__._global_samples_nums = num

    def _global_sample_nums_plus_one(self):
        self.__class__._global_samples_nums += 1



def _extract_body(sample: str, config: config_lib.Config) -> str:
    """
    Extract the function body from a response sample, removing any preceding descriptions
    and the function signature. Preserves indentation.
    ------------------------------------------------------------------------------------------------------------------
    Input example:
    ```
    This is a description...
    def function_name(...):
        return ...
    Additional comments...
    ```
    ------------------------------------------------------------------------------------------------------------------
    Output example:
    ```
        return ...
    Additional comments...
    ```
    ------------------------------------------------------------------------------------------------------------------
    If no function definition is found, returns the original sample.
    """
    lines = sample.splitlines()
    func_body_lineno = 0
    find_def_declaration = False
    
    for lineno, line in enumerate(lines):
        # find the first 'def' program statement in the response
        if line[:3] == 'def':
            func_body_lineno = lineno
            find_def_declaration = True
            break
    
    if find_def_declaration:
        # for gpt APIs
        if config.use_api:
            code = ''
            for line in lines[func_body_lineno + 1:]:
                code += line + '\n'
        
        # for mixtral
        else:
            code = ''
            indent = '    '
            for line in lines[func_body_lineno + 1:]:
                if line[:4] != indent:
                    line = indent + line
                code += line + '\n'
        
        return code
    
    return sample



class LocalLLM(LLM):
    def __init__(self, samples_per_prompt: int, batch_inference: bool = True, trim=True) -> None:
        """
        Args:
            batch_inference: Use batch inference when sample equation program skeletons. The batch size equals to the samples_per_prompt.
        """
        super().__init__(samples_per_prompt)

        self._instruction_prompt = ("You are a helpful assistant tasked with discovering new features/ dropping less important features for the given prediction task. \
                             Complete the 'modify_features' function below, considering the physical meaning and relationships of inputs.\n\n")
        self._batch_inference = batch_inference
        self._trim = trim
        self._samples_per_prompt = samples_per_prompt
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._pretrained_model_path = 'meta-llama/Llama-3.1-8B-Instruct'
        self._print_prompt = True

        # Load local model + tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self._pretrained_model_path,
            token=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        compute_dtype = getattr(torch, "bfloat16")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self._pretrained_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            token=True
        )

    def draw_samples(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        """Returns multiple equation program skeleton hypotheses for the given `prompt`."""
        if config.use_api:
            return self._draw_samples_api(prompt, config)
        else:
            return self._draw_samples_local(prompt, config)


    def _draw_samples_local(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        # Instruction prefix
        prompt = '\n'.join([self._instruction_prompt, prompt])
        if self._print_prompt:
            print(prompt)
            self._print_prompt = False
    
        while True:
            try:
                all_samples = []
    
                # Tokenize once
                inputs = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(self._device)
    
                if self._batch_inference:
                    # Repeat prompt for batch inference
                    inputs = torch.vstack([inputs] * self._samples_per_prompt)
    
                    outputs = self.model.generate(
                        inputs,
                        max_new_tokens=512,
                        temperature=0.8,
                        do_sample=True,
                        top_k=30,
                        top_p=0.9,
                        num_return_sequences=1,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
    
                    for i in range(outputs.shape[0]):
                        text = self.tokenizer.decode(outputs[i, len(inputs[i]):], skip_special_tokens=True)
                        all_samples.append(text)
    
                else:
                    # Sequential requests
                    for _ in range(self._samples_per_prompt):
                        outputs = self.model.generate(
                            inputs,
                            max_new_tokens=512,
                            temperature=0.8,
                            do_sample=True,
                            top_k=30,
                            top_p=0.9,
                            num_return_sequences=1,
                            eos_token_id=self.tokenizer.eos_token_id,
                            pad_token_id=self.tokenizer.pad_token_id
                        )
                        text = self.tokenizer.decode(outputs[0, len(inputs[0]):], skip_special_tokens=True)
                        all_samples.append(text)
    
                # Optional trim
                if self._trim:
                    all_samples = [_extract_body(sample, config) for sample in all_samples]
    
                return all_samples
    
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                if torch.cuda.device_count() > 0:
                    torch.cuda.empty_cache()
                continue
            except Exception as e:
                print(f"Error during generation: {e}")
                continue


    def _draw_samples_api(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        all_samples = []
        prompt = '\n'.join([self._instruction_prompt, prompt])
        
        for _ in range(self._samples_per_prompt):
            while True:
                try:
                    conn = http.client.HTTPSConnection("api.openai.com")
                    payload = json.dumps({
                        "max_tokens": 512,
                        "model": config.api_model,
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    })
                    headers = {
                        'Authorization': f"Bearer {os.environ['API_KEY']}",
                        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                        'Content-Type': 'application/json'
                    }
                    conn.request("POST", "/v1/chat/completions", payload, headers)
                    res = conn.getresponse()
                    data = json.loads(res.read().decode("utf-8"))
                    response = data['choices'][0]['message']['content']
                    
                    if self._trim:
                        response = _extract_body(response, config)
                    
                    all_samples.append(response)
                    break

                except Exception:
                    continue
        
        return all_samples
    
    
    def _do_request(self, content: str) -> str:
        content = content.strip('\n').strip()
        # repeat the prompt for batch inference
        repeat_prompt: int = self._samples_per_prompt if self._batch_inference else 1
        
        data = {
            'prompt': content,
            'repeat_prompt': repeat_prompt,
            'params': {
                'do_sample': True,
                'temperature': None,
                'top_k': None,
                'top_p': None,
                'add_special_tokens': False,
                'skip_special_tokens': True,
            }
        }
        
        headers = {'Content-Type': 'application/json'}
        response = requests.post(self._url, data=json.dumps(data), headers=headers)
        
        if response.status_code == 200: #Server status code 200 indicates successful HTTP request! 
            response = response.json()["content"]
            
            return response if self._batch_inference else response[0]

