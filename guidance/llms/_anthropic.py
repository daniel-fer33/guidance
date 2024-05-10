import random

import aiohttp
import anthropic
from anthropic import AsyncAnthropic, AsyncStream
import os
import copy
import time
import asyncio
import types
import collections
import json
import re
import regex
import logging

from ._llm import LLM, LLMSession, SyncSession

log = logging.getLogger(__name__)

import pyparsing as pp

role_start_tag = pp.Suppress(pp.Optional(pp.White()) + pp.Literal("<|im_start|>"))
role_start_name = pp.Word(pp.alphanums + "_")("role_name")
role_kwargs = pp.Suppress(pp.Optional(" ")) + pp.Dict(pp.Group(pp.Word(pp.alphanums + "_") + pp.Suppress("=") + pp.QuotedString('"')))("kwargs")
role_start = (role_start_tag + role_start_name + pp.Optional(role_kwargs) + pp.Suppress("\n")).leave_whitespace()
role_end = pp.Suppress(pp.Literal("<|im_end|>"))
role_content = pp.Combine(pp.ZeroOrMore(pp.CharsNotIn("<") | pp.Literal("<") + ~pp.FollowedBy("|im_end|>")))("role_content")
role_group = pp.Group(role_start + role_content + role_end)("role_group").leave_whitespace()
partial_role_group = pp.Group(role_start + role_content)("role_group").leave_whitespace()
roles_grammar = pp.ZeroOrMore(role_group) + pp.Optional(partial_role_group) + pp.StringEnd()


def prompt_to_messages(prompt):
    messages = []

    assert prompt.endswith("<|im_start|>assistant\n"), "When calling OpenAI chat models you must generate only directly inside the assistant role! The OpenAI API does not currently support partial assistant prompting."

    parsed_prompt = roles_grammar.parse_string(prompt)

    # pattern = r'<\|im_start\|>([^\n]+)\n(.*?)(?=<\|im_end\|>|$)'
    # matches = re.findall(pattern, prompt, re.DOTALL)

    # if not matches:
    #     return [{'role': 'user', 'content': prompt}]

    for role in parsed_prompt:
        if len(role["role_content"]) > 0: # only add non-empty messages (OpenAI does not support empty messages anyway)
            message = {'role': role["role_name"], 'content': role["role_content"]}
            if "kwargs" in role:
                for k, v in role["kwargs"].items():
                    message[k] = v
            messages.append(message)

    return messages


def anthropic_messages_response_to_opeanai_completion_dict(messages_response):
    out = messages_response.dict()
    if 'message' in out:
        # Stream
        message_type = out['type']
        out = out['message']
        out['type'] = message_type

    finish_reason = {
        'end_turn': 'stop',
        'max_tokens': 'length',
        'stop_sequence': 'stop'
    }.get(out.get('stop_reason', None), None)
    if 'content' in out:
        out['choices'] = [
            dict(
                finish_reason=finish_reason,
                index=i,
                message=dict(
                    content=s1['text'],
                    role=out['role'],
                    function_call=None,  # TODO: Implement
                    tool_calls=None  # TODO: Implement
                ),
                logprobs=None
            ) for i, s1 in enumerate(out['content'])
        ]

    # Remove keys
    remove_keys = ['content', 'role', 'stop_reason', 'stop_sequence']
    for k in remove_keys:
        if k in out:
            del out[k]
    return out


def add_text_to_chat_mode(chat_mode):
    if isinstance(chat_mode, (types.AsyncGeneratorType, types.GeneratorType, AsyncStream)):
        return add_text_to_chat_mode_generator(chat_mode)
    else:
        chat_mode = anthropic_messages_response_to_opeanai_completion_dict(chat_mode)
        for c in chat_mode['choices']:
            c['text'] = c['message']['content']
        return chat_mode


class Anthropic(LLM):
    llm_name: str = "anthropic"

    def __init__(self, model=None, caching=True, max_retries=5, max_calls_per_min=60,
                 api_key=None, api_type="anthropic", api_base=None, api_version=None, deployment_id=None,
                 temperature=0.0, rest_call=False,
                 allowed_special_tokens=None):
        super().__init__()

        # fill in default model value
        if model is None:
            model = os.environ.get("ANTHROPIC_MODEL", None)
        if model is None:
            try:
                with open(os.path.expanduser('~/.anthropic_model'), 'r') as file:
                    model = file.read().replace('\n', '')
            except:
                pass

        # fill in default API key value
        if api_key is None:  # get from environment variable
            api_key = os.environ.get("ANTHROPIC_API_KEY", None)
        if api_key is not None and not api_key.startswith("sk-ant-") and os.path.exists(
                os.path.expanduser(api_key)):  # get from file
            with open(os.path.expanduser(api_key), 'r') as file:
                api_key = file.read().replace('\n', '')
        if api_key is None:  # get from default file location
            try:
                with open(os.path.expanduser('~/.anthropic_api_key'), 'r') as file:
                    api_key = file.read().replace('\n', '')
            except:
                pass

        self._tokenizer = anthropic.Client().get_tokenizer()
        self.chat_mode = True  # Only Anthropic chat-mode is currently supported

        self.allowed_special_tokens = allowed_special_tokens
        self.model_name = model
        self.caching = caching
        self.max_retries = max_retries
        self.max_calls_per_min = max_calls_per_min
        if isinstance(api_key, str):
            api_key = api_key.replace("Bearer ", "")
        self.api_key = api_key
        self.api_type = api_type
        self.api_base = api_base
        self.api_version = api_version
        self.current_time = time.time()
        self.call_history = collections.deque()
        self.temperature = temperature
        self.rest_call = rest_call

        if not self.rest_call:
            self.caller = self._library_call
        else:
            self.caller = self._rest_call
            self._rest_headers = {
                "Content-Type": "application/json"
            }

    def session(self, asynchronous=False):
        if asynchronous:
            return AnthropicSession(self)
        else:
            return SyncSession(AnthropicSession(self))

    def role_start(self, role_name, **kwargs):
        assert self.chat_mode, "role_start() can only be used in chat mode"
        return "<|im_start|>" + role_name + "".join([f' {k}="{v}"' for k, v in kwargs.items()]) + "\n"

    def role_end(self, role=None):
        assert self.chat_mode, "role_end() can only be used in chat mode"
        return "<|im_end|>"

    def end_of_text(self):
        return "<|endoftext|>"

    # Define a function to add a call to the deque
    def add_call(self):
        # Get the current timestamp in seconds
        now = time.time()
        # Append the timestamp to the right of the deque
        self.call_history.append(now)

    # Define a function to count the calls in the last 60 seconds
    def count_calls(self):
        # Get the current timestamp in seconds
        now = time.time()
        # Remove the timestamps that are older than 60 seconds from the left of the deque
        while self.call_history and self.call_history[0] < now - 60:
            self.call_history.popleft()
        # Return the length of the deque as the number of calls
        return len(self.call_history)

    async def _library_call(self, **kwargs):
        """ Call the Anthropic API using the python package."""
        assert self.api_key is not None, "You must provide an Anthropic API key to use the Anthropic LLM. " \
                                         "Either pass it in the constructor, set the ANTHROPIC_API_KEY environment " \
                                         "variable, or create the file ~/.anthropic_api_key with your key in it."

        # Start API client
        client = AsyncAnthropic(api_key=self.api_key)

        # Get messages/system from prompt
        messages = prompt_to_messages(kwargs['prompt'])
        system_msgs = [s1['content'] for s1 in messages if s1['role'] == 'system']
        kwargs['system'] = system_msgs[-1] if len(system_msgs) > 0 else None
        kwargs['messages'] = [s1 for s1 in messages if s1['role'] != 'system']

        # Remove unused arguments
        del kwargs['prompt']
        del kwargs['stop_sequences']  # TODO fix error

        # Call LLM API
        out = await client.messages.create(**kwargs)
        log.info(f"LLM call response: {out}")
        out = add_text_to_chat_mode(out)

        return out

    async def _rest_call(self, **kwargs):
        raise NotImplementedError

    async def _rest_stream_handler(self, responses):
        raise NotImplementedError

    def encode(self, string, **kwargs):
        # note that is_fragment is not used for this tokenizer
        return self._tokenizer.encode(string, allowed_special=self.allowed_special_tokens, **kwargs)

    def decode(self, tokens, **kwargs):
        return self._tokenizer.decode(tokens, **kwargs)


# Define a deque to store the timestamps of the calls
class AnthropicSession(LLMSession):
    async def __call__(self, prompt, stop=None, stop_regex=None, temperature=None, n=1, max_tokens=1000, logprobs=None,
                       top_p=1.0, top_k=None, echo=False, logit_bias=None, token_healing=None, pattern=None, stream=None,
                       cache_seed=0, caching=None, function_call=None, **completion_kwargs):
        """ Generate a completion of the given prompt.
        """

        # we need to stream in order to support stop_regex
        if stream is None:
            stream = stop_regex is not None
        assert stop_regex is None or stream, "We can only support stop_regex for the Anthropic API when stream=True!"
        assert stop_regex is None or n == 1, "We don't yet support stop_regex combined with n > 1 with the Anthropic API!"

        assert token_healing is None or token_healing is False, "The Anthropic API does not yet support token healing! Please either switch to an endpoint that does, or don't use the `token_healing` argument to `gen`."

        # set defaults
        if temperature is None:
            temperature = self.llm.temperature

        # get the arguments as dictionary for cache key generation
        args = locals().copy()

        assert not pattern, "The Anthropic API does not support Guidance pattern controls! " \
                            "Please either switch to an endpoint that does, or don't use the `pattern` " \
                            "argument to `gen`."
        # assert not stop_regex, "The OpenAI API does not support Guidance stop_regex controls! Please either switch to an endpoint that does, or don't use the `stop_regex` argument to `gen`."

        # define the key for the cache
        cache_params = self._cache_params(args)
        llm_cache = self.llm.cache
        key = llm_cache.create_key(self.llm.llm_name, **cache_params)

        # allow streaming to use non-streaming cache (the reverse is not true)
        if key not in llm_cache and stream:
            cache_params["stream"] = False
            key1 = llm_cache.create_key(self.llm.llm_name, **cache_params)
            if key1 in llm_cache:
                key = key1

        # check the cache
        if key not in llm_cache or caching is False or (caching is not True and not self.llm.caching):

            # ensure we don't exceed the rate limit
            while self.llm.count_calls() > self.llm.max_calls_per_min:
                await asyncio.sleep(1)

            # TODO: Add tools
            # tools = extract_tools_defs(prompt)

            fail_count = 0
            while True:
                try_again = False
                try:
                    self.llm.add_call()
                    call_args = {
                        "model": self.llm.model_name,
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "stop_sequences": stop,
                        "stream": stream,
                        "temperature": temperature,
                        #"tools": tools,
                        "top_p": top_p,
                        "top_k": top_k,
                        **completion_kwargs
                    }

                    call_args = {k: v for k, v in call_args.items() if v is not None}
                    out = await self.llm.caller(**call_args)

                except (anthropic.RateLimitError,
                        anthropic.APIConnectionError,
                        anthropic.APIStatusError,
                        #anthropic.APIError,
                        anthropic.APITimeoutError) as e:
                    await asyncio.sleep(3)
                    try_again = True
                    fail_count += 1

                if not try_again:
                    break

                if fail_count > self.llm.max_retries:
                    raise Exception(
                        f"Too many (more than {self.llm.max_retries}) Anthropic API errors in a row!")

            if stream:
                #return self.llm.stream_then_save(out, key, stop_regex, n)
                raise NotImplementedError
            else:
                llm_cache[key] = out

        # wrap as a list if needed
        if stream:
            if isinstance(llm_cache[key], list):
                return llm_cache[key]
            return [llm_cache[key]]

        return llm_cache[key]
