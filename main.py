
import os
import re
import json
import pickle
import random
import logging
import asyncio
import time
from pathlib import Path
from base64 import b64decode
from functools import wraps
from statistics import (
    median,
    mean
)
from collections import (
    deque,
    defaultdict,
    Counter
)

from dotenv import load_dotenv

import aiohttp

import tiktoken

import numpy as np
import pandas as pd

import umap
from sklearn.cluster import (
    DBSCAN,
    KMeans
)

from bokeh.plotting import (
    figure,
    show
)
from bokeh.io import output_notebook
from bokeh.models import (
    HoverTool,
    ColumnDataSource
)
from bokeh.palettes import (
    Category10,
    Category20
)

from IPython.display import (
    HTML,
    display
)


######
#
#  LOGGING
#
####


logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger("main")
logger.setLevel(logging.DEBUG)


#######
#
#  JSON
#
####


def json_dump(data, path):
    with open(path, "w") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def json_load(path):
    with open(path) as file:
        return json.load(file)


######
#
#   PICKLE
#
#####


def pickle_dump(obj, path):
    with open(path, 'wb') as file:
        pickle.dump(obj, file)


def pickle_load(path):
    with open(path, 'rb') as file:
        return pickle.load(file)


######
#
#  TIKTOKEN
#
#######


def tiktoken_count(text, model):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


#######
#
#   PROMPT
#
######


def prompt_lmsys_hardness(instruction):
    messages = [
        {
            "role": "system",
            "content": """Your task is to evaluate how well the following input prompt can assess the capabilities of advanced AI assistants.

For the input prompt, please analyze it based on the following 7 criteria.
1. Specificity: Does the prompt ask for a specific output, such as code, a mathematical solution, a logical simplification, a problem-solving strategy, or a hardware setup recommendation? This specificity allows the AI to demonstrate its ability to understand and generate precise responses.
2. Domain Knowledge: Does the prompt cover a specific domain, such as programming, mathematics, logic, problem-solving, or hardware setup? Prompts spanning a range of topics test the AI's breadth of knowledge and its ability to apply that knowledge to different domains.
3. Complexity: Does the prompt vary in complexity, from straightforward tasks to more complex, multi-step problems? This allows evaluators to assess the AI's capability to handle problems of varying difficulty.
4. Problem-Solving Skills: Does the prompt directly involves the AI to demonstrate active problem-solving skills, such systemically coming up with a solution for a specific setup instead of regurgitating an existing fact? This tests the AI's ability to apply logical reasoning and provide practical solutions.
5. Creativity: Does the prompt involve a level of creativity in approaching the problem? This criterion tests the AI's ability to provide tailored solutions that take into account the user's specific needs and limitations.
6. Technical Accuracy: Does the prompt require technical accuracy in the response? This allows evaluators to assess the AI's precision and correctness in technical fields.
7. Real-world Application: Does the prompt relate to real-world applications, such as setting up a functional system or writing code for a practical use case? This tests the AI's ability to provide practical and actionable information that could be implemented in real-life scenarios.

You must list the criteria numbers that the prompt satisfies, separate numbers with comma. Do not explain your choice."""
        },
        {
            "role": "user",
            "content": instruction
        }
    ]
    mapping = {
        1: "specificity",
        2: "domain_knowledge",
        3: "complexity",
        4: "problem_solving",
        5: "creativity",
        6: "technical_accuracy",
        7: "real_world",
    }
    return messages, mapping


def prompt_wildchat_classify(instruction):
    cats = [
        "Information seeking",
        "Reasoning",
        "Planning",
        "Editing",
        "Coding & Debugging",
        "Math",
        "Role playing",
        "Data Analysis",
        "Creative Writing",
        "Advice seeking",
        "Brainstorming",
        "Others",
    ]
    messages = [
        {
            "role": "system",
            "content": """Your task is to classify the following input prompt into one of given categories.

- Information seeking - Users ask for specific information or facts about various topics.
- Reasoning - Queries require logical thinking, problem-solving, or processing of complex ideas.
- Planning - Users need assistance in creating plans or strategies for activities and projects.
- Editing - Involves editing, rephrasing, proofreading, or other tasks related to the composition of general written content.
- Coding & Debugging - Users seek help with writing, reviewing, or fixing code in programming.
- Math - Queries related to mathematical concepts, problems, and calculations.
- Role playing - Users engage in scenarios requiring ChatGPT to adopt a character or persona.
- Data Analysis - Requests involve interpreting data, statistics, or performing analytical tasks.
- Creative Writing - Users seek assistance with crafting stories, poems, or other creative texts.
- Advice seeking - Users ask for recommendations or guidance on various personal or professional issues.
- Brainstorming - Involves generating ideas, creative thinking, or exploring possibilities.
- Others - Any queries that do not fit into the above categories or are of a miscellaneous nature.

You must output category name that fits best. Do not explain your choice.""",
        },
        {
            "role": "user",
            "content": instruction,
        }
    ]
    return messages, cats


#####
#
#  COMMON OPENROUTER / OPENAI
#
####


class RateLimiter:
    def __init__(self, max_requests=1, time_window=1):
        self.max_requests = max_requests
        self.time_window = time_window
        self.request_times = deque()

    async def wait(self):
        while len(self.request_times) >= self.max_requests:
            oldest_time = self.request_times[0]
            time_delta = time.monotonic() - oldest_time

            if time_delta >= self.time_window:
                self.request_times.popleft()
            else:
                await asyncio.sleep(self.time_window - time_delta)

        self.request_times.append(time.monotonic())


class TokensRateLimiter:
    def __init__(self, max_tokens=1, max_requests=1, time_window=1):
        self.max_tokens = max_tokens
        self.max_requests = max_requests
        self.time_window = time_window
        self.request_times = deque()
        self.request_tokens = deque()
        self.window_tokens = 0

    async def wait(self, tokens=1):
        assert tokens <= self.max_tokens, tokens
        while (
                self.window_tokens + tokens > self.max_tokens
                or len(self.request_times) + 1 > self.max_requests
        ):
            oldest_time = self.request_times[0]
            time_delta = time.monotonic() - oldest_time
            
            if time_delta >= self.time_window:
                self.request_times.popleft()
                self.window_tokens -= self.request_tokens.popleft()
            else:
                await asyncio.sleep(self.time_window - time_delta)

        self.request_times.append(time.monotonic())
        self.request_tokens.append(tokens)
        self.window_tokens += tokens


def retrying(method):
    @wraps(method)
    async def wrapper(*args, **kwargs):
        retry = 0
        while True:
            try:
                return await method(*args, **kwargs)
            except (aiohttp.ClientError, TimeoutError) as error:
                retry += 1
                base_delay, max_delay, jitter = 1, 120, random.random()
                delay = min(base_delay * (2 ** (retry - 1) + jitter), max_delay)
                logger.warning(
                    'retry=%d delay=%.1f error="%s"',
                    retry, delay,
                    str(error) or error.__class__.__name__
                )
                await asyncio.sleep(delay)

    return wrapper


#######
#
#   OPENROUTER
#
#######


class OpenrouterClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.session = aiohttp.ClientSession()

        self.rate_limiter = RateLimiter()
        self.model_pricing = {}

    async def proc_response(self, response):
        response.raise_for_status()
        data = await response.json()

        if not data:
            # literally "null" returned by Deepseek
            raise aiohttp.ClientError("empty json")

        # {'error': {'message': 'Internal Server Error', 'code': 500},
        # 'user_id': 'user_2lxWCedeJ9IINL0t1I11HmmWnYo'}
        if "error" in data:
            raise aiohttp.ClientError(data["error"]["message"])

        return data

    @retrying
    async def post_request(self, url, data, timeout=None):
        await self.rate_limiter.wait()
        response = await self.session.post(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json=data,
            timeout=aiohttp.ClientTimeout(total=timeout)
        )
        return await self.proc_response(response)

    @retrying
    async def get_request(self, url, timeout=5):
        await self.rate_limiter.wait()
        response = await self.session.get(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
            },
            timeout=aiohttp.ClientTimeout(total=timeout)
        )
        return await self.proc_response(response)

    async def chat_completions(self, **kwargs):
        return await self.post_request(
            url="https://openrouter.ai/api/v1/chat/completions",
            data=kwargs,
        )

        # {"id": "gen-1733991024-UnDSgks1KnkCAbFjED5P",
        #  "provider": "OpenAI",
        #  "model": "openai/gpt-4o-mini-2024-07-18",
        #  "object": "chat.completion",
        #  "created": 1733991024,
        #  "choices": [{"logprobs": None,

        #    "finish_reason": "stop",
        #    "index": 0,
        #    "message": {"role": "assistant", "content": "[Б]", "refusal": ""}}],
        #  "system_fingerprint": "fp_bba3c8e70b",
        #  "usage": {"prompt_tokens": 1450,
        #   "completion_tokens": 3,
        #   "total_tokens": 1453}}

    async def auth_key(self):
        response = await self.get_request("https://openrouter.ai/api/v1/auth/key")
        return response["data"]

        # {"label": "sk-or-v1-446...596",
        #  "limit": None,
        #  "usage": 0.926944265,
        #  "limit_remaining": None,
        #  "is_free_tier": False,
        #  "rate_limit": {"requests": 40, "interval": "10s"}}}

    async def models(self):
        response = await self.get_request("https://openrouter.ai/api/v1/models")
        return response["data"]

        # [{"id": "liquid/lfm-7b",
        #   "name": "Liquid: LFM 7B",
        #   "created": 1737806883,
        #   "description": "LFM-7B, a new best-in .. for benchmarks and more info.",
        #   "context_length": 32768,
        #   "architecture": {"modality": "text->text",
        #    "tokenizer": "Other",
        #    "instruct_type": "vicuna"},
        #   "pricing": {"prompt": "0.00000001",
        #    "completion": "0.00000001",
        #    "image": "0",
        #    "request": "0"},
        #   "top_provider": {"context_length": 32768,
        #    "max_completion_tokens": None,
        #    "is_moderated": False},
        #   "per_request_limits": None},
        #  {"id": "liquid/lfm-3b",
        #   "name": "Liquid: LFM 3B"
        #  ...

    async def update_rate_limit(self):
        response = await self.auth_key()
        max_requests = response["rate_limit"]["requests"]
        time_window = response["rate_limit"]["interval"]
        match = re.match(r"^(\d+)s$", time_window)  # 10s
        assert match, time_window
        time_window = int(match.group(1))

        self.rate_limiter.max_requests = max_requests
        self.rate_limiter.time_window = time_window

    async def update_model_pricing(self):
        items = await self.models()
        for item in items:
            pricing = item["pricing"]
            pricing["prompt"] = float(pricing["prompt"])
            pricing["completion"] = float(pricing["completion"])
            self.model_pricing[item["id"]] = pricing

    def usage_cost(self, model, usage):
        pricing = self.model_pricing[model]
        return (
            usage["prompt_tokens"] * pricing["prompt"]
            + usage["completion_tokens"] * pricing["completion"]
        )

    async def __call__(self, model, messages, timeout=120, **kwargs):
        response = await self.chat_completions(
            model=model,
            messages=messages,
            timeout=timeout,
            **kwargs
        )
        response["cost"] = self.usage_cost(model, response["usage"])
        return response


######
#
#  OPENAI
#
######


class OpenaiEmbClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.session = aiohttp.ClientSession()

        # https://platform.openai.com/docs/models/text-embedding-3-large
        # tier4
        self.tokens_rate_limiter = TokensRateLimiter(
            max_requests=10_000,
            max_tokens=5_000_000,
            time_window=60
        )

    @retrying
    async def embeddings(
            self,
            input,
            input_tokens,
            model="text-embedding-3-large",
            dimensions=1024,
            encoding_format="base64"
    ):
        await self.tokens_rate_limiter.wait(input_tokens)
        response = await self.session.post(
            "https://api.openai.com/v1/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "input": input,
                "model": model,
                "dimensions": dimensions,
                "encoding_format": encoding_format
            },
            timeout=aiohttp.ClientTimeout(total=120)
        )

        if response.status != 200:
            text = await response.text()
            raise aiohttp.ClientError(text or response.status)

        return await response.json()

    # {'object': 'list',
    #  'data': [{'object': 'embedding',
    #    'index': 0,
    #    'embedding': 'y4t/...88gkcvQ=='}],
    #  'model': 'text-embedding-3-large',
    #  'usage': {'prompt_tokens': 1, 'total_tokens': 1}}


#######
#
#   UMAP
#
###


def umap_xy_proj(emb_items):
    X = np.vstack([
        _['array'] for _ in emb_items
    ])
    model = umap.UMAP(
        n_neighbors=15,
        n_components=2,
        metric='cosine',
    )
    Y = model.fit_transform(X)

    for index, (x, y) in enumerate(Y.tolist()):
        yield {
            'id': emb_items[index]['id'],
            'x': x,
            'y': y
        }


def umap_cluster_input_proj(emb_items):
    ids = [_['id'] for _ in emb_items]
    X = np.vstack([_['array'] for _ in emb_items])

    # params from
    # https://github.com/MaartenGr/BERTopic/blob/127e794f5630bc0d48071f012b07e9e41dd7d8ba/bertopic/_bertopic.py#L239
    model = umap.UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric='cosine',
    )
    return {
        'ids': ids,
        'X': model.fit_transform(X)
    }


#####
#
#   DBSCAN
#
#####


def dbscan_cluster(input, eps=.5, min_samples=5, metric="euclidean"):
    model = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric=metric
    ).fit(input["X"])

    ids = input["ids"]
    cluster_ids = model.labels_.tolist()
    for id, cluster_id in zip(ids, cluster_ids):
        yield {
            "id": id,
            "cluster_id": cluster_id,
        }


def kmeans_cluster(input, n_clusters=100):
    model = KMeans(
        n_clusters=n_clusters,
    ).fit(input["X"])

    ids = input["ids"]
    cluster_ids = model.labels_.tolist()
    for id, cluster_id in zip(ids, cluster_ids):
        yield {
            "id": id,
            "cluster_id": cluster_id,
        }


#####
#
#   SHOW
#
######


def show_html(content):
    display(HTML(content))


def show_h3(text):
    show_html(f"<h3>{text}</h3>")


def show_hr():
    show_html("<hr/>")
