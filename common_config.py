import os
import time

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI
from openai import RateLimitError
from dotenv import load_dotenv

load_dotenv(override=True)

api_key = os.getenv("LLM_API_KEY")

BASE_URL="https://monogpt.kr/api/monorouter/v1"

# MonoRouter는 분당 30회 제한이 있다. 여유를 두고 초당 0.4회(분당 24회)로 맞춘다.
# 모듈 전역으로 두어 llm_connect()로 만든 모든 LLM이 한도를 함께 쓴다.
_rate_limiter = InMemoryRateLimiter(requests_per_second=0.4, max_bucket_size=1)


def _retry_on_rate_limit(func, *args, wait=30, max_retries=6, **kwargs):
    # 429가 나면 wait초 기다린 뒤 다시 시도한다.
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except RateLimitError:
            if attempt == max_retries:
                raise
            print(f"[429] {wait}초 대기 후 재시도 ({attempt + 1}/{max_retries})")
            time.sleep(wait)


def llm_connect(
    model: str = "gpt-4.1",
    api_key: str = api_key,
    temperature: float = 0.0,
    max_tokens: int = 2048
):
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=BASE_URL,
        temperature=temperature,
        use_responses_api=False,  # base url로 할 때는 이부분 넣어야 함.(MonoRouter 사용)
        max_tokens=max_tokens,
        rate_limiter=_rate_limiter,
    )

from langchain_openai import OpenAIEmbeddings


class _RateLimitSafeEmbeddings(OpenAIEmbeddings):
    # 429(RateLimitError)가 나면 기다렸다가 다시 시도하는 임베딩 모델
    def embed_documents(self, texts, chunk_size=None, **kwargs):
        return _retry_on_rate_limit(super().embed_documents, texts, chunk_size, **kwargs)

    def embed_query(self, text, **kwargs):
        return _retry_on_rate_limit(super().embed_query, text, **kwargs)


def embedding_model():
    embedding_model = "text-embedding-3-small"
    embeddings = _RateLimitSafeEmbeddings(
        api_key=api_key,
        base_url=BASE_URL,
        # use_responses_api=False,  # base url로 할 때는 이부분 넣어야 함.(MonoRouter 사용)
        model=embedding_model
        )
    # print(embeddings)
    return embeddings