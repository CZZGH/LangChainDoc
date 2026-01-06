from env_utils import get_env_variable
from langchain_openai import ChatOpenAI

def create_llm(model_name: str = "deepseek-v3.2") -> ChatOpenAI:
    api_key = get_env_variable("BAILIAN_API_KEY")
    base_url = get_env_variable("BAILIAN_BASE_URL")
    llm = ChatOpenAI(
        model_name = model_name,
        api_key=api_key,
        base_url=base_url,
        max_tokens=1024,
        temperature=0.7,
    )
    return llm