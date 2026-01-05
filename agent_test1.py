from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from env_utils import get_env_variable


from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime

from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy

checkpointer = InMemorySaver()

SYSTEM_PROMPT = """You are an expert weather forecaster, who speaks in puns.

You have access to two tools:

- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location

If a user asks you for the weather, make sure you know the location. If you can tell from the question that they mean wherever they are, use the get_user_location tool to find their location."""

@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Retrieve user information based on user ID."""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"

@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    # A punny response (always required)
    punny_response: str
    # Any interesting information about the weather if available
    weather_conditions: str | None = None


# def get_weather(city: str) -> str:
#     """获取指定城市的天气信息。
#     Args:
#         city (str): 城市名称，例如 "北京"、"上海"。

#     Returns:
#         str: 格式化的天气信息，例如 "北京：晴，25℃"。
#     """
#     # This is a mock implementation. Replace with actual weather fetching logic.
#     return f"The current weather in {city} is sunny with a temperature of 25°C."

def create_llm(model_name: str = "deepseek-v3.2") -> ChatOpenAI:
    api_key = get_env_variable("BAILIAN_API_KEY")
    base_url = get_env_variable("BAILIAN_BASE_URL")
    llm = ChatOpenAI(
        model_name = model_name,
        api_key=api_key,
        base_url=base_url,
    )
    return llm

def create_agent_instance():
    llm = create_llm()
    agent = create_agent(
        model = llm,
        tools = [get_user_location,get_weather_for_location], 
        system_prompt=SYSTEM_PROMPT,
        context_schema=Context,
        response_format=ToolStrategy(ResponseFormat),
        checkpointer=checkpointer
    )

    return agent
if __name__ == "__main__":
    # Run agent
    # `thread_id` is a unique identifier for a given conversation.
    config = {"configurable": {"thread_id": "1"}}
    agent = create_agent_instance()
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "what is the weather outside?"}]},
        config=config,
        context=Context(user_id="1")
    )
    print(response['structured_response'])

    # Note that we can continue the conversation using the same `thread_id`.
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "thank you!"}]},
        config=config,
        context=Context(user_id="1")
    )

    print(response['structured_response'])