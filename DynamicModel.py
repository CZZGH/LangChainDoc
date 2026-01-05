'''
To use a dynamic model, create middleware using the @wrap_model_call 
decorator that modifies the model in the request:
'''
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call,ModelRequest, ModelResponse
from my_llm import create_llm   

basic_model = create_llm("deepseek-v3.2")
advanced_model = create_llm("deepseek-r1-0528") 

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """Choose model based on conversation complexity."""
    message_count = len(request.state["messages"])
    print(f"Message count: {message_count}")
    if message_count > 1:
        # Use an advanced model for longer conversations
        model = advanced_model
    else:
        model = basic_model

    return handler(request.override(model=model))

agent = create_agent(
    model=basic_model,  # Default model
    tools=[],
    middleware=[dynamic_model_selection]
)

if __name__ == "__main__":
    # Example usage
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "Tell me a joke."}]}
    )
    print(response)

    # Continue conversation with the same thread_id
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "What is the weather like?"},{"role": "user", "content": "What is the location?"}]}
    )
    print(response)


'''
output:

Message count: 1
{'messages': [HumanMessage(content='Tell me a joke.', additional_kwargs={}, response_metadata={}, id='328cd22d-3b49-4a3b-a9ea-3d9ec64a8091'), AIMessage(content="Why don't scientists trust atoms?  \n\nBecause they make up *everything*!", additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 16, 'prompt_tokens': 9, 'total_tokens': 25, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_provider': 'openai', 'model_name': 'deepseek-v3.2', 'system_fingerprint': None, 'id': 'chatcmpl-f2b2fcf5-db60-9cb8-8e0a-e196361f9502', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019b87b0-6f12-79e3-8787-27b0fd30418c-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 9, 'output_tokens': 16, 'total_tokens': 25, 'input_token_details': {}, 'output_token_details': {}})]}
Message count: 2
{'messages': [HumanMessage(content='What is the weather like?', additional_kwargs={}, response_metadata={}, id='9f12c77b-4f4e-45ca-aef6-bc5dbae1f903'), HumanMessage(content='What is the location?', additional_kwargs={}, response_metadata={}, id='914f74a8-f48a-4841-b801-ac7dea871f10'), AIMessage(content='To provide you with the current weather, I need to know the location! 🌤️ Please tell me the city, region, or country you\'re interested in. For example, you could say:\n\n- "What\'s the wea 
ther like in New York?"\n- "How\'s the weather in Tokyo?"\n- "Is it raining in London?"\n\nLet me know, and I\'ll get the details for you! 😊', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 255, 'prompt_tokens': 14, 'total_tokens': 269, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_provider': 'openai', 'model_name': 'deepseek-r1-0528', 'system_fingerprint': None, 'id': 'chatcmpl-cda88507-10f5-96db-bbc3-da72d0e17827', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019b87b0-742e-7ec2-a795-7699dc2604e3-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 14, 'output_tokens': 255, 'total_tokens': 269, 'input_token_details': {}, 'output_token_details': {}})]}
'''