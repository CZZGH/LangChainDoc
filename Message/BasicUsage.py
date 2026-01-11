from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from my_llm import create_llm

model = create_llm("deepseek-v3.2")


system_msg = SystemMessage("You are a helpful assistant.")
human_msg = HumanMessage("Hello, how are you?")

#1 Use with chat models
messages = [system_msg, human_msg]
# response = model.invoke(messages)  # Returns AIMessage


#2 Text prompts are strings
# response = model.invoke("Write a haiku about spring")


#3 list of message objects
messages = [
    SystemMessage("You are a poetry expert"),
    HumanMessage("Write a haiku about spring"),
    AIMessage("Cherry blossoms bloom..."),
]
#response = model.invoke(messages)



#4 Dictionary format
messages = [
    {"role": "system", "content": "You are a poetry expert"},
    {"role": "user", "content": "Write a haiku about spring"},
    {"role": "assistant", "content": "Cherry blossoms bloom..."}
]
response = model.invoke(messages)



print(response.content)
