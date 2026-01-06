from my_llm import create_llm

llm = create_llm("deepseek-v3.2")

# invoke the agent
# response = llm.invoke("Why do parrots have colorful feathers?")
# print(response)

# conversation = [
#     {"role": "system", "content": "You are a helpful assistant that translates English to French."},
#     {"role": "user", "content": "Translate: I love programming."},
#     {"role": "assistant", "content": "J'adore la programmation."},
#     {"role": "user", "content": "Translate: I love building applications."}
# ]
# response = llm.invoke(conversation)
# print(response)

# from langchain.messages import HumanMessage, AIMessage, SystemMessage

# conversation = [
#     SystemMessage("You are a helpful assistant that translates English to French."),
#     HumanMessage("Translate: I love programming."),
#     AIMessage("J'adore la programmation."),
#     HumanMessage("Translate: I love building applications.")
# ]

# response = llm.invoke(conversation)
# print(response)  # AIMessage("J'adore créer des applications.")


# batch output
responses = llm.batch([
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?"
])

for response in llm.batch_as_completed([
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?"
]):
    print(response)
