from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from my_llm import create_llm

llm = create_llm("deepseek-v3.2")

class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str

agent = create_agent(
    model=llm,
    tools=[],
    response_format=ToolStrategy(ContactInfo)
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

print(result["structured_response"])

'''
output:name='John Doe' email='john@example.com' phone='(555) 123-4567'
'''

# ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')