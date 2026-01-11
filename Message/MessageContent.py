from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from my_llm import create_llm
from my_tools import image_to_base64

# deepseek-v3.2 不支持多模态输入
model = create_llm("qwen3-vl-plus")

# String content
human_message = HumanMessage("Hello, how are you?")

# Provider-native format (e.g., OpenAI)
human_message = HumanMessage(content=[
    {"type": "text", "text": "Hello, how are you?"},
    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
])

# List of standard content blocks
human_message = HumanMessage(content_blocks=[
    {"type": "text", "text": "Hello, how are you?"},
    {"type": "image", "url": "https://example.com/image.jpg"},
])

'''
聊天模型可以接受多模态数据作为输入，并将其作为输出生成。
下面我们展示一些包含多模态数据的输入消息的简短示例。
'''
# ------------ image input
# From URL
message = [ 
            HumanMessage(content=[
                {
                    "type": "image_url",
                    "image_url": {"url": "https://www.northnews.cn/a/10001/202410/260413beb196581ee283cf9a1e572b00.png"},
                },
                {
                    "type": "text",
                    "text": "请描述图中内容"
                },
            ])
        ]

#

str_base64 = image_to_base64(r"C:\\Users\\navinfo-ai-cv\\Desktop\\road.png")

# From base64 data
message = [{
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the content of this image."},
        {
            "type": "image",
            "base64": str_base64,
            "mime_type": "image/jpeg",
        },
    ]
}]


# From provider-managed File ID
# message = {
#     "role": "user",
#     "content": [
#         {"type": "text", "text": "Describe the content of this image."},
#         {"type": "image", "file_id": "file-abc123"},
#     ]
# }


response = model.invoke(message)
print(response.content)