import sys
from langchain.agents import create_agent
from common_config import llm_connect

sys.path.append("..")

def get_weather(city: str) -> str:
    """Get weather for a given city"""
    return f"It's always sunny in {city}!"

model=llm_connect("gpt-4.1")  # or any other model you want to use

agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful assistant that provides weather information."
)

# Run the agent
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather like in New York?"}]}
)

print(result)