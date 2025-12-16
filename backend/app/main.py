from src.agent.finance_agent import finance_agent

print("🤖 Finance AI Agent (CLI)")
print("Type 'exit' to quit")

while True:
    q = input("You: ")
    if q.lower() in {"exit", "quit"}:
        break
    print("\nAgent:\n", finance_agent(q))
    print("-" * 80)

