print("🤖 Finance Multi-Agent AI")
print("Ask me about any stock (price / analysis/ news/ sentiment/ technical).")
print("Type 'exit' to end the conversation.\n")

while True:
    try:
        user_input = input("You: ")

        if user_input.lower().strip() in ["exit", "quit", "q"]:
            print("\nAgent: Session ended. 👋")
            break

        response = finance_agent(user_input)

        print("\nAgent:")
        print(response)
        print("\n" + "-"*80 + "\n")

    except KeyboardInterrupt:
        print("\nAgent: Session interrupted.")
        break

