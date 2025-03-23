from src.ui import ChatbotUI

def main():
    print("Starting main()...")
    try:
        print("Initializing ChatbotUI...")
        chatbot = ChatbotUI()
        print("ChatbotUI initialized. Running UI...")
        chatbot.run()
        print("UI closed.")
    except Exception as e:
        print(f"Error in main(): {str(e)}")
        raise  # Re-raise the exception for full stack trace

if __name__ == "__main__":
    print("Script started.")
    main()
    print("Script finished.")