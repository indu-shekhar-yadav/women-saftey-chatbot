import tkinter as tk
from src.nlp_model import NLPModel
from src.rag import RAG
from src.speech_to_text import SpeechToText

class ChatbotUI:
    def __init__(self):
        print("Initializing NLPModel...")
        self.nlp = NLPModel()
        print("NLPModel initialized.")

        print("Initializing RAG...")
        self.rag = RAG()
        print("RAG initialized.")

        print("Initializing SpeechToText...")
        try:
            self.stt = SpeechToText()
            print("SpeechToText initialized.")
        except Exception as e:
            print(f"Failed to initialize SpeechToText: {str(e)}. Proceeding without voice input.")
            self.stt = None

        print("Creating Tkinter UI...")
        self.root = tk.Tk()
        self.root.title("Women Safety Chatbot")
        self.root.geometry("400x500")

        self.chat_display = tk.Text(self.root, height=20, width=50)
        self.chat_display.pack(pady=10)

        self.input_field = tk.Entry(self.root, width=40)
        self.input_field.pack(pady=5)

        self.send_button = tk.Button(self.root, text="Send", command=self.process_text)
        self.send_button.pack(pady=5)

        if self.stt:
            self.voice_button = tk.Button(self.root, text="Speak", command=self.process_voice)
            self.voice_button.pack(pady=5)
        else:
            self.chat_display.insert(tk.END, "Voice input is disabled due to initialization error.\n")

        print("Tkinter UI created.")

    def process_text(self):
        query = self.input_field.get()
        self.chat_display.insert(tk.END, f"You: {query}\n")
        intent = self.nlp.predict_intent(query)
        response = self.rag.retrieve(query)
        self.chat_display.insert(tk.END, f"Bot ({intent}): {response}\n")
        self.input_field.delete(0, tk.END)

    def process_voice(self):
        if not self.stt:
            self.chat_display.insert(tk.END, "Voice input is not available.\n")
            return
        audio = self.stt.record_audio()
        query = self.stt.transcribe(audio)
        self.chat_display.insert(tk.END, f"You (voice): {query}\n")
        intent = self.nlp.predict_intent(query)
        response = self.rag.retrieve(query)
        self.chat_display.insert(tk.END, f"Bot ({intent}): {response}\n")

    def run(self):
        print("Starting Tkinter mainloop...")
        self.root.mainloop()
        print("Tkinter mainloop ended.")

if __name__ == "__main__":
    ui = ChatbotUI()
    ui.run()