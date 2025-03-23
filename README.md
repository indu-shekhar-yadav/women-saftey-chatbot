

# Women Safety Chatbot  

A chatbot designed for women’s safety, leveraging DeepSpeech and BERT models for speech recognition and text processing.  

## 🚀 Getting Started  

Follow the steps below to set up and run the project.  

### 1️⃣ Clone or Download the Project  

```sh
git clone https://github.com/indu-shekhar-yadav/women-saftey-chatbot.git
cd women-saftey-chatbot
```
OR  
Download the ZIP file, extract it, and navigate to the project directory.  

### 2️⃣ Place Required Model Files  

Download the required model files from the [Google Drive link](https://drive.google.com/drive/folders/1QWvKXkBt6u8vZCciEdA1HLdy0xd18JjJ?usp=sharing).  

- Place **`deepspeech-0.9.3-models.scorer`** and **`deepspeech-0.9.3-models.pbmm`** in the **root directory** of the project.  
- Place **`model.safetensors`** inside the **`trained_bert`** folder.  

### 3️⃣ Install Dependencies  

Ensure you have **Python 3.9** installed. Then, create and activate a virtual environment:  

```sh
python3 -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows
```

Now, install the required dependencies:  

```sh
pip install -r requirements.txt
```

### 4️⃣ Run the Chatbot  

Execute the following command to start the chatbot:  

```sh
python main.py
```

## ✅ Done! 🎉  

Your chatbot should now be up and running. If you face any issues, feel free to check the repository:  
🔗 [GitHub Repository](https://github.com/indu-shekhar-yadav/women-saftey-chatbot)  
