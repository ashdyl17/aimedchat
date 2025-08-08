# 🏥 Dr. AI - Multilingual Medical Translation Assistant

Welcome to **Dr. AI**, a cutting-edge AI-powered medical assistant designed to break down language barriers in healthcare settings. By enabling seamless, real-time communication between patients and healthcare providers, Dr. AI ensures accurate and accessible medical dialogues across multiple languages.

---

## 🩺 Problem Statement: Multilingual Medical Translation Assistant

Effective communication is critical in healthcare, yet language barriers often lead to misunderstandings, misdiagnoses, and patient dissatisfaction. **Dr. AI** addresses this challenge by providing a robust, real-time translation application tailored for healthcare environments.  

**Core requirements include:**  
- Translation of spoken or written medical dialogue between at least two languages.  
- Accurate handling of medical terminology and patient symptoms.  
- Low-latency operation for natural conversational flow.  
- Bonus features like speech-to-text transcription, text-to-speech output, and potential support for sign language (text/voice to sign animation) to enhance accessibility.

---

## ✨ Why Dr. AI Excels

Dr. AI goes beyond the core requirements, delivering a comprehensive, user-centric experience that transforms it into an indispensable medical companion.

### 🗣️ Advanced Multilingual & Medical-Focused Translation

- **Extensive Multilingual Support (13 Languages):** Dr. AI supports translation across 13 languages, including English, Hindi, Gujarati, Korean, Turkish, German, French, Arabic, Urdu, Tamil, Telugu, Chinese, and Japanese.  
- **Specialized Tamil ↔ English Medical Translations:** Uses the Gemini API (`gemini-1.5-flash`) with a specialized prompt for highly accurate, medical-focused Tamil ↔ English translations, preserving nuances in medical terminology and patient symptoms.  
- **Intelligent Language Detection:** Automatically detects whether the user is speaking in Tamil or English, ensuring a seamless experience without manual language selection.

### 🧠 Intelligent Medical Advice & Personalized Guidance

- **Specialized Medical AI Persona:** Built with LangChain's PromptTemplate, Dr. AI is designed as a dedicated medical assistant, politely redirecting non-medical queries to maintain focus.  
- **Actionable Medical Advice:** Powered by a SentenceTransformer model and a comprehensive medical dataset, Dr. AI deeply understands patient symptoms across languages, offering proactive, actionable advice, including potential medications and treatment approaches.  
- **Personalized Responses:** Users can input patient details (name, age, gender, weight, height, blood group, emergency contact), which Dr. AI incorporates into its prompts to deliver tailored, contextually relevant medical guidance.

### 📄 Comprehensive Medical Report Generation

- **Dynamic PDF Reports:** Generates professional PDF medical reports using the `reportlab` library, summarizing the entire chat history.  
- **AI-Generated Insights:** Reports include:  
  - **Key Medical Insights:** Summaries of health concerns, recurring symptoms, and medical themes.  
  - **Personalized Recommendations:** Actionable lifestyle changes, preventive measures, and follow-up actions tailored to the patient.  
- **Structured Output:** Reports feature custom styles, patient information tables, and clear sections for easy review and record-keeping.

### 🔊 Enhanced Accessibility & User Experience

- **Speech-to-Text & Text-to-Speech:** Full support for speech input and AI response output, optimized for platforms (e.g., native `say` on macOS).  
- **Intuitive UI:** Built with Streamlit and enhanced with custom CSS for an Apple-inspired, clean, and user-friendly interface.  
- **Chat History Management:** Displays conversation history clearly with a "Clear Chat History" option.  
- **Real-Time Feedback:** Visual (e.g., "🎤 Listening...") and audio cues enhance the user experience during speech recognition and synthesis.

---

## 🚀 How to Run

Follow these steps to run Dr. AI locally:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/ashdyl17/aimedchat
   cd aimedchat
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```  
   Ensure the following packages are installed:  
   `streamlit`, `speech_recognition`, `pyttsx3`, `langchain-community`, `langchain-core`, `langchain-ollama`, `pandas`, `sentence-transformers`, `deep-translator`, `google-generativeai`, `reportlab`, `gtts`, `langdetect`.

3. **Set Up Gemini API Key:**  
   In `chat1.py`, configure your Gemini API key:  
   ```python
   GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"  # Replace with your actual key
   genai.configure(api_key=GEMINI_API_KEY)
   ```

4. **Run the Streamlit Application:**
   ```bash
   streamlit run chat1.py
   ```  
   The application will launch in your web browser. Ensure your microphone is configured and permissions are granted for speech input.

---

## 👥 Team Members

- Abiram H  
- Ashley Dylan J  
- Shebin Sivakumar  
- Palaniyappan S  
- Madhavan S  

---

**Dr. AI** is more than a translation tool—it's a comprehensive medical assistant that empowers patients and providers with accurate, accessible, and actionable healthcare communication. 🌟
