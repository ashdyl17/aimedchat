import speech_recognition as sr
import pyttsx3
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
import pandas as pd
import random
from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator
import streamlit as st
from datetime import datetime
import io
import subprocess
import os
from langdetect import detect
from gtts import gTTS
import tempfile

# Page configuration
st.set_page_config(
    page_title="AI Medical Doctor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Apple-themed design
st.markdown("""
<style>
    /* Reset and base styles */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
    }
    
    /* Header styling */
    .header-box {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 30px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    .header-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    
    .header-subtitle {
        color: #666;
        font-size: 1.2rem;
        margin: 0;
    }
    
    /* Chat container */
    .chat-box {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        min-height: 400px;
        max-height: 500px;
        overflow-y: auto;
        margin-bottom: 20px;
    }
    
    /* Message styling */
    .user-msg {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 20px 20px 5px 20px;
        margin: 10px 0;
        max-width: 70%;
        margin-left: auto;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .bot-msg {
        background: rgba(255, 255, 255, 0.9);
        color: #333;
        padding: 15px 20px;
        border-radius: 20px 20px 20px 5px;
        margin: 10px 0;
        max-width: 70%;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(0, 0, 0, 0.05);
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 25px !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        padding: 12px 25px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Mic button styling */
    .mic-button {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%) !important;
        border-radius: 50% !important;
        width: 80px !important;
        height: 80px !important;
        font-size: 2rem !important;
        animation: pulse 1.5s infinite;
    }
    
    .mic-button.listening {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%) !important;
        animation: pulse 1s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div > select {
        border-radius: 15px !important;
        border: 2px solid rgba(102, 126, 234, 0.3) !important;
        background: rgba(255, 255, 255, 0.9) !important;
        padding: 8px 15px !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95) !important;
    }
    
    /* Footer styling */
    .footer-box {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Metric styling */
    .metric-container {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Voice status indicator */
    .voice-status {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    .listening {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%) !important;
        color: white !important;
        animation: pulse 1.5s infinite;
    }
</style>
""", unsafe_allow_html=True)

# Load medical dataset
df = pd.read_csv(r'/Users/ashleydylan/Documents/python_projects/codezilla/AI-Health-Assistant/dataset - Sheet1.csv')

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define medical keywords for fallback
medical_keywords = {
    "fever": "It sounds like you may have a fever. Stay hydrated and consider seeing a doctor if symptoms persist.",
    "cough": "A persistent cough might be due to an infection or allergy. Try warm fluids and rest.",
    "headache": "Headaches can have many causes, including stress and dehydration. Consider resting and drinking water.",
    "cold": "Common colds usually go away on their own. Stay warm, drink fluids, and get rest.",
}

# List of health tips categorized by keywords
health_tips = {
    "sleep": [
        "Try to get at least 7-8 hours of sleep each night.",
        "Establish a regular sleep routine to improve sleep quality.",
        "Avoid screens before bed to help your mind relax.",
    ],
    "energy": [
        "Make sure you're eating a balanced diet to maintain energy.",
        "Exercise regularly to boost your energy levels.",
        "Stay hydrated throughout the day to avoid fatigue.",
    ],
    "stress": [
        "Take short breaks throughout the day to reduce stress.",
        "Practice mindfulness or meditation to help manage stress.",
        "Engage in physical activity to reduce anxiety and stress.",
    ],
    "general": [
        "Drink plenty of water throughout the day.",
        "Get at least 30 minutes of exercise every day.",
        "Eat a balanced diet rich in fruits and vegetables.",
    ],
}

# Initialize LangChain components
llm = OllamaLLM(model="mistral")
chat_history = ChatMessageHistory()

# Initialize speech recognition
recognizer = sr.Recognizer()

# Language codes for translation
language_codes = {
    "English": "en",
    "Hindi": "hi",
    "Gujarati": "gu",
    "Korean": "ko",
    "Turkish": "tr",
    "German": "de",
    "French": "fr",
    "Arabic": "ar",
    "Urdu": "ur",
    "Tamil": "ta",
    "Telugu": "te",
    "Chinese": "zh",
    "Japanese": "ja",
}

# Function to check if question is medical-related
def is_medical_question(question):
    """Check if the question is medical-related"""
    medical_keywords = [
        'pain', 'hurt', 'ache', 'fever', 'cough', 'cold', 'flu', 'headache', 'stomach',
        'nausea', 'vomit', 'diarrhea', 'constipation', 'rash', 'itch', 'swelling',
        'bruise', 'cut', 'burn', 'bleeding', 'dizzy', 'tired', 'fatigue', 'weak',
        'sick', 'ill', 'disease', 'infection', 'virus', 'bacteria', 'medicine',
        'medication', 'pill', 'tablet', 'symptom', 'diagnosis', 'treatment',
        'doctor', 'hospital', 'clinic', 'emergency', 'ambulance', 'surgery',
        'operation', 'therapy', 'recovery', 'healing', 'wound', 'injury',
        'bone', 'muscle', 'joint', 'back', 'neck', 'shoulder', 'knee', 'ankle',
        'heart', 'lung', 'liver', 'kidney', 'brain', 'blood', 'pressure',
        'diabetes', 'cancer', 'asthma', 'allergy', 'asthma', 'arthritis',
        'sleep', 'insomnia', 'stress', 'anxiety', 'depression', 'mental',
        'diet', 'nutrition', 'vitamin', 'supplement', 'exercise', 'fitness',
        'weight', 'obesity', 'cholesterol', 'blood sugar', 'hypertension'
    ]
    
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in medical_keywords)

# Function to get medical-focused response for non-medical questions
def get_medical_focus_response():
    """Get a response redirecting to medical focus"""
    responses = [
        "I'm Dr. AI, your specialized medical assistant. I'm designed to help with health-related questions and medical concerns. For general questions, I'd recommend consulting other AI assistants. How can I help you with your health today?",
        "As your AI medical doctor, I focus specifically on health and medical issues. I can help you with symptoms, health advice, medication questions, and wellness tips. What health concern would you like to discuss?",
        "I'm a medical AI assistant, not a general chatbot. I'm here to help with your health questions, symptoms, medical advice, and wellness guidance. What medical issue would you like to discuss?",
        "Hello! I'm your AI medical doctor. I specialize in health-related questions and medical concerns. For non-medical topics, other AI assistants would be better suited. How can I assist you with your health today?"
    ]
    return random.choice(responses)

# Function to get personalized health tip
def get_personalized_health_tip(user_input):
    user_input_lower = user_input.lower()
    
    if "tired" in user_input_lower or "fatigue" in user_input_lower:
        return random.choice(health_tips["energy"])
    elif "sleep" in user_input_lower or "rest" in user_input_lower:
        return random.choice(health_tips["sleep"])
    elif "stress" in user_input_lower or "anxious" in user_input_lower:
        return random.choice(health_tips["stress"])
    else:
        return random.choice(health_tips["general"])

# Function to find the best cure based on similarity
def find_best_cure(user_input):
    user_input_embedding = model.encode(user_input, convert_to_tensor=True)
    disease_embeddings = model.encode(df['disease'].tolist(), convert_to_tensor=True)
    
    similarities = util.pytorch_cos_sim(user_input_embedding, disease_embeddings)[0]
    best_match_idx = similarities.argmax().item()
    best_match_score = similarities[best_match_idx].item()
    
    SIMILARITY_THRESHOLD = 0.5
    
    if best_match_score < SIMILARITY_THRESHOLD:
        for keyword, response in medical_keywords.items():
            if keyword in user_input.lower():
                return response
        return "I'm sorry, I don't have enough information on this medical condition. Please consult a healthcare professional for proper diagnosis and treatment."
    
    return df.iloc[best_match_idx]['cure']

# Function to translate text
def translate_text(text, dest_language='en'):
    try:
        translator = GoogleTranslator(source='auto', target=dest_language)
        return translator.translate(text)
    except Exception as e:
        return text

# Enhanced speech function with better language detection and gTTS
def speak(text):
    """Convert text to speech using gTTS with automatic language detection"""
    try:
        # Detect language automatically
        detected_lang = detect(text)
        
        # Map detected language to gTTS language codes
        lang_mapping = {
            'en': 'en',
            'hi': 'hi',
            'gu': 'gu',
            'ko': 'ko',
            'tr': 'tr',
            'de': 'de',
            'fr': 'fr',
            'ar': 'ar',
            'ur': 'ur',
            'ta': 'ta',
            'te': 'te',
            'zh': 'zh',
            'ja': 'ja'
        }
        
        # Use detected language or fallback to English
        lang_code = lang_mapping.get(detected_lang, 'en')
        
        # Create temporary file for audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            temp_filename = temp_file.name
        
        # Generate speech using gTTS
        tts = gTTS(text=text, lang=lang_code, slow=False)
        tts.save(temp_filename)
        
        # Play the audio using system command
        try:
            # Try using afplay (macOS) first
            subprocess.run(['afplay', temp_filename], check=True, timeout=30)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to say command
            try:
                subprocess.run(['say', text], check=True, timeout=30)
            except (subprocess.CalledProcessError, FileNotFoundError):
                # Final fallback to pyttsx3
                engine = pyttsx3.init()
                engine.setProperty("rate", 180)
                engine.say(text)
                engine.runAndWait()
        
        # Clean up temporary file
        try:
            os.unlink(temp_filename)
        except:
            pass
        
        return True
        
    except Exception as e:
        st.error(f"Speech error: {str(e)}")
        # Fallback to simple speech
        try:
            subprocess.run(['say', text], check=True, timeout=30)
            return True
        except:
            return False

# Function to listen for voice input
def listen():
    """Listen for voice input and convert to text"""
    try:
        with sr.Microphone() as source:
            st.info("🎤 Listening... Speak now!")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
        
        try:
            query = recognizer.recognize_google(audio)
            st.success(f"🎤 You said: {query}")
            return query.lower()
        except sr.UnknownValueError:
            st.warning("❌ Sorry, I couldn't understand. Please try again.")
            return ""
        except sr.RequestError:
            st.error("❌ Speech Recognition Service Unavailable")
            return ""
            
    except Exception as e:
        st.error(f"❌ Microphone error: {str(e)}")
        st.info("💡 You can also type your question in the text input below.")
        return ""

# Function to check microphone availability
def check_microphone():
    """Check if microphone is available"""
    try:
        with sr.Microphone() as source:
            return True
    except Exception as e:
        st.error(f"❌ Microphone not available: {str(e)}")
        st.info("💡 Please check your microphone permissions and try again.")
        return False

# Medical prompt template
medical_prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""You are Dr. AI, a specialized medical AI assistant. You focus ONLY on health-related questions and medical concerns. 

If the user asks non-medical questions, politely redirect them to your medical focus.

Previous Conversation: {chat_history}

User Question: {question}

Remember: You are a medical AI doctor, not a general chatbot. Always maintain your medical focus and provide helpful, accurate medical advice while recommending professional consultation for serious concerns.

Dr. AI Response:"""
)

# Function to process AI response
def run_chain(question):
    """Process user question through the AI chain"""
    try:
        # Check if question is medical-related
        if not is_medical_question(question):
            return get_medical_focus_response()
        
        # Get chat history as text
        chat_history_text = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in chat_history.messages])
        
        # Get medical response
        medical_response = find_best_cure(question)
        
        # If medical response is generic, use LLM for better response
        if "consult a healthcare professional" in medical_response.lower():
            response = llm.invoke(medical_prompt.format(chat_history=chat_history_text, question=question))
        else:
            response = medical_response
        
        # Add to chat history
        chat_history.add_user_message(question)
        chat_history.add_ai_message(response)
        
        return response
        
    except Exception as e:
        st.error(f"AI processing error: {str(e)}")
        return "I'm sorry, I encountered an error. Please try again."

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'listening' not in st.session_state:
    st.session_state.listening = False

# Header
st.markdown("""
<div class="header-box">
    <h1 class="header-title">🏥 Dr. AI - Your Medical Assistant</h1>
    <p class="header-subtitle">Your specialized AI medical doctor • Focused on health and medical concerns</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for settings
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    language_choice = st.selectbox(
        "🌍 Language",
        ["English", "Hindi", "Gujarati", "Korean", "Turkish", "German", "French", "Arabic", "Urdu", "Tamil", "Telugu", "Chinese", "Japanese"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 📊 Chat Statistics")
    
    # Chat stats in styled container
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric("Messages", len(st.session_state.chat_history))
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Main interface
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # Chat container
    st.markdown('<div class="chat-box">', unsafe_allow_html=True)
    
    # Display chat history
    if not st.session_state.chat_history:
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 50px 20px;">
            <h3>🏥 Welcome to Dr. AI!</h3>
            <p>I'm your specialized AI medical doctor. I can help with:</p>
            <ul style="text-align: left; display: inline-block;">
                <li>Health symptoms and concerns</li>
                <li>Medical advice and guidance</li>
                <li>Medication questions</li>
                <li>Wellness and prevention tips</li>
            </ul>
            <p>Click the microphone button to speak or type your medical question below.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for message in st.session_state.chat_history:
            if message['type'] == 'user':
                st.markdown(f'<div class="user-msg">{message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-msg">🏥 {message["content"]}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Text input
    st.markdown("<br>", unsafe_allow_html=True)
    user_input = st.text_input(
        "💬 Ask Dr. AI about your health...",
        placeholder="e.g., I have a headache, What should I do for better sleep?",
        key="text_input"
    )
    
    # Send button for text input
    if st.button("📤 Send", use_container_width=True):
        if user_input:
            # Add user message to chat
            st.session_state.chat_history.append({
                'type': 'user',
                'content': f"📝 {user_input}",
                'timestamp': datetime.now()
            })
            
            # Get AI response
            response = run_chain(user_input)
            
            # Translate response if needed
            if language_choice != "English":
                response = translate_text(response, dest_language=language_codes[language_choice])
            
            # Add bot response to chat
            st.session_state.chat_history.append({
                'type': 'bot',
                'content': response,
                'timestamp': datetime.now()
            })
            
            # Speak the response
            speak(response)
            
            st.rerun()
    
    # Mic button
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("🎤", key="mic_button", use_container_width=True, help="Click to speak"):
            if check_microphone():
                st.session_state.listening = True
                st.rerun()
    
    # Voice status indicator
    if st.session_state.listening:
        st.markdown('<div class="voice-status listening">🎤 Listening... Speak now!</div>', unsafe_allow_html=True)
        
        # Listen for voice input
        query = listen()
        
        if query:
            # Check for exit commands
            if "exit" in query or "stop" in query or "goodbye" in query:
                speak("Goodbye! Take care of your health.")
                st.session_state.listening = False
                st.rerun()
            
            # Process the query
            if query:
                # Add user message to chat
                st.session_state.chat_history.append({
                    'type': 'user',
                    'content': f"🎤 {query}",
                    'timestamp': datetime.now()
                })
                
                # Get AI response
                response = run_chain(query)
                
                # Translate response if needed
                if language_choice != "English":
                    response = translate_text(response, dest_language=language_codes[language_choice])
                
                # Add bot response to chat
                st.session_state.chat_history.append({
                    'type': 'bot',
                    'content': response,
                    'timestamp': datetime.now()
                })
                
                # Speak the response
                speak(response)
                
                # Stop listening
                st.session_state.listening = False
                st.rerun()
        else:
            # Stop listening if no input received
            st.session_state.listening = False
            st.rerun()

# Footer
st.markdown("""
<div class="footer-box">
    <p><strong>🏥 Dr. AI</strong> • Your specialized AI medical doctor</p>
    <p>⚠️ This is for informational purposes only. Always consult healthcare professionals for medical advice.</p>
</div>
""", unsafe_allow_html=True)
