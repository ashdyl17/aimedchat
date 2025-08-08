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
import google.generativeai as genai
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import base64
import platform
import threading
import time

# ---------------------------------------
# CONFIG
# ---------------------------------------
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
genai.configure(api_key=GEMINI_API_KEY)

# ---------------------------------------
# GEMINI TRANSLATION FUNCTION - MEDICAL FOCUSED
# ---------------------------------------
def translate_tamil_to_english_medical(tamil_text):
    prompt = f"""
    You are a professional bilingual translator specializing in Tamil ↔ English medical translations.

    Task:
    1. Translate the following Tamil text into natural, fluent English.
    2. If the text contains medical terms, ensure accurate medical context is preserved without oversimplification.
    3. If the text is unrelated to medicine, translate it normally without adding any medical interpretation.
    4. Maintain meaning, tone, and clarity exactly as intended in the original.

    Tamil:
    {tamil_text}

    English:
    """

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        translated_text = response.text.strip()
        
        # Check if translation contains medical keywords
        medical_keywords = ['pain', 'fever', 'headache', 'cough', 'cold', 'symptom', 'doctor', 'medicine', 'treatment', 'health', 'medical', 'disease', 'infection', 'blood', 'heart', 'lung', 'stomach', 'bone', 'muscle', 'joint', 'skin', 'eye', 'ear', 'nose', 'throat', 'chest', 'back', 'leg', 'arm', 'hand', 'foot']
        
        if any(keyword in translated_text.lower() for keyword in medical_keywords):
            return translated_text
        else:
            return "Medical consultation required. Please describe your health symptoms or concerns."
            
    except Exception as e:
        return "Medical consultation required. Please describe your health symptoms or concerns."

# ---------------------------------------
# ENHANCED SPEECH FUNCTION WITH TAMIL SUPPORT - MEDICAL FOCUSED
# ---------------------------------------
def listen_with_tamil_support():
    """Listen for voice input with Tamil language support - Medical context only"""
    try:
        with sr.Microphone() as source:
            st.info("🎤 Listening... Speak now! (Supports Tamil and English - Medical topics only)")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
        
        try:
            # First try to recognize as Tamil
            query = recognizer.recognize_google(audio, language="ta-IN")
            st.success(f"🎤 Tamil detected: {query}")
            
            # Translate Tamil to English with medical focus
            english_translation = translate_tamil_to_english_medical(query)
            st.info(f"🌐 English translation (medical): {english_translation}")
            
            return english_translation.lower(), "tamil"
            
        except sr.UnknownValueError:
            # If Tamil fails, try English
            try:
                query = recognizer.recognize_google(audio, language="en-US")
                st.success(f"🎤 English detected: {query}")
                
                # Check if English query is medical-related
                if is_medical_question(query):
                    return query.lower(), "english"
                else:
                    st.warning("⚠ Please ask medical or health-related questions only.")
                    return "", "non-medical"
                    
            except sr.UnknownValueError:
                st.warning("❌ Sorry, I couldn't understand. Please try again.")
                return "", "unknown"
        except sr.RequestError:
            st.error("❌ Speech Recognition Service Unavailable")
            return "", "error"
            
    except Exception as e:
        st.error(f"❌ Microphone error: {str(e)}")
        st.info("💡 You can also type your question in the text input below.")
        return "", "error"

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

# Create a simple medical dataset
def create_medical_dataset():
    """Create a basic medical dataset"""
    data = {
        'disease': [
            'headache', 'fever', 'cough', 'cold', 'stomach ache', 'back pain',
            'sore throat', 'runny nose', 'fatigue', 'insomnia', 'stress',
            'anxiety', 'depression', 'high blood pressure', 'diabetes',
            'asthma', 'allergies', 'migraine', 'dizziness', 'nausea'
        ],
        'cure': [
            'Rest in a quiet, dark room. Stay hydrated and consider over-the-counter pain relievers. If severe or persistent, consult a doctor.',
            'Rest, stay hydrated, and monitor your temperature. Take acetaminophen if needed. Seek medical attention if fever is high or persistent.',
            'Stay hydrated, use honey for soothing, and rest. Consider over-the-counter cough suppressants. See a doctor if cough persists.',
            'Rest, drink plenty of fluids, and use saline nasal drops. Over-the-counter cold medications may help. Symptoms usually resolve in 7-10 days.',
            'Try gentle abdominal massage, peppermint tea, or over-the-counter antacids. Avoid spicy foods. See a doctor if pain is severe.',
            'Rest, apply heat or ice, and consider over-the-counter pain relievers. Gentle stretching may help. Consult a doctor for persistent pain.',
            'Gargle with warm salt water, use throat lozenges, and stay hydrated. Rest your voice. See a doctor if symptoms worsen.',
            'Use saline nasal spray, stay hydrated, and rest. Over-the-counter decongestants may help. Symptoms usually improve within a week.',
            'Ensure adequate sleep, eat a balanced diet, and exercise regularly. Consider stress management techniques. See a doctor if persistent.',
            'Establish a regular sleep schedule, avoid screens before bed, and create a relaxing bedtime routine. Consider consulting a sleep specialist.',
            'Practice relaxation techniques, exercise regularly, and maintain a healthy work-life balance. Consider therapy if stress is overwhelming.',
            'Practice deep breathing, mindfulness, and consider therapy. Regular exercise and adequate sleep can help. Seek professional help if needed.',
            'Consider therapy, medication if prescribed, and maintain social connections. Exercise and healthy lifestyle habits are important.',
            'Reduce salt intake, exercise regularly, and maintain a healthy weight. Monitor blood pressure regularly and follow doctor\'s advice.',
            'Follow a balanced diet, exercise regularly, and monitor blood sugar levels. Take medications as prescribed and see your doctor regularly.',
            'Avoid triggers, use prescribed inhalers, and maintain good air quality. Have an action plan and see your doctor regularly.',
            'Avoid known allergens, use air purifiers, and consider over-the-counter antihistamines. See an allergist for severe allergies.',
            'Rest in a dark, quiet room. Avoid triggers like certain foods or stress. Consider preventive medications if prescribed.',
            'Sit or lie down to prevent falls. Stay hydrated and avoid sudden movements. See a doctor if dizziness is severe or persistent.',
            'Eat small, bland meals and stay hydrated. Avoid strong odors and fatty foods. See a doctor if nausea is severe or persistent.'
        ]
    }
    return pd.DataFrame(data)

# Load medical dataset
try:
    df = pd.read_csv(r'/Users/ashleydylan/Documents/python_projects/codezilla/AI-Health-Assistant/dataset - Sheet1.csv')
except:
    df = create_medical_dataset()

# Initialize the SentenceTransformer model
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
except:
    model = None

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
try:
    llm = OllamaLLM(model="mistral")
    chat_history = ChatMessageHistory()
except:
    llm = None
    chat_history = None

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
    if model is None:
        # Fallback to keyword matching
        for keyword, response in medical_keywords.items():
            if keyword in user_input.lower():
                return response
        
        # Check dataset
        for index, row in df.iterrows():
            if row['disease'] in user_input.lower():
                return row['cure']
        
        return "I understand your concern. While I can provide general information, it's important to consult a healthcare professional for proper diagnosis and treatment. Your health and safety are the top priority."
    
    try:
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
    except:
        # Fallback to keyword matching
        for keyword, response in medical_keywords.items():
            if keyword in user_input.lower():
                return response
        
        # Check dataset
        for index, row in df.iterrows():
            if row['disease'] in user_input.lower():
                return row['cure']
        
        return "I understand your concern. While I can provide general information, it's important to consult a healthcare professional for proper diagnosis and treatment. Your health and safety are the top priority."

# Function to translate text
def translate_text(text, dest_language='en'):
    try:
        translator = GoogleTranslator(source='auto', target=dest_language)
        return translator.translate(text)
    except Exception as e:
        return text

# Enhanced speech function with better language detection and gTTS
def speak(text: str) -> bool:
    try:
        sysname = platform.system()
        # macOS: use native 'say' (no file generation, no timeouts)
        if sysname == "Darwin":
            subprocess.run(["say", text], check=True)
            return True

        # Windows/Linux: use pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 180)
        # Try to select an English voice if available
        try:
            voices = engine.getProperty("voices")
            for v in voices:
                # Some engines expose languages, others only the name
                langs = []
                try:
                    langs = [str(l).lower() for l in getattr(v, "languages", []) or []]
                except Exception:
                    pass
                name = (getattr(v, "name", "") or "").lower()
                if any("en" in l for l in langs) or "english" in name:
                    engine.setProperty("voice", v.id)
                    break
        except Exception:
            pass

        engine.say(text)
        engine.runAndWait()
        return True
    except Exception:
        return False

def speak_async(text: str, delay: float = 0.2):
    def _run():
        try:
            time.sleep(delay)
            speak(text)
        except Exception:
            pass
    threading.Thread(target=_run, daemon=True).start()

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
    input_variables=["chat_history", "question", "patient_info"],
    template="""You are Dr. AI, a specialized medical AI assistant. You focus ONLY on health-related questions and medical concerns. 

If the user asks non-medical questions, politely redirect them to your medical focus.

Patient Information: {patient_info}

Previous Conversation: {chat_history}

User Question: {question}

Remember: You are a medical AI doctor, not a general chatbot. Always maintain your medical focus and provide helpful, accurate medical advice while recommending professional consultation for serious concerns. Consider the patient's age, gender, and other relevant factors when providing personalized medical guidance.

Dr. AI Response:"""
)

# Function to generate medical report from chat history using LangChain
def generate_medical_report(chat_history, patient_info=None):
    """Generate a comprehensive medical report from chat history using LangChain"""
    try:
        # Create temporary PDF file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            pdf_path = temp_file.name
        
        # Create PDF document
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=20,
            alignment=TA_LEFT,
            textColor=colors.darkgreen
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=12,
            spaceAfter=12,
            alignment=TA_JUSTIFY
        )
        
        # Header
        story.append(Paragraph("🏥 Dr. AI Medical Consultation Report", title_style))
        story.append(Spacer(1, 20))
        
        # Patient Information Section
        if patient_info:
            story.append(Paragraph("👤 Patient Information", subtitle_style))
            
            # Create patient info table
            patient_data = [
                ["Name", patient_info.get('name', 'Not provided')],
                ["Age", str(patient_info.get('age', 'Not provided'))],
                ["Gender", patient_info.get('gender', 'Not provided')],
                ["Weight", f"{patient_info.get('weight', 'Not provided')} kg"],
                ["Height", f"{patient_info.get('height', 'Not provided')} cm"],
                ["Blood Group", patient_info.get('blood_group', 'Not provided')],
                ["Emergency Contact", patient_info.get('emergency_contact', 'Not provided')]
            ]
            
            patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
            patient_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(patient_table)
            story.append(Spacer(1, 20))
        
        # Report metadata
        current_time = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        story.append(Paragraph(f"<b>Report Generated:</b> {current_time}", normal_style))
        story.append(Paragraph(f"<b>Total Messages:</b> {len(chat_history)}", normal_style))
        story.append(Spacer(1, 20))
        
        # Extract user concerns and AI responses - ENGLISH ONLY
        user_concerns = []
        ai_recommendations = []
        
        for message in chat_history:
            if message['type'] == 'user':
                # Clean up the content (remove emoji prefixes) - ENGLISH ONLY
                content = message['content']
                if content.startswith('📝 '):
                    content = content[3:]
                elif content.startswith('🎤 '):
                    content = content[3:]
                
                # Only include English content in the report (exclude Tamil)
                if content and not any(char in content for char in ['த', 'ம', 'ன', 'க', 'ப', 'ர', 'ல', 'ட', 'ண', 'ச', 'ய', 'வ', 'ழ', 'ள', 'ற', 'ந']):
                    user_concerns.append({
                        'timestamp': message['timestamp'].strftime("%I:%M %p"),
                        'concern': content
                    })
            elif message['type'] == 'bot':
                # Clean up the content (remove emoji prefixes) - ENGLISH ONLY
                content = message['content']
                if content.startswith('🏥 '):
                    content = content[3:]
                
                # Only include English content in the report (exclude Tamil)
                if content and not any(char in content for char in ['த', 'ம', 'ன', 'க', 'ப', 'ர', 'ல', 'ட', 'ண', 'ச', 'ய', 'வ', 'ழ', 'ள', 'ற', 'ந']):
                    ai_recommendations.append({
                        'timestamp': message['timestamp'].strftime("%I:%M %p"),
                        'recommendation': content
                    })
        
        # Summary section
        story.append(Paragraph("📋 Consultation Summary", subtitle_style))
        story.append(Paragraph(f"<b>Number of Health Concerns Discussed:</b> {len(user_concerns)}", normal_style))
        story.append(Paragraph(f"<b>Number of AI Recommendations Provided:</b> {len(ai_recommendations)}", normal_style))
        story.append(Spacer(1, 20))
        
        # User concerns section - ENGLISH ONLY
        if user_concerns:
            story.append(Paragraph("🤔 Patient Concerns & Symptoms", subtitle_style))
            for i, concern in enumerate(user_concerns, 1):
                story.append(Paragraph(f"<b>{i}. [{concern['timestamp']}]</b> {concern['concern']}", normal_style))
            story.append(Spacer(1, 20))
        
        # AI recommendations section - ENGLISH ONLY
        if ai_recommendations:
            story.append(Paragraph("💡 Dr. AI Recommendations", subtitle_style))
            for i, rec in enumerate(ai_recommendations, 1):
                story.append(Paragraph(f"<b>{i}. [{rec['timestamp']}]</b> {rec['recommendation']}", normal_style))
            story.append(Spacer(1, 20))
        
        # Key insights section using LangChain
        story.append(Paragraph("🔍 Key Medical Insights", subtitle_style))
        
        # Use LangChain to analyze the chat and generate insights
        chat_text = "\n".join([f"{msg['type'].capitalize()}: {msg['content']}" for msg in chat_history])
        
        # LangChain prompt for medical insights
        insights_prompt = PromptTemplate(
            input_variables=["chat_history", "patient_info"],
            template="""You are a medical AI assistant analyzing a patient consultation chat history. 
            Based on the conversation and patient information, provide 3-5 key medical insights that summarize the main health concerns, 
            patterns, and recommendations discussed.
            
            Patient Information:
            {patient_info}
            
            Chat History:
            {chat_history}
            
            Please provide insights in the following format:
            • [Insight 1]
            • [Insight 2]
            • [Insight 3]
            
            Focus on:
            - Main health concerns discussed
            - Recurring symptoms or patterns
            - Key recommendations provided
            - Important medical themes
            - Lifestyle or wellness advice given
            - Age and gender-specific considerations if relevant
            
            Keep each insight concise but informative. Start each with a bullet point (•).
            
            Medical Insights:"""
        )
        
        try:
            # Generate insights using LangChain
            patient_info_text = ""
            if patient_info:
                patient_info_text = f"Name: {patient_info.get('name', 'Not provided')}, Age: {patient_info.get('age', 'Not provided')}, Gender: {patient_info.get('gender', 'Not provided')}, Weight: {patient_info.get('weight', 'Not provided')} kg, Height: {patient_info.get('height', 'Not provided')} cm, Blood Group: {patient_info.get('blood_group', 'Not provided')}"
            
            if llm:
                insights_response = llm.invoke(insights_prompt.format(chat_history=chat_text, patient_info=patient_info_text))
                
                # Split insights into lines and add to PDF
                insights_lines = insights_response.strip().split('\n')
                for line in insights_lines:
                    if line.strip() and line.strip().startswith('•'):
                        story.append(Paragraph(line.strip(), normal_style))
                    elif line.strip():
                        story.append(Paragraph(f"• {line.strip()}", normal_style))
            else:
                # Fallback to basic insights if LangChain fails
                story.append(Paragraph("• Medical consultation completed successfully", normal_style))
                story.append(Paragraph("• Patient concerns were addressed with appropriate guidance", normal_style))
                story.append(Paragraph("• Professional medical consultation recommended for serious concerns", normal_style))
                    
        except Exception as e:
            # Fallback to basic insights if LangChain fails
            story.append(Paragraph("• Medical consultation completed successfully", normal_style))
            story.append(Paragraph("• Patient concerns were addressed with appropriate guidance", normal_style))
            story.append(Paragraph("• Professional medical consultation recommended for serious concerns", normal_style))
        
        story.append(Spacer(1, 20))
        
        # Generate personalized recommendations using LangChain
        story.append(Paragraph("💡 Personalized Recommendations", subtitle_style))
        
        # LangChain prompt for personalized recommendations
        recommendations_prompt = PromptTemplate(
            input_variables=["chat_history", "patient_info"],
            template="""You are a medical AI assistant creating personalized recommendations based on a patient consultation.
            Based on the chat history and patient information, provide 3-4 personalized recommendations for the patient.
            
            Patient Information:
            {patient_info}
            
            Chat History:
            {chat_history}
            
            Please provide recommendations in the following format:
            • [Recommendation 1]
            • [Recommendation 2]
            • [Recommendation 3]
            
            Focus on:
            - Specific lifestyle changes
            - Preventive measures
            - Follow-up actions
            - Wellness tips relevant to their concerns
            - Age and gender-specific recommendations if relevant
            
            Keep each recommendation actionable and specific. Start each with a bullet point (•).
            
            Personalized Recommendations:"""
        )
        
        try:
            # Generate recommendations using LangChain
            if llm:
                recommendations_response = llm.invoke(recommendations_prompt.format(chat_history=chat_text, patient_info=patient_info_text))
                
                # Split recommendations into lines and add to PDF
                recommendations_lines = recommendations_response.strip().split('\n')
                for line in recommendations_lines:
                    if line.strip() and line.strip().startswith('•'):
                        story.append(Paragraph(line.strip(), normal_style))
                    elif line.strip():
                        story.append(Paragraph(f"• {line.strip()}", normal_style))
            else:
                # Fallback to general recommendations if LangChain fails
                story.append(Paragraph("• Continue monitoring your symptoms and seek professional help if they worsen", normal_style))
                story.append(Paragraph("• Maintain a healthy lifestyle with proper diet, exercise, and sleep", normal_style))
                story.append(Paragraph("• Schedule regular check-ups with your healthcare provider", normal_style))
                    
        except Exception as e:
            # Fallback to general recommendations if LangChain fails
            story.append(Paragraph("• Continue monitoring your symptoms and seek professional help if they worsen", normal_style))
            story.append(Paragraph("• Maintain a healthy lifestyle with proper diet, exercise, and sleep", normal_style))
            story.append(Paragraph("• Schedule regular check-ups with your healthcare provider", normal_style))
        
        story.append(Spacer(1, 20))
        
        # Important disclaimer
        story.append(Paragraph("⚠ Important Medical Disclaimer", subtitle_style))
        disclaimer_text = """
        This report is generated by Dr. AI, an artificial intelligence medical assistant. 
        The information provided is for educational and informational purposes only and 
        should not be considered as medical advice, diagnosis, or treatment. 
        
        Always consult with qualified healthcare professionals for proper medical 
        diagnosis, treatment, and care. This report is not a substitute for 
        professional medical consultation.
        
        If you are experiencing a medical emergency, please contact emergency 
        services immediately.
        """
        story.append(Paragraph(disclaimer_text, normal_style))
        
        # Build PDF
        doc.build(story)
        
        # Read the generated PDF
        with open(pdf_path, 'rb') as pdf_file:
            pdf_data = pdf_file.read()
        
        # Clean up temporary file
        try:
            os.unlink(pdf_path)
        except:
            pass
        
        return pdf_data
        
    except Exception as e:
        st.error(f"Error generating report: {str(e)}")
        return None

# Function to create download button for PDF
def get_download_link(pdf_data, filename):
    """Create a download link for the PDF"""
    b64 = base64.b64encode(pdf_data).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}" target="_blank">📄 Download Medical Report</a>'
    return href

# Function to process AI response
def run_chain(question):
    """Process user question through the AI chain using LangChain"""
    try:
        # Check if question is medical-related
        if not is_medical_question(question):
            return get_medical_focus_response()
        
        # Get chat history as text
        if chat_history:
            chat_history_text = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in chat_history.messages])
        else:
            chat_history_text = ""
        
        # Collect patient information for personalized responses
        patient_info = {
            'name': st.session_state.get('patient_name', ''),
            'age': st.session_state.get('patient_age', ''),
            'gender': st.session_state.get('patient_gender', ''),
            'weight': st.session_state.get('patient_weight', ''),
            'height': st.session_state.get('patient_height', ''),
            'blood_group': st.session_state.get('patient_blood_group', ''),
            'emergency_contact': st.session_state.get('patient_emergency_contact', '')
        }
        
        # Create patient info text for the prompt
        patient_info_text = ""
        if patient_info.get('name'):
            patient_info_text = f"Name: {patient_info.get('name')}, Age: {patient_info.get('age')}, Gender: {patient_info.get('gender')}, Weight: {patient_info.get('weight')} kg, Height: {patient_info.get('height')} cm, Blood Group: {patient_info.get('blood_group')}"
        
        # Get comprehensive medical response using LangChain
        if llm:
            response = llm.invoke(medical_prompt.format(chat_history=chat_history_text, question=question, patient_info=patient_info_text))
            
            # Add to chat history
            if chat_history:
                chat_history.add_user_message(question)
                chat_history.add_ai_message(response)
        else:
            # Fallback to simple response
            response = find_best_cure(question)
        
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
    st.markdown("### ⚙ Settings")
    language_choice = st.selectbox(
        "🌐 Language",
        ["English", "Hindi", "Gujarati", "Korean", "Turkish", "German", "French", "Arabic", "Urdu", "Tamil", "Telugu", "Chinese", "Japanese"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 📊 Chat Statistics")
    
    # Chat stats in styled container
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric("Messages", len(st.session_state.chat_history))
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("🗑 Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📄 Medical Report")
    
    # Report generation section
    if len(st.session_state.chat_history) > 0:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("📋 Generate Report", key="generate_report"):
                with st.spinner("Generating medical report..."):
                    # Collect patient information
                    patient_info = {
                        'name': st.session_state.get('patient_name', ''),
                        'age': st.session_state.get('patient_age', ''),
                        'gender': st.session_state.get('patient_gender', ''),
                        'weight': st.session_state.get('patient_weight', ''),
                        'height': st.session_state.get('patient_height', ''),
                        'blood_group': st.session_state.get('patient_blood_group', ''),
                        'emergency_contact': st.session_state.get('patient_emergency_contact', '')
                    }
                    
                    pdf_data = generate_medical_report(st.session_state.chat_history, patient_info)
                    if pdf_data:
                        st.session_state.pdf_data = pdf_data
                        st.session_state.report_generated = True
                        st.success("✅ Report generated successfully!")
                        st.rerun()
        
        with col2:
            if hasattr(st.session_state, 'report_generated') and st.session_state.report_generated:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"Dr_AI_Medical_Report_{timestamp}.pdf"
                
                # Create download link
                download_link = get_download_link(st.session_state.pdf_data, filename)
                st.markdown(download_link, unsafe_allow_html=True)
    else:
        st.info("💬 Start a conversation to generate a medical report")

# Main interface
# Patient Information Section
st.markdown("### 👤 Patient Information")
patient_col1, patient_col2, patient_col3, patient_col4 = st.columns([2, 1, 1, 1])

with patient_col1:
    patient_name = st.text_input("Full Name", placeholder="Enter patient's full name", key="patient_name")
with patient_col2:
    patient_age = st.number_input("Age", min_value=0, max_value=120, value=25, key="patient_age")
with patient_col3:
    patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="patient_gender")
with patient_col4:
    patient_weight = st.number_input("Weight (kg)", min_value=0.0, max_value=300.0, value=70.0, key="patient_weight")

# Additional patient details
patient_col5, patient_col6, patient_col7 = st.columns([1, 1, 1])
with patient_col5:
    patient_height = st.number_input("Height (cm)", min_value=0.0, max_value=300.0, value=170.0, key="patient_height")
with patient_col6:
    patient_blood_group = st.selectbox("Blood Group", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-", "Unknown"], key="patient_blood_group")
with patient_col7:
    patient_emergency_contact = st.text_input("Emergency Contact", placeholder="Phone number", key="patient_emergency_contact")

st.markdown("---")

# Chat Interface
st.markdown("### 💬 Chat with Dr. AI")

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
        <p>🎤 <strong>NEW:</strong> Speak in Tamil or English! Click the microphone button to start.</p>
        <p>💬 Or type your medical question below.</p>
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
        # Check if input is medical-related
        if not is_medical_question(user_input):
            st.warning("⚠ Please ask medical or health-related questions only.")
        else:
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
            speak_async(response)
            
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
    st.markdown('<div class="voice-status listening">🎤 Listening... Speak in Tamil or English!</div>', unsafe_allow_html=True)
    
    # Listen for voice input
    query, detected_lang = listen_with_tamil_support()
    
    if query:
        # Check for exit commands
        if "exit" in query or "stop" in query or "goodbye" in query:
            speak_async("Goodbye! Take care of your health.", delay=0.1)
            st.session_state.listening = False
            st.rerun()
        
        # Process the query
        if query and detected_lang != "non-medical":
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
            speak_async(response)
            
            # Stop listening
            st.session_state.listening = False
            st.rerun()
        else:
            # Stop listening if no input received or non-medical
            st.session_state.listening = False
            st.rerun()
    else:
        # Stop listening if no input received
        st.session_state.listening = False
        st.rerun()

# Report preview section
if hasattr(st.session_state, 'report_generated') and st.session_state.report_generated and len(st.session_state.chat_history) > 0:
    st.markdown("---")
    st.markdown("### 📄 Medical Report Preview")
    
    # Show patient information preview
    if st.session_state.get('patient_name'):
        st.markdown("👤 Patient Information:")
        patient_info_col1, patient_info_col2, patient_info_col3 = st.columns([2, 1, 1])
        with patient_info_col1:
            st.info(f"*Name:* {st.session_state.get('patient_name', 'Not provided')}")
        with patient_info_col2:
            st.info(f"*Age:* {st.session_state.get('patient_age', 'Not provided')}")
        with patient_info_col3:
            st.info(f"*Gender:* {st.session_state.get('patient_gender', 'Not provided')}")
    
    # Show report summary
    user_messages = [msg for msg in st.session_state.chat_history if msg['type'] == 'user']
    bot_messages = [msg for msg in st.session_state.chat_history if msg['type'] == 'bot']
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.metric("📝 Patient Concerns", len(user_messages))
    
    with col2:
        st.metric("💡 AI Recommendations", len(bot_messages))
    
    with col3:
        st.metric("📊 Total Messages", len(st.session_state.chat_history))
    
    # Show recent concerns and recommendations - ENGLISH ONLY
    if user_messages:
        st.markdown("*Recent Patient Concerns:*")
        for i, msg in enumerate(user_messages[-3:], 1):  # Show last 3 concerns
            content = msg['content']
            if content.startswith('📝 '):
                content = content[3:]
            elif content.startswith('🎤 '):
                content = content[3:]
            
            # Only show English content
            if not any(char in content for char in ['த', 'ம', 'ன', 'க', 'ப', 'ர', 'ல', 'ட', 'ண', 'ச', 'ய', 'வ', 'ழ', 'ள', 'ற', 'ந']):
                st.markdown(f"{i}. {content}")
    
    if bot_messages:
        st.markdown("*Recent AI Recommendations:*")
        for i, msg in enumerate(bot_messages[-3:], 1):  # Show last 3 recommendations
            content = msg['content']
            if content.startswith('🏥 '):
                content = content[3:]
            
            # Only show English content
            if not any(char in content for char in ['த', 'ம', 'ன', 'க', 'ப', 'ர', 'ல', 'ட', 'ண', 'ச', 'ய', 'வ', 'ழ', 'ள', 'ற', 'ந']):
                st.markdown(f"{i}. {content[:100]}{'...' if len(content) > 100 else ''}")
    
    # Show LangChain-generated insights preview
    st.markdown("🔍 AI-Generated Insights:")
    st.info("The full report includes AI-generated medical insights and personalized recommendations based on your conversation with Dr. AI.")
    
    # Download button in main interface
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"Dr_AI_Medical_Report_{timestamp}.pdf"
    download_link = get_download_link(st.session_state.pdf_data, filename)
    st.markdown(f"<div style='text-align: center; margin: 20px 0;'>{download_link}</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer-box">
    <p><strong>🏥 Dr. AI</strong> • Your specialized AI medical doctor</p>
    <p>⚠ This is for informational purposes only. Always consult healthcare professionals for medical advice.</p>
</div>
""", unsafe_allow_html=True)
