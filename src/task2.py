import speech_recognition as sr
from langchain_community.vectorstores import Pinecone as LC_Pinecone  # Renaming to avoid collision
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
import time
import cv2
import numpy as np
from google.generativeai.types import GenerateContentResponse
from transformers import pipeline
import os
from dotenv import load_dotenv
import moviepy.editor as mp
from pinecone import Pinecone as PC_Pinecone, ServerlessSpec  # Renaming to avoid collision

# Load environment variables
load_dotenv()

# Initialize services

# Initialize Gemini
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-pro')

# Initialize other models
sentiment_analyzer = pipeline("sentiment-analysis")
embeddings = HuggingFaceEmbeddings()

def extract_audio_from_video(video_path):
    """Extract audio from video file"""
    video = mp.VideoFileClip(video_path)
    audio_path = video_path.rsplit('.', 1)[0] + '.wav'
    video.audio.write_audiofile(audio_path)
    return audio_path

def transcribe_audio(audio_path):
    """Transcribe audio using speech_recognition"""
    recognizer = sr.Recognizer()
    
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
        
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Speech recognition could not understand the audio"
    except sr.RequestError as e:
        return f"Could not request results from speech recognition service; {str(e)}"

def process_video(video_path):
    """Process uploaded video and extract necessary information"""
    # Extract audio
    audio_path = extract_audio_from_video(video_path)
    
    # Get video frames for analysis
    video = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
    
    video.release()
    
    # Transcribe audio
    transcript = transcribe_audio(audio_path)
    
    # Cleanup
    os.remove(audio_path)
    
    return frames, transcript

def create_vector_store(text, index_name="interview-index"):
    """Create Pinecone vector store from interview transcription"""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.create_documents([text])

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pc = PC_Pinecone(api_key=pinecone_api_key)  # Using the renamed Pinecone from the official SDK

    # Check whether index is already present in the database otherwise it creates new index.
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ) 
        )

        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

    index = pc.Index(index_name)
    
    # Create vector store using LangChain's Pinecone class
    vectorstore = LC_Pinecone.from_documents(
        docs,
        embeddings,
        index_name=index_name
    )
    return vectorstore

async def evaluate_response(transcript, criteria):
    """Evaluate candidate's response using Gemini"""
    prompt = f"""
    Analyze the following interview transcript based on these criteria: {criteria}
    
    Transcript: {transcript}
    
    Please evaluate the response on:
    1. Technical Understanding (1-10)
    2. Problem-Solving Ability (1-10)
    3. Experience Level (1-10)
    
    Provide a detailed score and brief explanation for each aspect.
    Format your response as:
    Technical Understanding: [score]
    Problem-Solving: [score]
    Experience: [score]
    
    Explanation: [your detailed explanation]
    """
    
    # Generate content synchronously since Gemini doesn't support async
    response = model.generate_content(prompt)
    generated_text = response.text
    
    return parse_evaluation_response(generated_text)

def parse_evaluation_response(response_text):
    """Parse the evaluation response into structured format"""
    try:
        # Split the response into lines
        lines = response_text.split('\n')
        scores = {}
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                # Try to extract the numeric score
                try:
                    score = int([num for num in value.split() if num.isdigit()][0])
                    scores[key] = min(max(score, 1), 10)  # Ensure score is between 1 and 10
                except (IndexError, ValueError):
                    continue
        
        # Ensure we have all required scores
        required_scores = {
            'technical_understanding': 7,
            'problem_solving': 7,
            'experience': 7
        }
        
        # Use default scores for any missing metrics
        for key, default_value in required_scores.items():
            if key not in scores:
                scores[key] = default_value
        
        return scores
    except Exception as e:
        print(f"Error parsing response: {e}")
        return {
            'technical_understanding': 7,
            'problem_solving': 7,
            'experience': 7
        }

async def analyze_communication_skills(transcript):
    """Analyze communication skills from transcript"""
    prompt = f"""
    Analyze the following interview transcript for communication skills:
    {transcript[:1000]}...
    
    Evaluate and provide scores (1-10) for:
    1. Clarity of Expression
    2. Confidence Level
    3. Professional Communication
    
    Format your response as:
    Clarity: [score]
    Confidence: [score]
    Professionalism: [score]
    
    Explanation: [your detailed explanation]
    """
    
    # Generate content synchronously
    response = model.generate_content(prompt)
    generated_text = response.text
    
    return parse_communication_response(generated_text)


def parse_communication_response(response_text):
    """Parse the communication analysis response"""
    try:
        lines = response_text.split('\n')
        scores = {}
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                try:
                    score = int([num for num in value.split() if num.isdigit()][0])
                    scores[key] = min(max(score, 1), 10)
                except (IndexError, ValueError):
                    continue
        
        # Ensure we have all required scores
        required_scores = {
            'clarity': 7,
            'confidence': 7,
            'professionalism': 7
        }
        
        for key, default_value in required_scores.items():
            if key not in scores:
                scores[key] = default_value
        
        return scores
    except Exception as e:
        print(f"Error parsing response: {e}")
        return {
            'clarity': 7,
            'confidence': 7,
            'professionalism': 7
        }

async def generate_feedback_report(evaluations, communication_scores):
    """Generate comprehensive feedback report using Gemini"""
    prompt = f"""
    Based on the following evaluation:
    Technical Scores: {evaluations}
    Communication Scores: {communication_scores}
    
    Generate a comprehensive feedback report including:
    1. Overall Score (0-10)
    2. Key Strengths (3 points)
    3. Areas for Improvement (3 points)
    4. Specific Recommendations
    
    Format your response in markdown with clear sections.
    """
    
    # Generate content synchronously
    response = model.generate_content(prompt)
    generated_text = response.text
    
    overall_score = calculate_overall_score(evaluations, communication_scores)
    
    report = {
        'technical_evaluation': evaluations,
        'communication_scores': communication_scores,
        'overall_score': overall_score,
        'feedback': generated_text
    }
    
    return report

async def analyze_interview(transcript, index_name):
    """Analyze the interview transcript"""
    # Create vector store
    vectorstore = create_vector_store(transcript, index_name)
    
    # Evaluate responses
    evaluation_criteria = {
        "technical_knowledge": "Understanding of core concepts",
        "problem_solving": "Ability to approach problems systematically",
        "experience": "Relevant work experience discussion"
    }
    
    # Execute tasks concurrently
    technical_scores, communication_scores = await asyncio.gather(
        evaluate_response(transcript, evaluation_criteria),
        analyze_communication_skills(transcript)
    )
    
    # Generate the report
    report = await generate_feedback_report(technical_scores, communication_scores)
    
    return technical_scores, communication_scores, report


def calculate_overall_score(evaluations, communication_scores):
    """Calculate overall candidate score"""
    technical_avg = sum(evaluations.values()) / len(evaluations)
    comm_avg = sum(communication_scores.values()) / len(communication_scores)
    return (technical_avg * 0.6) + (comm_avg * 0.4)  # Weighted average
