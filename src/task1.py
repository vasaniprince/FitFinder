import PyPDF2
import re
from typing import Dict, List, Optional
from FitFinder_logger import logger
from docx import Document
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document as LangchainDocument
import os
from dotenv import load_dotenv

load_dotenv()

def clean_text(text):
    """
    Cleans the extracted text by removing unwanted characters and formatting.
    """
    cleaned_text = re.sub(r'[\n\r\t\f\v]', ' ', text)
    cleaned_text = re.sub(r'[^\x20-\x7E]', '', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    return cleaned_text.strip()

def extract_text_from_pdf(pdf_file):
    """
    Extracts and cleans text from a PDF file.
    """
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return clean_text(text)

def extract_text_from_docx(docx_file):
    """
    Extracts and cleans text from a DOCX file.
    """
    doc = Document(docx_file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return clean_text(text)



def extract_resume_details(text: str) -> Dict:
    """
    Extracts essential details from resume text using NLP techniques.
    
    Args:
        text (str): The input resume text
        
    Returns:
        Dict: Dictionary containing extracted resume information
    """
    # Clean the text
    text = text.replace('\r', '').strip()
    
    # Name extraction (improved to handle multiple name formats)
    name_pattern = r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
    name_match = re.search(name_pattern, text)
    name = name_match.group() if name_match else "Not found"
    
    # Phone number extraction (handles international formats)
    phone_patterns = [
        r'(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',  # International
        r'(\+91|91|0)?[-\s]?\d{10}',  # Indian
        r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}'  # US
    ]
    phone_number = "Not found"
    for pattern in phone_patterns:
        match = re.search(pattern, text)
        if match:
            phone_number = match.group().strip()
            break
    
    # Email extraction (improved pattern)
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    email = re.search(email_pattern, text, re.I)
    email = email.group() if email else "Not found"
    
    # Education extraction (expanded patterns)
    education_patterns = [
        r'(?:Bachelor|B\.?[A-Za-z]*|Master|M\.?[A-Za-z]*|Ph\.?D\.?)[\s\w]*(?:of|in|\.)?[\s\w]*(?:\([^)]*\))?',
        r'(?:B\.Tech|B\.E|M\.Tech|M\.E|BBA|MBA|BCA|MCA)(?:\s+\([^)]*\))?'
    ]
    education = []
    for pattern in education_patterns:
        matches = re.finditer(pattern, text, re.I)
        education.extend(match.group().strip() for match in matches)
    
    # Skills extraction (expanded list and improved detection)
    skills_keywords = {
        'Programming': ['Python', 'Java', 'C++', 'JavaScript', 'R', 'SQL', 'PHP', 'Ruby'],
        'Data Science': ['Machine Learning', 'Deep Learning', 'NLP', 'Data Analysis', 
                        'Statistical Analysis', 'Data Visualization', 'Big Data'],
        'Frameworks': ['TensorFlow', 'PyTorch', 'Keras', 'scikit-learn', 'Django', 'Flask'],
        'Tools': ['Git', 'Docker', 'AWS', 'Azure', 'GCP', 'Linux', 'JIRA'],
        'Libraries': ['Pandas', 'NumPy', 'Matplotlib', 'Seaborn', 'Beautiful Soup', 'NLTK']
    }
    
    skills = {}
    for category, keywords in skills_keywords.items():
        found_skills = [skill for skill in keywords 
                       if re.search(r'\b' + re.escape(skill) + r'\b', text, re.I)]
        if found_skills:
            skills[category] = found_skills
    
    # Certification extraction (improved pattern)
    cert_patterns = [
        r'(?:Certification|Certificate|Certified|Licensed):?\s*([^.\n]*)',
        r'(?:Certification|Certificate|Certified|Licensed)\s+in\s+([^.\n]*)'
    ]
    certifications = []
    for pattern in cert_patterns:
        matches = re.finditer(pattern, text, re.I)
        certifications.extend(match.group().strip() for match in matches)
    
    # Work experience extraction (new feature)
    experience_pattern = r'(?:(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\s*(?:-|to)\s*(?:Present|Current|Now|(?:(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})))'
    experience_dates = re.findall(experience_pattern, text, re.I)
    
    return {
        "name": name,
        "contact": {
            "phone": phone_number,
            "email": email
        },
        "education": education,
        "skills": skills,
        "certifications": certifications,
        "experience_dates": experience_dates
    }


def analyze_resume(resumes_content, job_description):
    """
    Analyzes resumes using Gemini LLM and vector store
    """
    google_api_key = os.getenv("GOOGLE_API_KEY")
    llm = GoogleGenerativeAI(model="gemini-1.5-flash", api_key=google_api_key)
    
    # Create vector store from resumes
    # vector_store = initialize_vector_store([resumes_content])
    
    # Generate summary prompt
    summary_template = PromptTemplate.from_template("""
        Summarize the following resume details concisely:
        {resume_content}
    """)
    
    # Generate evaluation prompt
    evaluation_template = PromptTemplate.from_template("""
        Based on the job description:
        {job_description}
        
        And the following resume details:
        {resume_content}
        
        Evaluate the candidate's suitability for the role. 
        Summarize their strengths and areas for improvement.
    """)
    
    # Generate summary
    summary_chain = summary_template | llm
    summary = summary_chain.invoke({"resume_content": resumes_content})
    
    # Generate evaluation
    evaluation_chain = evaluation_template | llm
    evaluation = evaluation_chain.invoke({
        "job_description": job_description,
        "resume_content": resumes_content
    })
    
    return summary, evaluation

def pre_process_documents(uploaded_files):
    """
    Pre-processes uploaded resumes
    """
    documents_content = ""
    extracted_details = []
    
    for uploaded_file in uploaded_files:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'pdf':
            file_text = extract_text_from_pdf(uploaded_file)
        elif file_extension == 'docx':
            file_text = extract_text_from_docx(uploaded_file)
        else:
            file_text = "Unsupported file format."
        
        documents_content += file_text
        extracted_details.append(extract_resume_details(file_text))
    
    return documents_content, extracted_details