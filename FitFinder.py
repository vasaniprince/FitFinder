import streamlit as st

def main():
    
    st.set_page_config(page_title="FitFinder", page_icon="💼", layout="wide")

    with st.sidebar:
        st.markdown("The one-stop solution for the Candidate Analysis")

    # Header
    st.title("Welcome to FitFinder 💼")
    st.subheader("Your AI-powered Candidate Analysis", divider='rainbow')

    # Project Description
    st.header("About FitFinder")
    st.write("""
    FitFinder is an innovative tool designed to revolutionize how organizations evaluate candidates by leveraging cutting-edge AI technologies. 
    FitFinder offers a comprehensive solution for managing both Resume Analysis and Interview Analysis, ensuring that the recruitment process is more efficient, insightful, and streamlined.
    
   Our platform addresses common pain points in candidate evaluation, such as:
   - Lack of clear insights from resumes and video interviews
   - Inefficiency in summarizing relevant candidate information
   - Difficulty in providing feedback and actionable insights
   - Time-consuming manual assessments
   FitFinder solves these issues by providing a comprehensive, AI-driven solution that covers the entire candidate evaluation lifecycle.
    """)

    # Key Features
    st.header("Key Features 🌟", divider="rainbow")
    
    st.subheader("1. Resume Analysis Module", divider="gray")
    st.write("""
    - Enable users to upload resumes in formats like PDF and DOCX.
    - Use natural language processing (NLP) to extract essential information, including name, contact details, education, work experience, skills, and certifications. 
    """)

    st.subheader("2. Interview Analysis Module", divider="gray")
    st.write("""
   - Create a user-friendly interface for uploading interview videos.
   - Process the interview data without directly mentioning audio extraction or transcription to the user.
   """)

   #  st.subheader("3. Meeting Recording and Real-Time Tracking", divider="gray")
   #  st.write("""
   #  - High-quality audio and video recording capabilities
   #  - The speech-to-text transcription for accurate documentation
   #  - AI-powered topic tracking to match discussions with agenda items
   #  - Automatic flagging of unresolved issues.
   #  """)

   #  st.subheader("4. Comprehensive Post-Meeting Summary", divider="gray")
   #  st.write("""
   #  - AI-generated detailed summaries of the entire meeting
   #  - Extraction and highlighting of key decisions made during the meeting
   #  - Clear listing of assigned action items with responsible participants
   #  """)


    # Technologies Used
    st.header("Powered by Advanced Technologies 🔬", divider="rainbow")
    st.write("""
    FitFinder leverages state-of-the-art technologies to provide an unparalleled meeting management experience:

    1. **Large Language Models (LLMs)**:
       - Utilize advanced natural language processing for intelligent summarization
       - Extract key points, decisions, and action items with high accuracy
       - Generate meeting agendas and summaries

    2. **Retrieval-Augmented Generation (RAG)**:
       - Enhance LLM outputs with relevant context from your data
       - Improve the relevance and accuracy of AI-generated content

    3. **Vector Databases**:
       - Efficiently store and index meeting transcripts, summaries, and related documents
       - Enable semantic search capabilities for quick information retrieval

    4. **Speech-to-Text and Natural Language Understanding**:
       - Accurately transcribe meeting audio
       - Identify speakers and attribute statements correctly
       - Understand context, sentiment, and intent in spoken discussions

    """)



    # Footer
    st.markdown("---")
    st.write("© 2024 FitFinder. All rights reserved. 💼✨")

if __name__ == "__main__":
    main()