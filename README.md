<br />
<p align="center">
	<img src="./assets/logo.jpg" alt="Logo" width="400">
	<h1 align="center">Candidate Evaluation Tool</h1>

  <p align="center">
    AI-Powered Resume and Interview Analysis Platform
    <br/>
  </p>
</p>

This tool simplifies candidate evaluation by integrating generative AI with traditional development techniques. It automates resume analysis and video interview evaluation, offering insights into a candidate's suitability for a role. Built using Python, LangChain, Streamlit, and other key technologies, the project provides a seamless and efficient workflow for managing every stage of the recruitment process.

<!-- TABLE OF CONTENTS -->
## Table of Contents
<details open="open">
  <ol>
    <li>
      <a href="#Objective-and-Goal">Objective and Goal</a>
    </li>
    <li>
      <a href="#Setup-Instructions">Setup Instructions</a>
    </li>
    <li>
      <a href="#Usage-Guidelines">Usage Guidelines</a>
    </li>
    <li>
      <a href="#System-Architecture">System Architecture</a>
    </li>
    <li>
      <a href="#Dependencies">Dependencies</a>
    </li>
  </ol>
</details>

## Objective and Goal

**Objective**: Build a tool that helps evaluate candidates more efficiently, focusing on two main areas: **Resume Analysis** and **Interview Analysis**. The tool should use advanced AI to make the evaluation process faster and more accurate, giving insights into how well a candidate fits a particular job.

**Goal**:

- Automate the extraction and summarization of important details from resumes to provide a clear recommendation on whether the candidate is suitable for the job.
- Analyze video interviews using Retrieval-Augmented Generation (RAG) to assess the candidate’s answers for accuracy and communication skills, offering detailed feedback.

## Setup Instructions

### Prerequisites
- **Python**: Make sure you have Python 3.11.3 installed on your machine.
- **API Keys**: You need the following API keys:
  - **Gemini LLM**: Get this from [Gemini API](https://ai.google.dev/).
  - **Pinecone Vector Database**: Create an account on [Pinecone](https://www.pinecone.io/) and obtain your API key.

### Environment Setup
1. **Extract the Code**:
  - Download the zip file of the project and extract it to your desired location.
  - Navigate to the project directory:

    ```bash
    cd <extracted-directory>
    ```

2. **Create a Virtual Environment**:
   ```bash
   python3.11 -m venv myenv
   source myenv/bin/activate    # On Windows: myenv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   You may need to add the following line to download the spaCy model:
   python -m spacy download en_core_web_sm
   ```

4. **Set up the environment variables**:
   Create a `.env` file in the root directory with the following contents:
   ```bash
   GOOGLE_API_KEY=<your_gemini_api_key>
   PINECONE_API_KEY=<your_pinecone_api_key>
   ```

5. **Run the Application**:
   ```bash
   streamlit run FitFinder.py
   ```

## Usage Guidelines

**Step 1: Resume Analysis**
- Upload documents such as PDFs, Docs of Resume.
- Write the Job Description.

**Step 2: Interview Analysis**
- Upload the interview Video.
- Without any clicks, wait for a few minutes to receive a scores,Transcript,Recommendation for that particular candidate

## System Architecture

This tool integrates several AI-powered and traditional components to manage the full lifecycle of candidate analysis. Here's a brief overview:

1. **Resume Analysis**:
   - Uploaded Resume are converted into text, cleaned, and stored.
   - Then using the googel gemini that provide a candidate resume summary.

2. **Interview Analysis**:
   - Convert video into text and then text are divided into chunks, converted to embeddings, and stored in Pinecone for efficient retrieval.
   - The system finds the top 3 related chunks for generate a recommendation for the candidate.

## Dependencies

Ensure you have all the required dependencies by installing them through `requirements.txt`. Here's an outline of key packages:
- Python 3.11.3
- Langchain
- Whisper
- Streamlit
- Hugging Face (for embeddings)
- Pinecone for vector database
- Gemini LLM via API

Install dependencies:
```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---
