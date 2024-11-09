import streamlit as st
from FitFinder_logger import logger
from src.task1 import pre_process_documents, analyze_resume

def main():
    st.set_page_config(page_title="Resume Analysis", page_icon="📄", layout="wide")
    st.header("Resume Analysis Module")

    ALLOWED_EXTENSIONS = ['pdf', 'docx']

    with st.form(key="resume_form"):
        st.subheader("Upload Resumes")
        uploaded_files = st.file_uploader(
            "Upload resumes (PDF, DOCX)", 
            type=ALLOWED_EXTENSIONS, 
            accept_multiple_files=True
        )
        job_description = st.text_area("Enter the Job Description", height=200)
        submitted = st.form_submit_button("Submit")

        if submitted:
            if uploaded_files and job_description:
                try:
                    # Process uploaded resumes
                    with st.spinner("Processing resumes..."):
                        resumes_content, extracted_details = pre_process_documents(uploaded_files)

                    # Display extracted content
                    # st.subheader("Extracted Resume Content")
                    # st.write(resumes_content)

                    # # Display extracted details
                    # st.subheader("Extracted Resume Details")
                    # for detail in extracted_details:
                    #     st.write(detail)

                    # Generate summary and evaluation
                    with st.spinner("Analyzing resumes..."):
                        summary, evaluation = analyze_resume(resumes_content, job_description)

                    # Display results
                    st.subheader("Resume Summary")
                    st.write(summary)

                    st.subheader("Role Suitability Evaluation")
                    st.write(evaluation)

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
            else:
                st.error("Please upload resumes and enter the job description.")

if __name__ == "__main__":
    main()