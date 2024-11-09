import streamlit as st
import tempfile
import os
import asyncio
from src.task2 import (
    process_video,
    create_vector_store,
    evaluate_response,
    analyze_communication_skills,
    generate_feedback_report
)
import plotly.graph_objects as go
from dotenv import load_dotenv
import cv2

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(page_title="Interview Analyzer", layout="wide")

def create_radar_chart(scores, title):
    """Create a radar chart for scores visualization"""
    categories = list(scores.keys())
    values = list(scores.values())
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10])
        ),
        showlegend=False,
        title=title
    )
    return fig

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
    
    # Use asyncio.gather to run coroutines concurrently
    technical_scores, communication_scores = await asyncio.gather(
        evaluate_response(transcript, evaluation_criteria),
        analyze_communication_skills(transcript)
    )
    
    # Generate the report after getting both scores
    report = await generate_feedback_report(technical_scores, communication_scores)
    
    return technical_scores, communication_scores, report

async def process_interview(transcript, index_name):
    """Wrapper function to handle the interview analysis process"""
    try:
        return await analyze_interview(transcript, index_name)
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return None, None, None

def process_video_with_memory_management(video_path):
    """Process video with memory-efficient approach"""
    try:
        # Set OpenCV memory buffer size
        cv2.setNumThreads(1)  # Reduce number of threads
        
        # Open video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Error opening video file")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Process frames in chunks
        chunk_size = 100  # Adjust this based on your memory constraints
        frames = []
        
        for i in range(0, total_frames, chunk_size):
            chunk_frames = []
            for j in range(chunk_size):
                if i + j >= total_frames:
                    break
                    
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Resize frame to reduce memory usage
                frame = cv2.resize(frame, (640, 480))
                chunk_frames.append(frame)
            
            # Process chunk
            frames.extend(chunk_frames)
            
            # Clear memory
            del chunk_frames
        
        cap.release()
        return frames
        
    except Exception as e:
        raise Exception(f"Video processing error: {str(e)}")

def main():
    st.title("AI-Powered Interview Analysis System")
    
    with st.sidebar:
        st.header("Configuration")
        index_name = st.text_input("Pinecone Index Name", "interview-index")
    
    # File upload section
    st.header("Upload Interview Video")
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        try:
            # Create temporary file with a specific cleanup strategy
            temp_dir = tempfile.mkdtemp()
            video_path = os.path.join(temp_dir, "temp_video.mp4")
            
            # Write uploaded file to temporary location
            with open(video_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            
            with st.spinner("Processing interview..."):
                try:
                    # Process video with memory management
                    frames = process_video_with_memory_management(video_path)
                    
                    # Get transcript (assuming this is implemented in your task2 module)
                    transcript = process_video(video_path)[1]
                    
                    if transcript:
                        st.success("Video processed successfully!")
                        
                        # Create a new event loop for the async operations
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            # Run the async analysis
                            results = loop.run_until_complete(analyze_interview(transcript, index_name))
                            technical_scores, communication_scores, report = results
                        finally:
                            loop.close()
                        
                        if technical_scores and communication_scores and report:
                            # Display results in tabs
                            tab1, tab2, tab3 = st.tabs(["Scores", "Transcript", "Recommendations"])
                            
                            with tab1:
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    technical_chart = create_radar_chart(
                                        technical_scores,
                                        "Technical Evaluation"
                                    )
                                    st.plotly_chart(technical_chart)
                                
                                with col2:
                                    comm_chart = create_radar_chart(
                                        communication_scores,
                                        "Communication Skills"
                                    )
                                    st.plotly_chart(comm_chart)
                                
                                # Overall Score
                                st.subheader("Overall Score")
                                score = report['overall_score']
                                st.progress(score / 10)
                                st.metric("Final Score", f"{score:.2f}/10")
                            
                            with tab2:
                                st.subheader("Interview Transcript")
                                st.write(transcript)
                            
                            with tab3:
                                st.subheader("Analysis & Recommendations")
                                st.markdown(report['feedback'])
                            
                            # Download report option
                            st.download_button(
                                label="Download Full Report",
                                data=str(report),
                                file_name="interview_analysis_report.txt",
                                mime="text/plain"
                            )
                        
                    else:
                        st.error("Could not process the audio from the video. Please ensure the video has clear audio.")
                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                
                finally:
                    # Proper cleanup with error handling
                    try:
                        if os.path.exists(video_path):
                            # Make sure file is not in use
                            cv2.destroyAllWindows()
                            # Remove the file
                            os.remove(video_path)
                        # Remove the temporary directory
                        os.rmdir(temp_dir)
                    except Exception as e:
                        st.warning(f"Cleanup warning: {str(e)}")
        
        except Exception as e:
            st.error(f"File handling error: {str(e)}")

if __name__ == "__main__":
    main()