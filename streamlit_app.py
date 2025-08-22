import streamlit as st
import requests
import tempfile
import os
import json
from PIL import Image
import time

# Configure Streamlit page
st.set_page_config(
    page_title="PoseRAC Exercise Counter",
    page_icon="🏋️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# FastAPI backend URL - detect if running locally or in container
def get_backend_url():
    # Check if we're running in a container or locally
    backend_url = os.getenv("BACKEND_URL")
    if backend_url:
        return backend_url
    
    # Try to detect the current host
    try:
        import socket
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        # Check if we can reach the backend on the local IP
        import requests
        test_url = f"http://{local_ip}:8000/health"
        response = requests.get(test_url, timeout=2)
        if response.status_code == 200:
            return f"http://{local_ip}:8000"
    except:
        pass
    
    # Default to localhost
    return "http://localhost:8000"

BACKEND_URL = get_backend_url()

def check_backend_health():
    """Check if the backend is running"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.status_code == 200, response.json()
    except requests.exceptions.RequestException as e:
        return False, {"error": str(e)}

def get_available_exercises():
    """Get list of available exercises from backend"""
    try:
        response = requests.get(f"{BACKEND_URL}/exercises", timeout=5)
        if response.status_code == 200:
            return response.json()["exercises"]
        return []
    except requests.exceptions.RequestException:
        return []

def analyze_video(video_file):
    """Send video to backend for analysis"""
    try:
        files = {"file": ("video.mp4", video_file, "video/mp4")}
        response = requests.post(f"{BACKEND_URL}/analyze_video", files=files, timeout=60)
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, {"error": f"Backend error: {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return False, {"error": f"Connection error: {str(e)}"}

def reset_real_time_state():
    """Reset real-time exercise counting state"""
    try:
        response = requests.post(f"{BACKEND_URL}/reset_real_time", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def get_real_time_status():
    """Get current real-time status"""
    try:
        response = requests.get(f"{BACKEND_URL}/real_time_status", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except requests.exceptions.RequestException:
        return None

def show_real_time_mode():
    """Show real-time camera mode interface"""
    st.header("Real-Time Exercise Counter")
    st.markdown("Use your webcam for real-time exercise recognition and counting!")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Camera Instructions
        
        **Real-time mode is available via WebSocket connection.**
        
        For the full real-time experience:
        1. Open your browser developer tools (F12)
        2. Use the JavaScript console to connect to the WebSocket
        3. Send camera frames to the backend
        
        **WebSocket Endpoint:** `ws://localhost:8000/ws/real_time`
        
        **Alternative:** Use our standalone real-time interface at:
        **http://localhost:8501** (separate Streamlit app)
        """)
        
        st.code("""
        // Example JavaScript code for WebSocket connection
        const ws = new WebSocket('ws://localhost:8000/ws/real_time');
        
        ws.onopen = function(event) {
            console.log('Connected to real-time exercise counter');
        };
        
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            console.log('Exercise:', data.exercise);
            console.log('Count:', data.count);
            console.log('Confidence:', data.confidence);
        };
        """, language="javascript")
    
    with col2:
        st.markdown("### Controls")
        
        if st.button("Reset Counter", type="secondary"):
            if reset_real_time_state():
                st.success("Counter reset!")
            else:
                st.error("Failed to reset counter")
        
        st.markdown("### Current Status")
        status = get_real_time_status()
        
        if status:
            st.metric("Exercise", status["exercise"].replace('_', ' ').title())
            st.metric("Count", status["count"])
            st.metric("Confidence", f"{status['confidence']:.1%}")
            st.metric("Buffer Size", status["buffer_size"])
        else:
            st.info("Connect to see real-time status")
        
        # Auto-refresh every 2 seconds
        time.sleep(2)
        st.rerun()
    
    st.markdown("---")
    st.markdown("""
    ### Technical Notes
    
    - **Real-time processing** uses WebSocket for low-latency communication
    - **Pose detection** powered by MediaPipe
    - **Exercise classification** using the trained PoseRAC model
    - **Repetition counting** with dual-threshold action triggers
    
    For the best experience, use the dedicated real-time interface or integrate 
    the WebSocket API into your own application.
    """)

def main():
    # Header
    st.title("PoseRAC Exercise Counter")
    st.markdown("Upload a video of your exercise and get an automatic repetition count!")
    
    # Mode selection
    mode = st.radio(
        "Choose Mode:",
        ["Video Upload", "Real-Time Camera"],
        horizontal=True
    )
    
    if mode == "Real-Time Camera":
        show_real_time_mode()
        return
    
    # Sidebar
    st.sidebar.title("System Status")
    
    # Check backend health
    health_status, health_data = check_backend_health()
    
    if health_status:
        st.sidebar.success("Backend is running")
        if "available_exercises" in health_data:
            st.sidebar.markdown("**Available Exercises:**")
            for exercise in health_data["available_exercises"]:
                st.sidebar.markdown(f"• {exercise.replace('_', ' ').title()}")
    else:
        st.sidebar.error("Backend is not responding")
        st.sidebar.markdown(f"Error: {health_data.get('error', 'Unknown error')}")
        st.error("Please make sure the FastAPI backend is running on port 8000")
        st.stop()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Upload Exercise Video")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video showing you performing an exercise"
        )
        
        if uploaded_file is not None:
            # Display video info
            st.success(f"Video uploaded: {uploaded_file.name}")
            st.info(f"File size: {uploaded_file.size / (1024*1024):.2f} MB")
            
            # Show video player
            st.video(uploaded_file)
            
            # Analyze button
            if st.button("Analyze Exercise", type="primary", use_container_width=True):
                # Show progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Uploading video...")
                progress_bar.progress(25)
                
                # Reset file pointer
                uploaded_file.seek(0)
                
                status_text.text("Processing video...")
                progress_bar.progress(50)
                
                # Analyze video
                success, result = analyze_video(uploaded_file)
                
                progress_bar.progress(100)
                status_text.text("Analysis complete!")
                
                # Display results
                if success:
                    st.balloons()
                    
                    with col2:
                        st.header("Results")
                        
                        if result["status"] == "success":
                            # Main results
                            st.metric(
                                label="Exercise Type",
                                value=result["exercise_type"].replace('_', ' ').title()
                            )
                            
                            st.metric(
                                label="Repetition Count",
                                value=result["repetition_count"]
                            )
                            
                            st.metric(
                                label="Confidence",
                                value=f"{result['confidence']:.2%}"
                            )
                            
                            st.metric(
                                label="Total Frames",
                                value=result["total_frames"]
                            )
                            
                            # Success message
                            if result["repetition_count"] > 0:
                                st.success(f"Great job! You completed {result['repetition_count']} {result['exercise_type'].replace('_', ' ')} repetitions!")
                            else:
                                st.warning("No repetitions detected. Try making sure your exercise movements are clear and visible.")
                                
                        else:
                            st.error(f"Analysis failed: {result.get('error_message', 'Unknown error')}")
                
                else:
                    st.error(f"Failed to analyze video: {result.get('error', 'Unknown error')}")
                
                # Clear progress indicators
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
    
    with col2:
        if uploaded_file is None:
            st.header("Instructions")
            st.markdown("""
            **How to use:**
            
            1. **Record a video** of yourself performing an exercise
            2. **Upload the video** using the file uploader
            3. **Click Analyze** to get automatic repetition counting
            
            **Tips for best results:**
            
            • Make sure your full body is visible
            • Good lighting helps pose detection
            • Perform exercises with clear movements
            • Videos should be 10-60 seconds long
            
            **Supported exercises:**
            """)
            
            exercises = get_available_exercises()
            for exercise in exercises:
                st.markdown(f"• {exercise.replace('_', ' ').title()}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Powered by PoseRAC - PyTorch-based Exercise Recognition"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()