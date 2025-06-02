import os
import re
import json
from typing import List, Dict, Tuple, Optional
import faiss
import numpy as np
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from youtube_transcript_api import YouTubeTranscriptApi
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure API keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Please set GOOGLE_API_KEY in your environment variables or .env file")
    st.stop()

# Constants
MAX_TOKENS = 10000
MODEL_NAME = "gemini-1.5-flash-latest"
EMBEDDING_MODEL = "models/embedding-001"
CACHE_DIR = "vector_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Initialize models
llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.3, convert_system_message_to_human=True)
embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

# Pydantic models
class VideoSegment(BaseModel):
    text: str
    start: float
    duration: float

class VideoSummary(BaseModel):
    summary: str
    segments: List[VideoSegment]

# Utility functions
def extract_video_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from URL."""
    patterns = [
        r"(?:https?:\/\/)?(?:www\.)?youtu\.?be(?:\.com)?\/(?:watch\?v=)?([^&?\n]+)",
        r"youtu\.be\/([^&?\n]+)",
        r"embed\/([^&?\n]+)"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_transcript(video_id: str) -> Optional[List[Dict]]:
    """Fetch YouTube transcript with fallback to English."""
    try:
        # First try English
        try:
            return YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US'])
        except:
            # Fallback to any available transcript
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            for transcript in transcript_list:
                if transcript.is_generated:
                    return transcript.fetch()
            return None
    except Exception as e:
        st.error(f"Error fetching transcript: {str(e)}")
        return None

def get_cache_path(video_id: str) -> str:
    """Get path for cached data."""
    return os.path.join(CACHE_DIR, f"{video_id}")

def save_full_cache(video_id: str, data: dict):
    """Save all processed data to cache."""
    cache_path = get_cache_path(video_id)
    # Save FAISS vectors
    data['vector_store'].save_local(cache_path)
    # Save other data as JSON
    with open(f"{cache_path}_data.json", "w") as f:
        json.dump({
            'transcript': data['transcript'],
            'segments': [seg.dict() for seg in data['segments']],
            'summary': data['summary'].dict()
        }, f)

def load_full_cache(video_id: str) -> Optional[dict]:
    """Load all processed data from cache."""
    cache_path = get_cache_path(video_id)
    if not os.path.exists(f"{cache_path}_data.json"):
        return None
        
    try:
        # Load FAISS
        vector_store = FAISS.load_local(cache_path, embeddings, allow_dangerous_deserialization=True)
        
        # Load other data
        with open(f"{cache_path}_data.json", "r") as f:
            data = json.load(f)
        
        return {
            'vector_store': vector_store,
            'transcript': data['transcript'],
            'segments': [VideoSegment(**seg) for seg in data['segments']],
            'summary': VideoSummary(**data['summary'])
        }
    except Exception as e:
        st.warning(f"Cache loading failed: {str(e)}")
        return None

def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def create_segments(transcript: List[Dict], chunk_size: int = 5) -> List[VideoSegment]:
    """Create video segments from transcript."""
    segments = []
    current_segment = []
    current_start = 0
    
    for i, entry in enumerate(transcript):
        if not current_segment:
            current_start = entry['start']
        
        current_segment.append(entry['text'])
        
        # Create segment every chunk_size entries or at end of transcript
        if (i + 1) % chunk_size == 0 or i == len(transcript) - 1:
            segment_text = ' '.join(current_segment)
            duration = entry['start'] + entry['duration'] - current_start
            segments.append(VideoSegment(
                text=segment_text,
                start=current_start,
                duration=duration
            ))
            current_segment = []
    
    return segments

def generate_summary(segments: List[VideoSegment]) -> VideoSummary:
    """Generate summary of video content with timestamps."""
    combined_text = "\n\n".join(
        f"[{format_timestamp(seg.start)}] {seg.text}"
        for seg in segments
    )
    
    # Split text if too large
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_TOKENS,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(combined_text)
    
    # Generate summary for each chunk
    summary_parts = []
    for chunk in chunks:
        prompt = f"""
        Summarize this video transcript with key points and timestamps:
        
        {chunk}
        
        Provide clear timestamps and concise descriptions.
        """
        
        response = llm.invoke(prompt)
        summary_parts.append(response.content)
    
    full_summary = "\n\n".join(summary_parts)
    
    return VideoSummary(
        summary=full_summary,
        segments=segments
    )

def create_vector_store(segments: List[VideoSegment]) -> FAISS:
    """Create FAISS vector store from video segments."""
    texts = [seg.text for seg in segments]
    metadatas = [{"start": seg.start, "duration": seg.duration} for seg in segments]
    
    # Split longer texts
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    docs = []
    for i, text in enumerate(texts):
        splits = text_splitter.split_text(text)
        for split in splits:
            docs.append({
                "text": split,
                "metadata": {
                    "start": metadatas[i]["start"],
                    "duration": metadatas[i]["duration"]
                }
            })
    
    # Create vector store
    db = FAISS.from_texts(
        [doc["text"] for doc in docs],
        embeddings,
        metadatas=[doc["metadata"] for doc in docs]
    )
    
    return db

def get_context_from_vector_store(db: FAISS, query: str, k: int = 3) -> List[Tuple[str, Dict]]:
    """Retrieve relevant context from vector store."""
    docs = db.similarity_search_with_score(query, k=k)
    return [(doc.page_content, doc.metadata) for doc, _ in docs]

def format_context(contexts: List[Tuple[str, Dict]]) -> str:
    """Format context for prompt."""
    formatted = []
    for text, metadata in contexts:
        timestamp = format_timestamp(metadata["start"])
        formatted.append(f"[{timestamp}] {text}")
    return "\n\n".join(formatted)

def answer_question(db: FAISS, question: str, chat_history: List) -> str:
    """Generate answer to question based on video content."""
    # Retrieve relevant context
    contexts = get_context_from_vector_store(db, question)
    
    if not contexts:
        return "Sorry, I couldn't find relevant information in the video to answer your question."
    
    # Format chat history (last 5 exchanges)
    history_str = ""
    for msg in chat_history[-10:]:  # Keep last 10 messages for context
        if isinstance(msg, HumanMessage):
            history_str += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            history_str += f"Assistant: {msg.content}\n"
    
    # Format prompt to ask for responses without timestamps
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful YouTube video assistant. Answer questions based only on the provided video content.
         If the question cannot be answered from the video, politely say so.
         
         Video Context:
         {context}
         
         Chat History:
         {history}
         
         Current Question: {question}
         
         Provide a detailed answer but DO NOT include any timestamps in your response.
         Focus on answering the question directly using the video content."""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])
    
    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_context(contexts),
            history=lambda x: x.get("chat_history", [])
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    response = chain.invoke({
        "question": question,
        "chat_history": chat_history
    })
    
    return response

def process_video(url: str):
    """Handle the complete video processing pipeline."""
    video_id = extract_video_id(url)
    if not video_id:
        st.error("Invalid YouTube URL")
        return None, None
    
    # Check if already processed
    if st.session_state.get('current_video_id') == video_id:
        return st.session_state.summary, st.session_state.vector_store
    
    # Try to load from cache
    with st.spinner("Checking cache..."):
        cache_data = load_full_cache(video_id)
        if cache_data:
            st.session_state.summary = cache_data['summary']
            st.session_state.vector_store = cache_data['vector_store']
            st.session_state.current_video_id = video_id
            return cache_data['summary'], cache_data['vector_store']
    
    # Process fresh data
    with st.spinner("Fetching and processing video..."):
        transcript = get_transcript(video_id)
        if not transcript:
            st.error("Could not fetch transcript for this video.")
            return None, None
        
        segments = create_segments(transcript)
        summary = generate_summary(segments)
        vector_store = create_vector_store(segments)
        
        # Save to cache
        save_full_cache(video_id, {
            'vector_store': vector_store,
            'transcript': transcript,
            'segments': segments,
            'summary': summary
        })
        
        # Update session state
        st.session_state.summary = summary
        st.session_state.vector_store = vector_store
        st.session_state.current_video_id = video_id
        
        return summary, vector_store

# Streamlit UI Configuration
st.set_page_config(page_title="YouTube Video Assistant", page_icon="📺", layout="wide")

# Initialize session state
if "current_video_id" not in st.session_state:
    st.session_state.current_video_id = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for video input
with st.sidebar:
    st.title("YouTube Video Input")
    youtube_url = st.text_input("Enter YouTube Video URL:", key="video_url")
    
    if youtube_url:
        video_id = extract_video_id(youtube_url)
        if video_id:
            st.video(f"https://www.youtube.com/watch?v={video_id}")
            summary, vector_store = process_video(youtube_url)
            
            if summary and vector_store:
                st.success("Video processed successfully!")

# Main content area
tab1, tab2 = st.tabs(["Chatbot", "Video Summary"])

with tab1:
    st.title("📺 YouTube Video Chatbot")
    st.write("Ask questions about the video content")
    
    if st.session_state.vector_store:
        # Display chat messages
        for message in st.session_state.messages[-10:]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about the video..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.chat_history.append(HumanMessage(content=prompt))
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.spinner("Thinking..."):
                response = answer_question(
                    st.session_state.vector_store,
                    prompt,
                    st.session_state.chat_history
                )
                
                # Add AI response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.chat_history.append(AIMessage(content=response))
                
                # Display AI response
                with st.chat_message("assistant"):
                    st.markdown(response)
    else:
        st.info("Enter a YouTube URL in the sidebar to start chatting about the video")

with tab2:
    if st.session_state.summary:
        st.title("🎬 Video Summary")
        st.subheader("Key Points")
        st.markdown(st.session_state.summary.summary)
        
        st.subheader("Detailed Timestamps")
        for segment in st.session_state.summary.segments:
            with st.expander(f"{format_timestamp(segment.start)} - {segment.text[:100]}..."):
                st.write(segment.text)
    else:
        st.info("Enter a YouTube URL in the sidebar to view the video summary")
