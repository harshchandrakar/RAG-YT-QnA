import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from helper import common_language_codes, transcribe_extractor, format_docs
from dotenv import load_dotenv
import streamlit as st
import time
import os
# Load environment variables
load_dotenv()

def setup_google_api_key():
    """Setup Google API key for both local development and Streamlit Cloud"""
    try:
        # Try to get from Streamlit secrets first (for deployed app)
        api_key = st.secrets["GOOGLE_API_KEY"]
        os.environ["GOOGLE_API_KEY"] = api_key
    except:
        # For local development, it should already be in environment from .env
        # Just make sure it exists
        if not os.getenv("GOOGLE_API_KEY"):
            st.error("⚠️ GOOGLE_API_KEY not found! Please set it in Streamlit secrets or .env file")
            st.stop()

# Add this function call right after your imports and before initialize_components()
setup_google_api_key()

# Configure Streamlit page
st.set_page_config(
    page_title="🎥 YouTube Transcript Q&A",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #4ECDC4;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #4ECDC4;
    }
    
    .stSelectbox > div > div > select {
        border-radius: 10px;
        border: 2px solid #4ECDC4;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #4ECDC4;
        background-color: #f8f9fa;
    }
    
    .question-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .answer-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .error-box {
        background: #ffebee;
        border: 1px solid #f44336;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .success-box {
        background: #e8f5e8;
        border: 1px solid #4caf50;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize LangChain components
@st.cache_resource
def initialize_components():
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3
    )
    
    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.
        
        {context}
        Question: {question}
        """,
        input_variables=['context', 'question']
    )
    
    parser = StrOutputParser()
    
    return splitter, embeddings, llm, prompt, parser

# Initialize session state
def initialize_session_state():
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    if 'video_processed' not in st.session_state:
        st.session_state.video_processed = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_video_url' not in st.session_state:
        st.session_state.current_video_url = ""
    if 'last_error' not in st.session_state:
        st.session_state.last_error = None

def process_video(url, language_code, splitter, embeddings, llm, prompt, parser):
    """Process YouTube video and create QA chain"""
    try:
        # Clear any previous errors
        st.session_state.last_error = None
        
        with st.spinner("🎬 Extracting transcript..."):
            transcript = transcribe_extractor(url, language_code)
        
        if not transcript or len(transcript.strip()) < 50:
            raise Exception("Transcript is too short or empty. The video might not have captions.")
        
        with st.spinner("📄 Processing transcript..."):
            chunks = splitter.create_documents([transcript])
        
        if not chunks:
            raise Exception("Failed to create document chunks from transcript.")
        
        with st.spinner("🔍 Creating vector database..."):
            vector_store = FAISS.from_documents(chunks, embeddings)
            retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 4})
        
        with st.spinner("⚡ Setting up Q&A chain..."):
            parallel_chain = RunnableParallel({
                'context': retriever | RunnableLambda(format_docs),
                'question': RunnablePassthrough()
            })
            
            main_chain = parallel_chain | prompt | llm | parser
        
        return main_chain, transcript[:500] + "..." if len(transcript) > 500 else transcript
    
    except Exception as e:
        error_msg = str(e)
        st.session_state.last_error = error_msg
        st.error(f"❌ Error processing video: {error_msg}")
        
        # Provide helpful suggestions based on error type
        if "available languages" in error_msg.lower():
            st.info("💡 **Tip**: Try selecting a different language from the dropdown, or the video might have captions in a different language than expected.")
        elif "no captions" in error_msg.lower():
            st.info("💡 **Tip**: This video doesn't have captions/subtitles available. Try a different video that has captions enabled.")
        elif "invalid youtube url" in error_msg.lower():
            st.info("💡 **Tip**: Please check the YouTube URL format. It should be like: https://www.youtube.com/watch?v=VIDEO_ID")
        else:
            st.info("💡 **Tip**: Try a different video or check if the video has captions enabled. Some videos may not be accessible due to restrictions.")
        
        return None, None

def streamlit_main():
    # Header
    st.markdown('<div class="main-header">🎥 YouTube Transcript Q&A</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ask questions about any YouTube video transcript!</div>', unsafe_allow_html=True)
    
    # Initialize components and session state
    splitter, embeddings, llm, prompt, parser = initialize_components()
    initialize_session_state()
    
    # Sidebar for video input
    with st.sidebar:
        st.header("🎬 Video Settings")
        
        # Show last error if any
        if st.session_state.last_error:
            with st.expander("⚠️ Last Error Details", expanded=False):
                st.error(st.session_state.last_error)
        
        # YouTube URL input
        video_url = st.text_input(
            "YouTube URL:",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste the YouTube video URL here"
        )
        
        # Language selection
        language_display = st.selectbox(
            "Select Language:",
            options=list(common_language_codes.values()),
            index=0,
            help="Choose the language of the video"
        )
        
        # Get language code from display name
        language_code = [code for code, lang in common_language_codes.items() if lang == language_display][0]
        
        # Process video button
        if st.button("🚀 Process Video", type="primary"):
            if video_url:
                if video_url != st.session_state.current_video_url:
                    # Reset chat history for new video
                    st.session_state.chat_history = []
                    st.session_state.current_video_url = video_url
                
                qa_chain, transcript_preview = process_video(
                    video_url, language_code, splitter, embeddings, llm, prompt, parser
                )
                
                if qa_chain:
                    st.session_state.qa_chain = qa_chain
                    st.session_state.video_processed = True
                    st.success("✅ Video processed successfully!")
                    
                    # Show transcript preview
                    with st.expander("📄 Transcript Preview"):
                        st.text(transcript_preview)
                else:
                    st.session_state.video_processed = False
            else:
                st.warning("⚠️ Please enter a YouTube URL first!")
        
        # Reset button
        if st.button("🔄 Reset", help="Clear current video and chat history"):
            st.session_state.qa_chain = None
            st.session_state.video_processed = False
            st.session_state.chat_history = []
            st.session_state.current_video_url = ""
            st.session_state.last_error = None
            st.success("Reset completed!")
        
        # Troubleshooting section
        with st.expander("🔧 Troubleshooting Tips"):
            st.markdown("""
            **Common Issues:**
            - **No captions**: Video must have captions/subtitles
            - **Language mismatch**: Select correct video language
            - **Private videos**: Must be public videos
            - **Recent videos**: Very new videos might not have captions yet
            
            **Supported Video Types:**
            - Public YouTube videos with captions
            - Videos with auto-generated subtitles
            - Videos with manual captions in multiple languages
            """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat interface
        st.header("💬 Ask Questions")
        
        # Display chat history
        if st.session_state.chat_history:
            st.subheader("Chat History")
            for i, (question, answer) in enumerate(st.session_state.chat_history):
                st.markdown(f"""
                <div class="question-box">
                    <strong>🤔 Question {i+1}:</strong><br>
                    {question}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="answer-box">
                    <strong>🤖 Answer:</strong><br>
                    {answer}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
        
        # Question input
        if st.session_state.video_processed and st.session_state.qa_chain:
            question = st.text_input(
                "Your Question:",
                placeholder="Ask anything about the video...",
                key="question_input"
            )
            
            col_ask, col_clear = st.columns([3, 1])
            
            with col_ask:
                if st.button("🎯 Ask Question", type="primary"):
                    if question.strip():
                        with st.spinner("🤔 Thinking..."):
                            try:
                                answer = st.session_state.qa_chain.invoke(question)
                                st.session_state.chat_history.append((question, answer))
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error getting answer: {str(e)}")
                    else:
                        st.warning("⚠️ Please enter a question!")
            
            with col_clear:
                if st.button("🗑️ Clear Chat"):
                    st.session_state.chat_history = []
                    st.rerun()
        
        else:
            st.info("👈 Please process a video first using the sidebar!")
    
    with col2:
        # Stats and info
        st.header("📊 Session Info")
        
        if st.session_state.video_processed:
            st.metric("Video Status", "✅ Processed")
            st.metric("Questions Asked", len(st.session_state.chat_history))
            st.metric("Language", language_display)
            
            if st.session_state.current_video_url:
                st.info(f"🔗 Current Video:\n{st.session_state.current_video_url[:50]}...")
        else:
            st.metric("Video Status", "❌ Not Processed")
            st.info("Process a video to start asking questions!")
        
        # Tips
        with st.expander("💡 Tips for Better Results"):
            st.markdown("""
            - **Be specific**: Ask detailed questions about the content
            - **Use context**: Reference specific parts or topics from the video
            - **Multiple angles**: Ask follow-up questions for deeper understanding
            - **Check language**: Ensure the selected language matches the video
            """)
        
        # About
        with st.expander("ℹ️ About This App"):
            st.markdown("""
            This app uses:
            - **LangChain** for document processing
            - **Google Gemini** for AI responses
            - **FAISS** for vector search
            - **Streamlit** for the interface
            
            Simply paste a YouTube URL, select the language, and start asking questions!
            """)

if __name__ == "__main__":
    streamlit_main()