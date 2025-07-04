import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import time

from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import OpenAI
from langchain_google_vertexai import VertexAI
from langchain_google_genai import ChatGoogleGenerativeAI

# ✅ Load environment variables at the very start
load_dotenv()

# ✅ Set environment variables manually if needed
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Custom CSS for conversational layout
def load_custom_css():
    st.markdown("""
    <style>
    .user-message {
        color:#000;
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 18px 18px 4px 18px;
        margin: 10px 0px 10px 20%;
        border-left: 4px solid #2196f3;
    }
    
    .assistant-message {
        color:#000;
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 18px 18px 18px 4px;
        margin: 10px 20% 10px 0px;
        border-left: 4px solid #4caf50;
    }
    
    .message-header {
        font-weight: bold;
        font-size: 0.9em;
        margin-bottom: 8px;
        opacity: 0.8;
    }
    
    .user-message .message-header {
        color: #1976d2;
        text-align: right;
    }
    
    .assistant-message .message-header {
        color: #388e3c;
    }
    
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 10px;
    }
    
    .example-questions {
        color:#000;
        background-color: #fff3e0;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #ffb74d;
        margin: 15px 0;
    }
    
    .stButton > button {
        color:#000;
        width: 100%;
        margin: 5px 0;
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 8px 12px;
        text-align: left;
    }
    
    .stButton > button:hover {
        background-color: #e9ecef;
        border-color: #adb5bd;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to get the selected model
def get_llm(model_choice):
    if model_choice == "OpenAI":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
        return OpenAI(temperature=0, api_key=api_key)
    
    elif model_choice == "VertexAI":
        credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not credentials_path:
            raise ValueError("Google credentials not found. Please set GOOGLE_APPLICATION_CREDENTIALS in your .env file.")
        return VertexAI(model_name="gemini-pro", temperature=0.2)
    
    elif model_choice == "Gemini":
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key not found. Please set GOOGLE_API_KEY in your .env file.")
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            temperature=0, 
            google_api_key=api_key
        )
    else:
        raise ValueError("Unsupported model selected.")

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "df" not in st.session_state:
        st.session_state.df = None
    if "model_choice" not in st.session_state:
        st.session_state.model_choice = "OpenAI"

def load_data():
    """Load the dataset"""
    try:
        df = pd.read_csv("supermarket_sales.csv")
        st.session_state.df = df
        return df
    except FileNotFoundError:
        st.error("⚠️ 'supermarket_sales.csv' not found in the project directory.")
        return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def create_agent(model_choice, df):
    """Create the pandas dataframe agent"""
    try:
        llm = get_llm(model_choice)
        agent = create_pandas_dataframe_agent(
            llm, 
            df, 
            verbose=True, 
            allow_dangerous_code=True,
            handle_parsing_errors=True
        )
        return agent
    except Exception as e:
        st.error(f"❌ Model initialization failed: {e}")
        return None

def display_message(message, role):
    """Display a message with conversational styling"""
    timestamp = time.strftime("%H:%M", time.localtime(message.get("timestamp", time.time())))
    
    if role == "user":
        st.markdown(f"""
        <div class="user-message">
            <div class="message-header">You • {timestamp}</div>
            {message["content"]}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="assistant-message">
            <div class="message-header">🤖 Lane Analytics • {timestamp}</div>
            {message["content"]}
        </div>
        """, unsafe_allow_html=True)

def get_initial_analysis(df):
    """Generate initial analysis of the dataset"""
    analysis = f"""
🎯 **Dataset Overview:**
- **Total Records:** {len(df):,}
- **Columns:** {len(df.columns)}
- **Date Range:** {df['Date'].min() if 'Date' in df.columns else 'N/A'} to {df['Date'].max() if 'Date' in df.columns else 'N/A'}

📊 **Key Insights:**
- **Total Sales:** ${df['Total'].sum():,.2f} (if 'Total' column exists)
- **Average Transaction:** ${df['Total'].mean():.2f} (if 'Total' column exists)
- **Top Product Lines:** {', '.join(df['Product line'].value_counts().head(3).index.tolist()) if 'Product line' in df.columns else 'N/A'}

💡 **What would you like to explore?**
- Sales trends and patterns
- Customer behavior analysis  
- Product performance insights
- Branch/location comparisons
- Payment method preferences

Just ask me anything about your data! 🚀
"""
    return analysis

def main():
    st.set_page_config(
        page_title="Lane Analytics Chatbot", 
        page_icon="🤖",
        layout="wide"
    )
    
    # Load custom CSS
    load_custom_css()
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("🤖 Lane Analytics Assistant")
    st.markdown("*Your intelligent data analysis companion*")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_choice = st.selectbox(
            "Choose AI Model:", 
            ["OpenAI", "VertexAI", "Gemini"],
            index=["OpenAI", "VertexAI", "Gemini"].index(st.session_state.model_choice)
        )
        
        # Update model if changed
        if model_choice != st.session_state.model_choice:
            st.session_state.model_choice = model_choice
            st.session_state.agent = None  # Reset agent
            st.rerun()
        
        st.markdown("---")
        
        # Data loading section
        st.header("📊 Data Status")
        
        if st.button("🔄 Load/Reload Data"):
            with st.spinner("Loading data..."):
                df = load_data()
                if df is not None:
                    st.success(f"✅ Data loaded: {len(df):,} rows")
                    # Reset conversation when new data is loaded
                    st.session_state.messages = []
                    st.session_state.agent = None
        
        # Show data info if loaded
        if st.session_state.df is not None:
            df = st.session_state.df
            st.write(f"**Rows:** {len(df):,}")
            st.write(f"**Columns:** {len(df.columns)}")
            
            with st.expander("📋 Column Names"):
                for col in df.columns:
                    st.write(f"• {col}")
        
        st.markdown("---")
        
        # Quick actions
        st.header("🚀 Quick Actions")
        if st.button("🔄 Clear Chat"):
            st.session_state.messages = []
            st.rerun()
        
        if st.button("📈 Get Data Summary"):
            if st.session_state.df is not None:
                summary_msg = get_initial_analysis(st.session_state.df)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": summary_msg,
                    "timestamp": time.time()
                })
                st.rerun()

    # Main chat interface
    if st.session_state.df is None:
        st.info("👋 Welcome! Please load your data using the sidebar to get started.")
        return
    
    # Initialize agent if not exists
    if st.session_state.agent is None:
        with st.spinner(f"Initializing {st.session_state.model_choice} model..."):
            agent = create_agent(st.session_state.model_choice, st.session_state.df)
            if agent:
                st.session_state.agent = agent
                st.success(f"✅ {st.session_state.model_choice} model ready!")
                
                # Add welcome message if this is the first time
                if not st.session_state.messages:
                    welcome_msg = f"""
👋 **Hello! I'm your Lane Analytics Assistant powered by {st.session_state.model_choice}.**

I've loaded your supermarket sales data and I'm ready to help you analyze it! 

{get_initial_analysis(st.session_state.df)}
"""
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": welcome_msg,
                        "timestamp": time.time()
                    })
            else:
                st.error("Failed to initialize the model. Please check your API keys.")
                return

    # Chat interface with conversational layout
    st.markdown("---")
    
    # Display chat messages with conversational styling
    chat_container = st.container()
    
    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for message in st.session_state.messages:
            display_message(message, message["role"])
    st.markdown('</div>', unsafe_allow_html=True)

    
    # Example questions (only show if no messages yet or user wants suggestions)
    if len(st.session_state.messages) <= 1:
        st.markdown("""
        <div class="example-questions">
            <h4>💡 Try asking me:</h4>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        example_questions = [
            "What's the total sales by city?",
            "Which product line performs best?",
            "Show me sales trends over time",
            "What's the average customer rating?",
            "Which payment method is most popular?",
            "Compare branch performances"
        ]
        
        for i, question in enumerate(example_questions):
            col = [col1, col2, col3][i % 3]
            with col:
                if st.button(f"💬 {question}", key=f"example_{i}", use_container_width=True):
                    # Add user question to messages
                    st.session_state.messages.append({
                        "role": "user", 
                        "content": question,
                        "timestamp": time.time()
                    })
                    st.rerun()

    # Chat input at the bottom
    if prompt := st.chat_input("Ask me anything about your data..."):
        # Add user message
        user_message = {
            "role": "user", 
            "content": prompt,
            "timestamp": time.time()
        }
        st.session_state.messages.append(user_message)
        
        # Generate assistant response
        with st.spinner("🤔 Analyzing your data..."):
            try:
                response = st.session_state.agent.run(prompt)
                
                # Add assistant response to messages
                assistant_message = {
                    "role": "assistant", 
                    "content": response,
                    "timestamp": time.time()
                }
                st.session_state.messages.append(assistant_message)
                
            except Exception as e:
                error_msg = f"❌ Sorry, I encountered an error: {str(e)}\n\nTry rephrasing your question or ask me something else!"
                
                assistant_message = {
                    "role": "assistant", 
                    "content": error_msg,
                    "timestamp": time.time()
                }
                st.session_state.messages.append(assistant_message)
        
        # Rerun to display the new messages
        st.rerun()

    # Footer
    st.markdown("---")
    st.markdown("*💬 Having a conversation with your data - Powered by LangChain & Streamlit*")

if __name__ == "__main__":
    main()