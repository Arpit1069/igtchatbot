import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import time
import io

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
            model="gemini-2.0-flash-exp", 
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
    if "kpi_df" not in st.session_state:
        st.session_state.kpi_df = None
    if "model_choice" not in st.session_state:
        st.session_state.model_choice = "OpenAI"
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = {}

def load_file(uploaded_file):
    """Load CSV or Excel file"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            # For Excel files, try to load all sheets
            excel_file = pd.ExcelFile(uploaded_file)
            if len(excel_file.sheet_names) == 1:
                df = pd.read_excel(uploaded_file)
            else:
                # If multiple sheets, let user choose or combine
                sheets_data = {}
                for sheet in excel_file.sheet_names:
                    sheets_data[sheet] = pd.read_excel(uploaded_file, sheet_name=sheet)
                
                # For now, use the first sheet as main data
                df = sheets_data[excel_file.sheet_names[0]]
                
                # Store all sheets for reference
                st.session_state.uploaded_files[uploaded_file.name] = sheets_data
        else:
            st.error("Unsupported file format. Please upload CSV or Excel files.")
            return None
        
        return df
    except Exception as e:
        st.error(f"Error loading file {uploaded_file.name}: {e}")
        return None

def analyze_kpi_file(df):
    """Analyze uploaded file to identify KPI-related columns"""
    kpi_keywords = [
        'kpi', 'metric', 'target', 'goal', 'objective', 'priority', 
        'weight', 'importance', 'factor', 'score', 'rating', 'performance',
        'benchmark', 'threshold', 'critical', 'key'
    ]
    
    kpi_columns = []
    for col in df.columns:
        if any(keyword in col.lower() for keyword in kpi_keywords):
            kpi_columns.append(col)
    
    analysis = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'kpi_columns': kpi_columns,
        'all_columns': df.columns.tolist(),
        'sample_data': df.head(3).to_dict('records')
    }
    
    return analysis

def create_agent_with_context(model_choice, main_df, kpi_df=None):
    """Create the pandas dataframe agent with KPI context"""
    try:
        llm = get_llm(model_choice)
        
        # Create context for the agent
        context_prompt = ""
        if kpi_df is not None:
            kpi_analysis = analyze_kpi_file(kpi_df)
            context_prompt = f"""
            You are analyzing data with KPI context. Here's what you need to know:
            
            MAIN DATASET: {len(main_df)} rows, {len(main_df.columns)} columns
            Columns: {', '.join(main_df.columns)}
            
            KPI REFERENCE DATA: {len(kpi_df)} rows, {len(kpi_df.columns)} columns  
            KPI Columns: {', '.join(kpi_analysis['kpi_columns'])}
            All KPI Columns: {', '.join(kpi_df.columns)}
            
            IMPORTANT: When answering questions, prioritize insights based on the KPI data and factors that matter most according to the KPI file. 
            Focus on metrics, targets, priorities, and performance indicators mentioned in the KPI data.
            
            Available dataframes:
            - main_df: Primary dataset for analysis
            - kpi_df: KPI definitions and priorities (if available)
            """
        
        # If we have KPI data, combine both dataframes for the agent
        if kpi_df is not None:
            combined_data = {
                'main_data': main_df,
                'kpi_data': kpi_df
            }
            agent = create_pandas_dataframe_agent(
                llm, 
                [main_df, kpi_df],
                verbose=True, 
                allow_dangerous_code=True,
                handle_parsing_errors=True,
                prefix=context_prompt
            )
        else:
            agent = create_pandas_dataframe_agent(
                llm, 
                main_df,
                verbose=True, 
                allow_dangerous_code=True,
                handle_parsing_errors=True
            )
        
        return agent
    except Exception as e:
        st.error(f"❌ Model initialization failed: {e}")
        return None

def get_kpi_focused_analysis(main_df, kpi_df=None):
    """Generate KPI-focused initial analysis"""
    analysis = f"""
🎯 **Dataset Overview:**
- **Main Data Records:** {len(main_df):,}
- **Main Data Columns:** {len(main_df.columns)}
"""
    
    if kpi_df is not None:
        kpi_analysis = analyze_kpi_file(kpi_df)
        analysis += f"""
- **KPI Reference Data:** {len(kpi_df):,} rows
- **KPI-Related Columns:** {', '.join(kpi_analysis['kpi_columns']) if kpi_analysis['kpi_columns'] else 'Auto-detecting...'}

📊 **KPI-Focused Insights:**
Based on your KPI definitions, I'll prioritize analysis on the factors that matter most to your business objectives.

🔍 **Key Focus Areas Identified:**
"""
        
        # Try to identify key metrics from KPI file
        if kpi_analysis['kpi_columns']:
            for col in kpi_analysis['kpi_columns'][:5]:  # Show top 5
                analysis += f"- **{col}**\n"
    
    analysis += f"""

💡 **What I can help you with:**
- **KPI Performance Analysis** - How are you performing against targets?
- **Priority-Based Insights** - Focus on what matters most per your KPIs
- **Trend Analysis** - Track KPI metrics over time
- **Comparative Analysis** - Compare performance across different dimensions
- **Root Cause Analysis** - Understand drivers behind KPI performance

🚀 **Ask me questions like:**
- "Which KPIs are underperforming?"
- "What factors impact our top priority metrics?"
- "Show performance against targets"
- "Analyze trends for critical KPIs"

Just ask me anything about your data with focus on what matters most! 📈
"""
    
    return analysis

def main():
    st.set_page_config(
        page_title="Lane Analytics KPI Assistant", 
        page_icon="📈",
        layout="wide"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("📈 Lane Analytics KPI Assistant")
    st.markdown("*Intelligent analysis focused on what matters most to your business*")
    
    # Sidebar for configuration and file upload
    with st.sidebar:
        st.header("📁 Data Upload")
        
        # File uploaders
        st.subheader("📊 Main Dataset")
        main_file = st.file_uploader(
            "Upload your main data file", 
            type=['csv', 'xlsx', 'xls'],
            key="main_file",
            help="This is your primary dataset for analysis"
        )
        
        st.subheader("🎯 KPI Reference File (Optional)")
        kpi_file = st.file_uploader(
            "Upload KPI definitions/priorities", 
            type=['csv', 'xlsx', 'xls'],
            key="kpi_file",
            help="File containing KPI definitions, targets, priorities, or weights"
        )
        
        # Process uploaded files
        if main_file:
            if st.button("📥 Load Main Data"):
                with st.spinner("Loading main dataset..."):
                    df = load_file(main_file)
                    if df is not None:
                        st.session_state.df = df
                        st.success(f"✅ Main data loaded: {len(df):,} rows")
                        # Reset conversation when new data is loaded
                        st.session_state.messages = []
                        st.session_state.agent = None
        
        if kpi_file:
            if st.button("🎯 Load KPI Data"):
                with st.spinner("Loading KPI reference data..."):
                    kpi_df = load_file(kpi_file)
                    if kpi_df is not None:
                        st.session_state.kpi_df = kpi_df
                        st.success(f"✅ KPI data loaded: {len(kpi_df):,} rows")
                        # Reset agent to include KPI context
                        st.session_state.agent = None
        
        st.markdown("---")
        
        # Model selection
        st.header("⚙️ AI Configuration")
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
        
        # Data status
        st.header("📋 Data Status")
        
        if st.session_state.df is not None:
            df = st.session_state.df
            st.write(f"**📊 Main Data:** {len(df):,} rows, {len(df.columns)} cols")
            
            with st.expander("📋 Main Data Columns"):
                for col in df.columns:
                    st.write(f"• {col}")
        
        if st.session_state.kpi_df is not None:
            kpi_df = st.session_state.kpi_df
            st.write(f"**🎯 KPI Data:** {len(kpi_df):,} rows, {len(kpi_df.columns)} cols")
            
            kpi_analysis = analyze_kpi_file(kpi_df)
            if kpi_analysis['kpi_columns']:
                with st.expander("🎯 KPI Columns Detected"):
                    for col in kpi_analysis['kpi_columns']:
                        st.write(f"• {col}")
        
        st.markdown("---")
        
        # Quick actions
        st.header("🚀 Quick Actions")
        if st.button("🔄 Clear Chat"):
            st.session_state.messages = []
            st.rerun()
        
        if st.button("📈 Get KPI Analysis"):
            if st.session_state.df is not None:
                analysis_msg = get_kpi_focused_analysis(st.session_state.df, st.session_state.kpi_df)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": analysis_msg,
                    "timestamp": time.time()
                })
                st.rerun()

    # Main chat interface
    if st.session_state.df is None:
        st.info("👋 Welcome! Please upload your main dataset to get started.")
        st.markdown("""
        ### 📁 How to get started:
        1. **Upload Main Dataset**: Your primary data file (CSV/Excel)
        2. **Upload KPI File (Optional)**: File with KPI definitions, targets, priorities
        3. **Choose AI Model**: Select your preferred AI assistant
        4. **Start Analysis**: Ask questions focused on your key metrics!
        
        ### 🎯 KPI File Examples:
        Your KPI file might contain columns like:
        - `KPI_Name`, `Target`, `Weight`, `Priority`
        - `Metric`, `Threshold`, `Importance_Score`
        - `Factor`, `Impact_Level`, `Critical_Flag`
        """)
        return
    
    # Initialize agent if not exists
    if st.session_state.agent is None:
        with st.spinner(f"Initializing {st.session_state.model_choice} model with KPI context..."):
            agent = create_agent_with_context(
                st.session_state.model_choice, 
                st.session_state.df, 
                st.session_state.kpi_df
            )
            if agent:
                st.session_state.agent = agent
                kpi_status = "with KPI prioritization" if st.session_state.kpi_df is not None else "for general analysis"
                st.success(f"✅ {st.session_state.model_choice} model ready {kpi_status}!")
                
                # Add welcome message if this is the first time
                if not st.session_state.messages:
                    welcome_msg = f"""
👋 **Hello! I'm your Lane Analytics KPI Assistant powered by {st.session_state.model_choice}.**

{get_kpi_focused_analysis(st.session_state.df, st.session_state.kpi_df)}
"""
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": welcome_msg,
                        "timestamp": time.time()
                    })
            else:
                st.error("Failed to initialize the model. Please check your API keys.")
                return

    # Chat interface
    st.markdown("---")
    
    # Display chat messages
    chat_container = st.container()
    
    with chat_container:
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # KPI-focused example questions
    if len(st.session_state.messages) <= 1:
        st.markdown("### 💡 KPI-Focused Questions You Can Ask:")
        
        if st.session_state.kpi_df is not None:
            col1, col2, col3 = st.columns(3)
            
            kpi_questions = [
                "Which KPIs are underperforming?",
                "What factors impact top priority metrics?",
                "Show performance vs targets",
                "Analyze critical KPI trends",
                "Which metrics need immediate attention?",
                "Compare KPI performance across segments"
            ]
        else:
            col1, col2, col3 = st.columns(3)
            
            kpi_questions = [
                "What are the key performance indicators?",
                "Show me top metrics and trends",
                "Identify critical success factors",
                "Which metrics show best performance?",
                "Analyze performance patterns",
                "What drives the best results?"
            ]
        
        for i, question in enumerate(kpi_questions):
            col = [col1, col2, col3][i % 3]
            with col:
                if st.button(question, key=f"kpi_example_{i}", use_container_width=True):
                    # Add user question to messages
                    st.session_state.messages.append({
                        "role": "user", 
                        "content": question,
                        "timestamp": time.time()
                    })
                    st.rerun()

    # Chat input with KPI context
    placeholder_text = "Ask about KPIs, performance metrics, or key factors..." if st.session_state.kpi_df is not None else "Ask about your data analysis..."
    
    if prompt := st.chat_input(placeholder_text):
        # Add user message
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt,
            "timestamp": time.time()
        })
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing with KPI focus..."):
                try:
                    response = st.session_state.agent.run(prompt)
                    st.markdown(response)
                    
                    # Add assistant response to messages
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "timestamp": time.time()
                    })
                    
                except Exception as e:
                    error_msg = f"❌ Sorry, I encountered an error: {str(e)}\n\nTry rephrasing your question or ask me something else about your KPIs!"
                    st.markdown(error_msg)
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg,
                        "timestamp": time.time()
                    })

    # Footer
    st.markdown("---")
    st.markdown("*📈 KPI-focused conversations with your data - Powered by LangChain & Streamlit*")

if __name__ == "__main__":
    main()