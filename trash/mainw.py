import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import time
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# Use Google embeddings instead of OpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document

# ✅ Load environment variables at the very start
load_dotenv()

# ✅ Set environment variables manually if needed
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")


class AdvancedRAGAnalyzer:
    """Advanced RAG-based data analyzer with function calling capabilities"""
    
    def __init__(self, df, api_key, model_choice="Gemini"):
        self.df = df
        self.model_choice = model_choice
        self.conversation_history = []
        
        # Use Google embeddings instead of OpenAI
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=api_key
            )
            self.vector_store = self._create_vector_store()
        except Exception as e:
            st.error(f"Failed to initialize embeddings: {e}")
            # Fallback: work without embeddings
            self.embeddings = None
            self.vector_store = None
        
        # Initialize the Gemini model for chat
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp", 
                temperature=0, 
                google_api_key=api_key
            )
        except Exception as e:
            st.error(f"Failed to initialize LLM: {e}")
            self.llm = None
        
        # Initialize function tools
        self.available_functions = {
            "calculate_statistics": self._calculate_statistics,
            "create_visualization": self._create_visualization,
            "filter_data": self._filter_data,
            "group_analysis": self._group_analysis,
            "correlation_analysis": self._correlation_analysis,
            "time_series_analysis": self._time_series_analysis
        }
    
    def _create_vector_store(self):
        """Create vector store from dataset metadata and statistics"""
        if not self.embeddings:
            return None
            
        documents = []
        
        # Dataset overview
        overview = f"""
        Dataset Overview:
        Shape: {self.df.shape[0]} rows, {self.df.shape[1]} columns
        Columns: {', '.join(self.df.columns)}
        Data Types: {dict(self.df.dtypes)}
        Memory Usage: {self.df.memory_usage().sum()} bytes
        """
        documents.append(Document(page_content=overview, metadata={"type": "overview"}))
        
        # Column descriptions
        for col in self.df.columns:
            col_info = f"""
            Column: {col}
            Data Type: {self.df[col].dtype}
            Unique Values: {self.df[col].nunique()}
            Missing Values: {self.df[col].isnull().sum()}
            Sample Values: {self.df[col].dropna().head(5).tolist()}
            """
            if self.df[col].dtype in ['int64', 'float64']:
                col_info += f"""
                Statistics:
                Mean: {self.df[col].mean():.2f}
                Median: {self.df[col].median():.2f}
                Standard Deviation: {self.df[col].std():.2f}
                Min: {self.df[col].min()}
                Max: {self.df[col].max()}
                """
            documents.append(Document(page_content=col_info, metadata={"type": "column", "column": col}))
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        
        # Create vector store
        try:
            vector_store = FAISS.from_documents(splits, self.embeddings)
            return vector_store
        except Exception as e:
            st.warning(f"Could not create vector store: {e}")
            return None
    
    def _calculate_statistics(self, columns=None, operation="describe"):
        """Calculate statistics for specified columns"""
        try:
            if columns is None:
                if operation == "describe":
                    return self.df.describe().to_dict()
                elif operation == "info":
                    return {
                        "shape": self.df.shape,
                        "columns": list(self.df.columns),
                        "dtypes": dict(self.df.dtypes),
                        "memory_usage": self.df.memory_usage().to_dict()
                    }
            else:
                if isinstance(columns, str):
                    columns = [columns]
                return self.df[columns].describe().to_dict()
        except Exception as e:
            return f"Error calculating statistics: {str(e)}"
    
    def _create_visualization(self, chart_type, x_col, y_col=None, color_col=None):
        """Create visualizations using plotly"""
        try:
            if chart_type == "histogram":
                fig = px.histogram(self.df, x=x_col, color=color_col, title=f"Distribution of {x_col}")
            elif chart_type == "scatter":
                fig = px.scatter(self.df, x=x_col, y=y_col, color=color_col, title=f"{x_col} vs {y_col}")
            elif chart_type == "bar":
                data = self.df.groupby(x_col).size().reset_index(name='count')
                fig = px.bar(data, x=x_col, y='count', title=f"Count by {x_col}")
            elif chart_type == "line":
                fig = px.line(self.df, x=x_col, y=y_col, color=color_col, title=f"{y_col} over {x_col}")
            else:
                return "Unsupported chart type"
            
            return fig.to_json()
        except Exception as e:
            return f"Error creating visualization: {str(e)}"
    
    def _filter_data(self, conditions):
        """Filter data based on conditions"""
        try:
            # Simple filtering implementation
            filtered_df = self.df.copy()
            for condition in conditions:
                col, op, value = condition['column'], condition['operator'], condition['value']
                if op == "==":
                    filtered_df = filtered_df[filtered_df[col] == value]
                elif op == ">":
                    filtered_df = filtered_df[filtered_df[col] > value]
                elif op == "<":
                    filtered_df = filtered_df[filtered_df[col] < value]
                elif op == ">=":
                    filtered_df = filtered_df[filtered_df[col] >= value]
                elif op == "<=":
                    filtered_df = filtered_df[filtered_df[col] <= value]
            
            return {
                "shape": filtered_df.shape,
                "sample": filtered_df.head().to_dict(),
                "summary": filtered_df.describe().to_dict()
            }
        except Exception as e:
            return f"Error filtering data: {str(e)}"
    
    def _group_analysis(self, group_by, agg_col, agg_func="mean"):
        """Perform group analysis"""
        try:
            if agg_func == "mean":
                result = self.df.groupby(group_by)[agg_col].mean()
            elif agg_func == "sum":
                result = self.df.groupby(group_by)[agg_col].sum()
            elif agg_func == "count":
                result = self.df.groupby(group_by)[agg_col].count()
            elif agg_func == "max":
                result = self.df.groupby(group_by)[agg_col].max()
            elif agg_func == "min":
                result = self.df.groupby(group_by)[agg_col].min()
            
            return result.to_dict()
        except Exception as e:
            return f"Error in group analysis: {str(e)}"
    
    def _correlation_analysis(self, columns=None):
        """Calculate correlation matrix"""
        try:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if columns:
                numeric_cols = [col for col in columns if col in numeric_cols]
            
            correlation_matrix = self.df[numeric_cols].corr()
            return correlation_matrix.to_dict()
        except Exception as e:
            return f"Error in correlation analysis: {str(e)}"
    
    def _time_series_analysis(self, date_col, value_col, freq="D"):
        """Perform time series analysis"""
        try:
            df_ts = self.df.copy()
            df_ts[date_col] = pd.to_datetime(df_ts[date_col])
            df_ts = df_ts.set_index(date_col)
            
            # Resample data
            resampled = df_ts[value_col].resample(freq).sum()
            
            return {
                "time_series": resampled.to_dict(),
                "trend": "increasing" if resampled.iloc[-1] > resampled.iloc[0] else "decreasing",
                "statistics": resampled.describe().to_dict()
            }
        except Exception as e:
            return f"Error in time series analysis: {str(e)}"
    
    def _get_relevant_context(self, query):
        """Get relevant context using RAG"""
        if self.vector_store:
            try:
                # Use vector search for context
                docs = self.vector_store.similarity_search(query, k=3)
                context = "\n\n".join([doc.page_content for doc in docs])
                return context
            except Exception as e:
                st.warning(f"Vector search failed: {e}")
        
        # Fallback: basic context from dataframe info
        context = f"""
        Dataset Information:
        - Shape: {self.df.shape}
        - Columns: {list(self.df.columns)}
        - Data Types: {dict(self.df.dtypes)}
        - Sample data: {self.df.head(3).to_dict()}
        """
        return context
    
    def run(self, query):
        """Process query with advanced RAG and function calling"""
        try:
            # Get relevant context
            context = self._get_relevant_context(query)
            
            # Create enhanced prompt with context and function information
            enhanced_prompt = f"""
You are an advanced data analyst with access to a dataset and various analysis functions.

Dataset Context:
{context}

Available Functions:
- calculate_statistics: Get statistical summaries
- create_visualization: Generate charts and plots
- filter_data: Filter data based on conditions
- group_analysis: Perform groupby operations
- correlation_analysis: Calculate correlations
- time_series_analysis: Analyze temporal data

User Query: {query}

Provide a comprehensive analysis addressing the user's question. If relevant, mention what specific functions could be called to get more detailed results.

Response Guidelines:
1. Be specific and data-driven
2. Use actual column names and values from the dataset
3. Suggest relevant visualizations or analysis techniques
4. Provide actionable insights
5. If the query asks for something specific that a function can handle, explain what that function would do
"""

            if self.llm:
                try:
                    response = self.llm.invoke(enhanced_prompt)
                    return response.content
                except Exception as e:
                    return f"LLM Error: {str(e)}\n\nFallback: I can see your dataset has {self.df.shape[0]} rows and {self.df.shape[1]} columns. The columns are: {', '.join(self.df.columns)}. What specific analysis would you like me to perform?"
            else:
                return self._generate_fallback_response(query, context)
                
        except Exception as e:
            return f"❌ Error processing query: {str(e)}"
    
    def _generate_fallback_response(self, query, context):
        """Generate a basic response when LLM is not available"""
        response = f"""Based on your query: "{query}"

Dataset Overview:
- Records: {self.df.shape[0]:,}
- Columns: {self.df.shape[1]}
- Available columns: {', '.join(self.df.columns)}

I can help you with:
1. Statistical analysis of your data
2. Data visualization and charts  
3. Finding patterns and correlations
4. Time series analysis (if applicable)
5. Data filtering and grouping

What specific aspect would you like to explore? For example:
- "Show me statistics for [column name]"
- "Create a chart comparing [column1] vs [column2]"
- "Find correlations between numeric columns"
- "Analyze trends over time"
"""
        return response


class DashboardGenerator:
    """Generate analytics dashboard components"""
    
    def __init__(self, df):
        self.df = df
    
    def create_overview_metrics(self):
        """Create overview metrics cards"""
        metrics = {}
        
        # Basic metrics
        metrics['total_records'] = len(self.df)
        metrics['total_columns'] = len(self.df.columns)
        
        # Numeric columns analysis
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Try common column names
            for col_name in ['Total', 'total', 'amount', 'Amount', 'sales', 'Sales', 'revenue', 'Revenue']:
                if col_name in self.df.columns:
                    metrics[f'total_{col_name.lower()}'] = self.df[col_name].sum()
                    metrics[f'avg_{col_name.lower()}'] = self.df[col_name].mean()
                    break
        
        return metrics
    
    def create_distribution_chart(self, column):
        """Create distribution chart for a column"""
        if column in self.df.columns:
            if self.df[column].dtype == 'object':
                # Categorical distribution
                data = self.df[column].value_counts().head(10)
                fig = px.bar(x=data.index, y=data.values, 
                           title=f"Distribution of {column}")
                fig.update_layout(
    xaxis_title=column,
    yaxis_title="Count"
)

            else:
                # Numeric distribution
                fig = px.histogram(self.df, x=column, 
                                 title=f"Distribution of {column}")
            return fig
        return None
    
    def create_correlation_heatmap(self):
        """Create correlation heatmap for numeric columns"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = self.df[numeric_cols].corr()
            fig = px.imshow(corr_matrix, 
                          text_auto=True, 
                          aspect="auto",
                          title="Correlation Heatmap")
            return fig
        return None
    
    def create_time_series_chart(self, date_col, value_col):
        """Create time series chart if date column exists"""
        if date_col in self.df.columns and value_col in self.df.columns:
            df_ts = self.df.copy()
            try:
                df_ts[date_col] = pd.to_datetime(df_ts[date_col])
                df_ts = df_ts.groupby(date_col)[value_col].sum().reset_index()
                fig = px.line(df_ts, x=date_col, y=value_col,
                            title=f"{value_col} Over Time")
                return fig
            except:
                pass
        return None


def get_llm(model_choice):
    """Get the language model based on choice"""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Google API key not found. Please set GOOGLE_API_KEY in your .env file.")
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp", 
        temperature=0, 
        google_api_key=api_key
    )


def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "df" not in st.session_state:
        st.session_state.df = None
    if "model_choice" not in st.session_state:
        st.session_state.model_choice = "Gemini"
    if "dashboard_generator" not in st.session_state:
        st.session_state.dashboard_generator = None


def load_data():
    """Load the dataset"""
    try:
        df = pd.read_csv("supermarket_sales.csv")
        st.session_state.df = df
        st.session_state.dashboard_generator = DashboardGenerator(df)
        return df
    except FileNotFoundError:
        st.error("⚠️ 'supermarket_sales.csv' not found in the project directory.")
        return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


def create_agent(model_choice, df):
    """Create the advanced RAG analyzer"""
    try:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key not found. Please set GOOGLE_API_KEY in your .env file.")
        return AdvancedRAGAnalyzer(df, api_key, model_choice)
    except Exception as e:
        st.error(f"❌ Model initialization failed: {e}")
        return None


def display_data_preview(df):
    """Display comprehensive data preview"""
    st.header("📊 Data Preview & Overview")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Basic Info", "🔍 Sample Data", "📈 Statistics", "📊 Visualizations"])
    
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", f"{len(df):,}")
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("Memory Usage", f"{df.memory_usage().sum() / 1024:.1f} KB")
        with col4:
            missing_cells = df.isnull().sum().sum()
            st.metric("Missing Values", missing_cells)
        
        st.subheader("Column Information")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Unique Values': df.nunique()
        }).reset_index(drop=True)
        st.dataframe(col_info, use_container_width=True)
    
    with tab2:
        st.subheader("Sample Data")
        
        col1, col2 = st.columns([3, 1])
        with col2:
            sample_size = st.selectbox("Sample Size", [5, 10, 20, 50], index=0)
        
        with col1:
            st.dataframe(df.head(sample_size), use_container_width=True)
        
        st.subheader("Random Sample")
        st.dataframe(df.sample(min(5, len(df))), use_container_width=True)
    
    with tab3:
        st.subheader("Statistical Summary")
        
        # Numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.write("**Numeric Columns:**")
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        
        # Categorical columns
        cat_cols = df.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            st.write("**Categorical Columns:**")
            cat_summary = pd.DataFrame({
                'Column': cat_cols,
                'Unique Values': [df[col].nunique() for col in cat_cols],
                'Most Frequent': [df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'N/A' for col in cat_cols],
                'Frequency of Most Common': [df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0 for col in cat_cols]
            })
            st.dataframe(cat_summary, use_container_width=True)
    
    with tab4:
        if st.session_state.dashboard_generator:
            dashboard = st.session_state.dashboard_generator
            
            # Distribution charts
            st.subheader("Data Distributions")
            
            # Select column for distribution
            all_cols = df.columns.tolist()
            selected_col = st.selectbox("Select column for distribution", all_cols)
            
            if selected_col:
                fig = dashboard.create_distribution_chart(selected_col)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            # Correlation heatmap
            st.subheader("Correlation Analysis")
            corr_fig = dashboard.create_correlation_heatmap()
            if corr_fig:
                st.plotly_chart(corr_fig, use_container_width=True)
            else:
                st.info("Not enough numeric columns for correlation analysis")
            
            # Time series if date column exists
            date_cols = []
            for col in df.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    date_cols.append(col)
            
            if date_cols:
                st.subheader("Time Series Analysis")
                date_col = st.selectbox("Select date column", date_cols)
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    value_col = st.selectbox("Select value column", numeric_cols)
                    ts_fig = dashboard.create_time_series_chart(date_col, value_col)
                    if ts_fig:
                        st.plotly_chart(ts_fig, use_container_width=True)


def display_analytics_dashboard(df):
    """Display comprehensive analytics dashboard"""
    st.header("📈 Analytics Dashboard")
    
    if st.session_state.dashboard_generator:
        dashboard = st.session_state.dashboard_generator
        
        # Overview metrics
        metrics = dashboard.create_overview_metrics()
        
        # Display metrics in columns
        if metrics:
            cols = st.columns(len(metrics))
            for i, (key, value) in enumerate(metrics.items()):
                with cols[i % len(cols)]:
                    if isinstance(value, (int, float)):
                        if 'total' in key.lower() and value > 1000:
                            st.metric(key.replace('_', ' ').title(), f"${value:,.2f}" if 'sales' in key or 'amount' in key else f"{value:,.0f}")
                        else:
                            st.metric(key.replace('_', ' ').title(), f"{value:.2f}" if isinstance(value, float) else f"{value:,}")
        
        st.markdown("---")
    
    # Initialize agent if not exists
    if st.session_state.agent is None:
        with st.spinner(f"Initializing Advanced {st.session_state.model_choice} RAG System..."):
            agent = create_agent(st.session_state.model_choice, st.session_state.df)
            if agent:
                st.session_state.agent = agent
                st.success(f"✅ Advanced {st.session_state.model_choice} RAG System Ready!")
                
                # Add welcome message if this is the first time
                if not st.session_state.messages:
                    welcome_msg = f"""
👋 **Welcome to Enhanced Lane Analytics powered by Advanced RAG!**

I've analyzed your dataset and created a comprehensive knowledge base. Here's what I can do:

🔍 **Advanced Capabilities:**
- **Contextual Analysis**: I understand your data structure and relationships
- **Function Calling**: Automated statistical analysis and visualizations  
- **RAG Pipeline**: Retrieval-augmented responses for accurate insights
- **Interactive Dashboard**: Real-time analytics and metrics

📊 **Your Dataset:**
- **{len(df):,} records** across **{len(df.columns)} columns**
- **Data Types**: {len(df.select_dtypes(include=[np.number]).columns)} numeric, {len(df.select_dtypes(include=['object']).columns)} categorical
- **Memory Usage**: {df.memory_usage().sum() / 1024:.1f} KB

💡 **Try asking me:**
- Complex analytical questions
- Requests for visualizations
- Statistical comparisons
- Trend analysis
- Data insights and patterns

I'm ready to dive deep into your data! 🚀
"""
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": welcome_msg,
                        "timestamp": time.time()
                    })
            else:
                st.error("Failed to initialize the advanced RAG system. Please check your API keys.")
                return

    # Chat interface
    st.header("💬 Intelligent Data Conversation")
    
    # Display chat messages
    chat_container = st.container()
    
    with chat_container:
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Advanced example questions
    if len(st.session_state.messages) <= 1:
        st.markdown("### 💡 Advanced Analytics Questions:")
        
        # Create columns for different types of questions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📊 Statistical Analysis**")
            stat_questions = [
                "What are the key statistical insights from this data?",
                "Perform correlation analysis on numeric columns",
                "Show me outliers in the dataset",
                "Calculate advanced statistics for sales data"
            ]
            for i, question in enumerate(stat_questions):
                if st.button(question, key=f"stat_{i}", use_container_width=True):
                    st.session_state.messages.append({
                        "role": "user", 
                        "content": question,
                        "timestamp": time.time()
                    })
                    st.rerun()
        
        with col2:
            st.markdown("**📈 Visualization & Trends**")
            viz_questions = [
                "Create visualizations for top-performing categories",
                "Show me sales trends over time",
                "Generate a comprehensive dashboard view",
                "Compare performance across different segments"
            ]
            for i, question in enumerate(viz_questions):
                if st.button(question, key=f"viz_{i}", use_container_width=True):
                    st.session_state.messages.append({
                        "role": "user", 
                        "content": question,
                        "timestamp": time.time()
                    })
                    st.rerun()
        
        with col3:
            st.markdown("**🔍 Advanced Insights**")
            insight_questions = [
                "What are the hidden patterns in this data?",
                "Identify business opportunities from the analysis",
                "Provide strategic recommendations based on data",
                "What questions should I be asking about this data?"
            ]
            for i, question in enumerate(insight_questions):
                if st.button(question, key=f"insight_{i}", use_container_width=True):
                    st.session_state.messages.append({
                        "role": "user", 
                        "content": question,
                        "timestamp": time.time()
                    })
                    st.rerun()

    # Enhanced chat input with suggestions
    st.markdown("---")
    
    # Quick action buttons
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("📊 Data Summary", use_container_width=True):
            summary_query = "Provide a comprehensive summary of the dataset including key insights and recommendations"
            st.session_state.messages.append({
                "role": "user", 
                "content": summary_query,
                "timestamp": time.time()
            })
            st.rerun()
    
    with col2:
        if st.button("📈 Create Charts", use_container_width=True):
            viz_query = "Create relevant visualizations for this dataset and explain the insights"
            st.session_state.messages.append({
                "role": "user", 
                "content": viz_query,
                "timestamp": time.time()
            })
            st.rerun()
    
    with col3:
        if st.button("🔍 Find Patterns", use_container_width=True):
            pattern_query = "Analyze the data to find interesting patterns, correlations, and anomalies"
            st.session_state.messages.append({
                "role": "user", 
                "content": pattern_query,
                "timestamp": time.time()
            })
            st.rerun()
    
    with col4:
        if st.button("💡 Business Insights", use_container_width=True):
            business_query = "What are the key business insights and actionable recommendations from this data?"
            st.session_state.messages.append({
                "role": "user", 
                "content": business_query,
                "timestamp": time.time()
            })
            st.rerun()

 
    # Main chat input
    if prompt := st.chat_input("Ask me anything about your data... (powered by Advanced RAG)"):
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
            with st.spinner("🧠 Analyzing with Advanced RAG..."):
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
                    error_msg = f"❌ Sorry, I encountered an error: {str(e)}\n\nThe advanced RAG system is still learning. Try rephrasing your question or ask me something else!"
                    st.markdown(error_msg)
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg,
                        "timestamp": time.time()
                    })

    st.markdown("---")
    
    # Performance metrics
    if st.session_state.agent and hasattr(st.session_state.agent, 'conversation_history'):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Conversations", len(st.session_state.messages))
        with col2:
            st.metric("RAG Context", "Vector Search" if st.session_state.model_choice != "Claude" else "Enhanced Context")
        with col3:
            st.metric("Functions Available", len(st.session_state.agent.available_functions))
        with col4:
            st.metric("Model", st.session_state.model_choice)
    
    st.markdown("""
    <div style='text-align: center; color: #666; margin-top: 20px;'>
        💬 <strong>Enhanced Lane Analytics</strong> - Advanced RAG Pipeline | Function Calling | Interactive Dashboard<br>
        🚀 Powered by LangChain, Streamlit, and Modern AI
    </div>
    """, unsafe_allow_html=True)
    
    # Expandable technical details
    with st.expander("🔧 Technical Features"):
        st.markdown("""
        **🔍 Advanced RAG Pipeline:**
        - Vector embeddings for semantic search
        - Context-aware response generation
        - Intelligent document chunking and retrieval
        
        **🛠️ Function Calling System:**
        - Automated statistical analysis
        - Dynamic visualization creation
        - Data filtering and grouping operations
        - Correlation and time-series analysis
        
        **📊 Smart Dashboard:**
        - Real-time data metrics
        - Interactive visualizations
        - Comprehensive data preview
        - Business insights generation
        
        **🤖 AI Models Supported:**
        - OpenAI GPT with embeddings
        - Google Gemini/VertexAI
        - Anthropic Claude 3.5 Sonnet
        - Custom function integration
        """)

def main():
    st.set_page_config(
        page_title="Enhanced Lane Analytics", 
        page_icon="🤖",
        layout="wide"
    )
    initialize_session_state()

    st.title("🤖 Enhanced Lane Analytics Assistant")
    st.markdown("*Advanced RAG-powered data analysis with interactive dashboard*")

    with st.sidebar:
        st.header("⚙️ Configuration")

        model_choice = st.selectbox(
            "Choose AI Model:", 
            ["OpenAI", "VertexAI", "Gemini"],
            index=["OpenAI", "VertexAI", "Gemini"].index(st.session_state.model_choice)
        )

        if model_choice != st.session_state.model_choice:
            st.session_state.model_choice = model_choice
            st.session_state.agent = None  
            st.rerun()
        else:
            st.info(f"🧠 Using {model_choice} with Vector Search RAG")
        
        st.markdown("---")
        st.header("📊 Data Management")
        
        if st.button("🔄 Load/Reload Data"):
            with st.spinner("Loading data..."):
                df = load_data()
                if df is not None:
                    st.success(f"✅ Data loaded: {len(df):,} rows")
                    st.session_state.messages = []
                    st.session_state.agent = None
        if st.session_state.df is not None:
            df = st.session_state.df
            st.write(f"**Rows:** {len(df):,}")
            st.write(f"**Columns:** {len(df.columns)}")
            st.write(f"**Size:** {df.memory_usage().sum() / 1024:.1f} KB")
            
            with st.expander("📋 Columns"):
                for col in df.columns:
                    dtype = str(df[col].dtype)
                    unique = df[col].nunique()
                    st.write(f"• **{col}** ({dtype}) - {unique} unique")
        
        st.markdown("---")
        st.header("🚀 Advanced Features")
        
        feature_info = """
        **🔍 RAG Pipeline:** Context-aware responses using vector search
        
        **🛠️ Function Calling:** Automated data analysis functions
        
        **📊 Smart Dashboard:** Interactive visualizations and insights
        
        **💬 Conversational AI:** Natural language data queries
        """
        st.markdown(feature_info)
        
        if st.button("🔄 Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    if st.session_state.df is None:
        st.info("👋 Welcome to Enhanced Lane Analytics! Please load your data using the sidebar to get started.")
        st.markdown("""
        ### 🌟 New Features:
        - **Advanced RAG Pipeline** for better context understanding
        - **Function Calling** for automated data analysis
        - **Interactive Dashboard** with real-time insights
        - **Data Preview** with comprehensive statistics
        - **Smart Visualizations** based on your data
        """)
        return
    df = st.session_state.df
    display_data_preview(df)
    st.markdown("---")
    display_analytics_dashboard(df)
    
    st.markdown("---")

if __name__ == "__main__":
    main()
    
    if "df" in st.session_state and st.session_state.df is not None:
        df = st.session_state.df
        st.subheader("🔍 Quick Insights")
        
        col1, col2 = st.columns(2)

        with col1:
            cat_cols = df.select_dtypes(include=['object']).columns
            if len(cat_cols) > 0:
                for col in cat_cols[:2]:
                    st.write(f"**Top {col} values:**")
                    top_values = df[col].value_counts().head(5)
                    for val, count in top_values.items():
                        st.write(f"• {val}: {count:,} ({count/len(df)*100:.1f}%)")
                    st.write("")

        with col2:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.write("**Numeric Column Ranges:**")
                for col in numeric_cols[:3]:
                    min_val = df[col].min()
                    max_val = df[col].max()
                    mean_val = df[col].mean()
                    st.write(f"**{col}:** {min_val:.2f} - {max_val:.2f} (avg: {mean_val:.2f})")
