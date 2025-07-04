import os
import pandas as pd
from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from dotenv import load_dotenv
import time
import uuid
import re

from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import OpenAI
from langchain_google_vertexai import VertexAI
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Set environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
gcp_key_json = os.getenv("GCP_KEY_JSON")
if gcp_key_json:
    with open("vertex_key.json", "w") as f:
        f.write(gcp_key_json)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "vertex_key.json"

else:
    print("⚠️ GCP_KEY_JSON not found. Vertex AI might fail without it.")
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-secret-key-here")
CORS(app)

# Global variables to store data and agents
df = None
df_kpis = None
agents = {}  # Store agents per session

def get_llm(model_choice):
    """Get the selected language model"""
    if model_choice == "OpenAI":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
        return OpenAI(temperature=0, api_key=api_key)
    
    elif model_choice == "VertexAI":
        credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not credentials_path:
            raise ValueError("Google credentials not found. Please set GOOGLE_APPLICATION_CREDENTIALS in your .env file.")
        return VertexAI(model_name="gemini-2.0-flash", temperature=0, location="us-east4")
    
    elif model_choice == "Gemini":
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key not found. Please set GOOGLE_API_KEY in your .env file.")
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            temperature=0.7, 
            google_api_key=api_key
        )
    else:
        raise ValueError("Unsupported model selected.")

def load_data():
    """Load the dataset"""
    global df
    try:
        df = pd.read_csv("supermarket_sales_final.csv")
        return True, f"Data loaded successfully: {len(df):,} rows, {len(df.columns)} columns"
    except FileNotFoundError:
        return False, "'supermarket_sales.csv' not found in the project directory."
    except Exception as e:
        return False, f"Error loading file: {str(e)}"

def create_agent(model_choice, session_id):
    global df, df_kpis, agents

    if df is None or df_kpis is None:
        print("🚨 ERROR: One or both dataframes are not loaded.")
        return None, "Both datasets must be loaded before creating the agent."

    try:
        llm = get_llm(model_choice)

        # ✅ Use only df for agent — df_kpis is available globally
        agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            allow_dangerous_code=True,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )

        # Inject df_kpis into global scope so it's accessible during agent execution
        globals()["kpis"] = df_kpis

        agents[session_id] = {
            'agent': agent,
            'model': model_choice,
            'created_at': time.time()
        }

        return agent, f"{model_choice} model initialized with sales and KPI data"
    except Exception as e:
        print("❌ Exception during agent creation:", str(e))
        return None, f"Model initialization failed: {str(e)}"

def get_initial_analysis():
    """Generate initial analysis of the dataset"""
    global df
    
    if df is None:
        return "No data loaded."
    
    total_sales = df['Total'].sum() if 'Total' in df.columns else 0
    avg_transaction = df['Total'].mean() if 'Total' in df.columns else 0
    date_range = f"{df['Date'].min()} to {df['Date'].max()}" if 'Date' in df.columns else 'N/A'
    top_products = ', '.join(df['Product line'].value_counts().head(3).index.tolist()) if 'Product line' in df.columns else 'N/A'
    
    analysis = f"""
📈 **Dataset Overview:**
- **Total Records:** {len(df):,}
- **Columns:** {len(df.columns)}
- **Date Range:** {date_range}

📊 **Key Insights:**
- **Total Sales:** ${total_sales:,.2f}
- **Average Transaction:** ${avg_transaction:.2f}
- **Top Product Lines:** {top_products}

💡 **What would you like to explore?**
- Sales trends and patterns
- Customer behavior analysis  
- Product performance insights
- Branch/location comparisons
- Payment method preferences

Just ask me anything about your data! 🤖
"""
    return analysis

def get_system_prompt():
    return (
        "You are a helpful data assistant.\n"
        "You have access to two pandas dataframes:\n"
        "- `df`: supermarket sales data\n"
        "- `kpis`: key performance indicators from a separate Excel file\n"
        "Even though the agent was initialized with only `df`, you can access both `df` and `kpis` during execution."
    )



def extract_numerical_answer(response_text):
    """Extract numerical values from response text and format them properly"""
    # Look for numbers (including decimals) in the response
    numbers = re.findall(r'\b\d+\.?\d*\b', response_text)
    if numbers:
        # Try to convert to float and format with 2 decimal places
        try:
            main_number = float(numbers[0])
            return f"{main_number:.2f}"
        except ValueError:
            return numbers[0]
    return None

@app.route('/')
def index():
    """Serve the main chat interface"""
    # Initialize session
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    if 'messages' not in session:
        session['messages'] = []
    
    return render_template('index.html')

@app.route('/api/load_data', methods=['POST'])
def load_data():
    global df, df_kpis
    try:
        df = pd.read_csv("supermarket_sales_final.csv")
        df_kpis = pd.read_excel("Lane_KPIs.xlsx")

        df['Date'] = pd.to_datetime(df['Date'])
        df_kpis.columns = df_kpis.columns.str.strip()

        print("✅ Data loaded:", df.shape, df_kpis.shape)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "message": f"Error loading data: {str(e)}"})

@app.route('/api/initialize_model', methods=['POST'])
def api_initialize_model():
    """Initialize the selected model"""
    data = request.get_json()
    model_choice = data.get('model', 'OpenAI')
    session_id = session.get('session_id')
    
    agent, message = create_agent(model_choice, session_id)
    
    if agent:
        # Add welcome message to session
        welcome_msg = f"👋 **Hello! I'm your Lane Analytics Assistant powered by {model_choice}.**\n\nI've loaded your supermarket sales data and I'm ready to help you analyze it!\n\n{get_initial_analysis()}"
        
        if 'messages' not in session:
            session['messages'] = []
        
        session['messages'].append({
            'role': 'assistant',
            'content': welcome_msg,
            'timestamp': time.time()
        })
        
        return jsonify({
            'success': True,
            'message': message,
            'welcome_message': welcome_msg
        })
    else:
        return jsonify({'success': False, 'message': message}), 400

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """Handle chat messages with descriptive responses using Final Answer pattern"""
    data = request.get_json()
    user_message = data.get('message', '').strip()
    session_id = session.get('session_id')

    if not user_message:
        return jsonify({'success': False, 'message': 'No message provided'}), 400

    if session_id not in agents:
        return jsonify({'success': False, 'message': 'Model not initialized. Please initialize a model first.'}), 400

    # Save user message
    if 'messages' not in session:
        session['messages'] = []
    session['messages'].append({
        'role': 'user',
        'content': user_message,
        'timestamp': time.time()
    })

    # Simple quick replies
    greetings = ["hello", "hi", "hey"]
    thanks = ["thank you", "thanks", "thx"]

    if user_message.lower() in greetings:
        response = "👋 Hello! I'm your Lane Analytics Assistant. Ask me anything about your supermarket sales data!"
        viz_suggestion = None
    elif user_message.lower() in thanks:
        response = "You're welcome! Let me know if you have more questions about your data."
        viz_suggestion = None
    else:
        try:
            agent = agents[session_id]['agent']

            # Force the agent to structure the output using Final Answer
            formatted_prompt = f"""
You are a helpful data analyst.

You must always return your final response after the phrase: `Final Answer:` (without backticks).

Please answer the user's query in a descriptive and analytical way using the dataset.

User's query:
{user_message}

Provide context, trends, statistics, and any relevant analysis. End your message with:
Final Answer: <your full response here>
            """

            response_raw = agent.invoke({"input": formatted_prompt})

            if isinstance(response_raw, dict):
                raw_text = response_raw.get('output') or response_raw.get('result') or str(response_raw)
            else:
                raw_text = str(response_raw)

            print(f"DEBUG: Raw agent output:\n{raw_text}")

            # Extract just the Final Answer part
            match = re.search(r'Final Answer:\s*(.*)', raw_text, re.DOTALL)
            response = match.group(1).strip() if match else raw_text

            # Format numbers
            def round_match(match):
                try:
                    return f"{float(match.group()):.2f}"
                except:
                    return match.group()
            response = re.sub(r'(?<![\w-])(\d+\.\d+)(?![\w-])', round_match, response)

            # Visualization suggestion
            viz_suggestion = None
            try:
                viz_prompt = f"Based on this question: '{user_message}' and response: '{response}', suggest a chart type (bar/line/pie/scatter/table) and relevant columns. Respond with JSON: {{'chart_type': 'bar', 'columns': ['col1', 'col2'], 'reason': '...'}}"
                viz_response = agent.invoke({"input": viz_prompt})
                viz_suggestion = viz_response.get("output") if isinstance(viz_response, dict) else str(viz_response)
            except Exception as ve:
                print(f"Visualization suggestion failed: {ve}")
                viz_suggestion = None

        except Exception as e:
            print(f"ERROR: {e}")
            response = "❌ Something went wrong. Please try rephrasing your question."
            viz_suggestion = None

    # Save assistant message
    assistant_msg = {
        'role': 'assistant',
        'content': response,
        'timestamp': time.time()
    }
    session['messages'].append(assistant_msg)

    return jsonify({
        'success': True,
        'response': response,
        'timestamp': assistant_msg['timestamp'],
        'visualization': viz_suggestion
    })

@app.route('/api/get_messages', methods=['GET'])
def api_get_messages():
    """Get all messages for the current session"""
    messages = session.get('messages', [])
    return jsonify({'messages': messages})

@app.route('/api/clear_chat', methods=['POST'])
def api_clear_chat():
    """Clear chat messages"""
    session['messages'] = []
    return jsonify({'success': True, 'message': 'Chat cleared successfully'})

@app.route('/api/get_data_summary', methods=['GET'])
def api_get_data_summary():
    """Get data summary"""
    global df
    
    if df is None:
        return jsonify({'success': False, 'message': 'No data loaded'}), 400
    
    summary = get_initial_analysis()
    
    # Add to session messages
    if 'messages' not in session:
        session['messages'] = []
    
    session['messages'].append({
        'role': 'assistant',
        'content': summary,
        'timestamp': time.time()
    })
    
    return jsonify({
        'success': True,
        'summary': summary
    })

@app.route('/api/get_full_data', methods=['GET'])
@app.route('/api/get_full_data', methods=['GET'])
def api_get_full_data():
    global df
    if df is None:
        return jsonify({'success': False, 'message': 'No data loaded'}), 400

    try:
        df['Date'] = pd.to_datetime(df['Date'])  # Ensure proper date type

        data_to_send = df.to_dict(orient='records')

        # Compute metrics
        total_sales = df['Total'].sum()
        total_transactions = len(df)
        avg_rating = df['Rating'].mean() if 'Rating' in df.columns else 0

        # ✅ NEW: Average Sales per Day
        daily_sales = df.groupby('Date')['Total'].sum()
        avg_sales_per_day = daily_sales.mean()

        data_info = {
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': list(df.columns),
            'total_sales': total_sales,
            'total_transactions': total_transactions,
            'avg_sales_per_day': avg_sales_per_day,
            'avg_rating': avg_rating
        }

        return jsonify({
            'success': True,
            'data': data_to_send,
            'data_info': data_info
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f"Error retrieving full data: {str(e)}"}), 500

@app.route('/api/get_product_line_sales', methods=['GET'])
def api_get_product_line_sales():
    """Returns sales data by product line for the pie chart."""
    global df
    if df is None:
        return jsonify({'success': False, 'message': 'No data loaded'}), 400

    try:
        if 'Product line' in df.columns and 'Total' in df.columns:
            product_sales = df.groupby('Product line')['Total'].sum().reset_index()
            labels = product_sales['Product line'].tolist()
            data = product_sales['Total'].tolist()
            return jsonify({'success': True, 'labels': labels, 'data': data})
        else:
            return jsonify({'success': False, 'message': 'Required columns (Product line, Total) not found'}), 400
    except Exception as e:
        return jsonify({'success': False, 'message': f"Error calculating product line sales: {str(e)}"}), 500

@app.route('/api/status', methods=['GET'])
def api_status():
    """Get current status"""
    session_id = session.get('session_id')
    
    return jsonify({
        'session_id': session_id,
        'data_loaded': df is not None,
        'model_initialized': session_id in agents if session_id else False,
        'current_model': agents[session_id]['model'] if session_id and session_id in agents else None,
        'message_count': len(session.get('messages', []))
    })

@app.route('/api/get_city_sales', methods=['GET'])
def api_get_city_sales():
    """Returns sales data by city for the bar chart."""
    global df
    if df is None:
        return jsonify({'success': False, 'message': 'No data loaded'}), 400

    try:
        if 'City' in df.columns and 'Total' in df.columns:
            city_sales = df.groupby('City')['Total'].sum().reset_index()
            labels = city_sales['City'].tolist()
            data = city_sales['Total'].tolist()
            return jsonify({'success': True, 'labels': labels, 'data': data})
        else:
            return jsonify({'success': False, 'message': 'Required columns (City, Total) not found'}), 400
    except Exception as e:
        return jsonify({'success': False, 'message': f"Error calculating city sales: {str(e)}"}), 500

@app.route('/api/get_correlation_matrix', methods=['GET'])
def get_correlation_matrix():
    global df
    if df is None:
        return jsonify({'success': False, 'message': 'No data loaded'}), 400

    try:
        numeric_df = df.select_dtypes(include='number')
        corr_matrix = numeric_df.corr().round(2)
        return jsonify({
            'success': True,
            'labels': corr_matrix.columns.tolist(),
            'matrix': corr_matrix.values.tolist()
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f"Correlation error: {str(e)}"}), 500

@app.route('/api/get_time_series_sales', methods=['GET'])
def get_time_series_sales():
    global df
    if df is None:
        return jsonify({'success': False, 'message': 'No data loaded'}), 400

    try:
        df['Date'] = pd.to_datetime(df['Date'])
        ts = df.groupby('Date')['Total'].sum().reset_index()
        labels = ts['Date'].dt.strftime('%Y-%m-%d').tolist()
        values = ts['Total'].round(2).tolist()

        return jsonify({
            'success': True,
            'labels': labels,
            'data': values
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f"Time series error: {str(e)}"}), 500

@app.route('/api/get_gross_income_by_product_line', methods=['GET'])
def api_get_gross_income_by_product_line():
    """Returns gross income by product line for a bar chart."""
    global df
    if df is None:
        return jsonify({'success': False, 'message': 'No data loaded'}), 400

    try:
        # Try to find the correct column for gross income (case-insensitive, flexible)
        income_col = None
        for col in df.columns:
            if col.strip().lower() in ["gross income", "grossincome", "gross_income"]:
                income_col = col
                break
        product_col = None
        for col in df.columns:
            if col.strip().lower() in ["product line", "productline", "product_line"]:
                product_col = col
                break
        if product_col and income_col:
            gross_income_by_product = df.groupby(product_col)[income_col].sum().reset_index()
            labels = gross_income_by_product[product_col].tolist()
            data = gross_income_by_product[income_col].round(2).tolist()
            return jsonify({'success': True, 'labels': labels, 'data': data})
        else:
            return jsonify({'success': False, 'message': f'Required columns (Product line, Gross income) not found. Columns found: {list(df.columns)}'}), 400
    except Exception as e:
        return jsonify({'success': False, 'message': f"Error calculating gross income by product line: {str(e)}"}), 500

@app.route('/api/get_gross_margin_by_product_line', methods=['GET'])
def api_get_gross_margin_by_product_line():
    """Returns average gross margin % by product line for the bar chart."""
    global df
    if df is None:
        return jsonify({'success': False, 'message': 'No data loaded'}), 400

    try:
        # Try both possible column names for gross margin
        margin_col = None
        for col in df.columns:
            if col.strip().lower() in ["gross margin %", "gross margin percentage"]:
                margin_col = col
                break
        if 'Product line' in df.columns and margin_col:
            grouped = df.groupby('Product line')[margin_col].mean().reset_index()
            labels = grouped['Product line'].tolist()
            data = grouped[margin_col].round(2).tolist()
            return jsonify({'success': True, 'labels': labels, 'data': data})
        else:
            return jsonify({'success': False, 'message': 'Required columns (Product line, Gross margin %) not found'}), 400
    except Exception as e:
        return jsonify({'success': False, 'message': f"Error calculating gross margin: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)