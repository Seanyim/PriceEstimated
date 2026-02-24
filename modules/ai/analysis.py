
import streamlit as st
import google.generativeai as genai
import pandas as pd

from modules.core.db import get_company_meta, get_market_history

def generate_report_markdown(ticker, df_raw):
    """
    Generates a structured markdown report for the given ticker and financial data.
    """
    meta = get_company_meta(ticker)
    company_name = meta.get('name', ticker)
    region = meta.get('region', 'Unknown')
    unit = meta.get('unit', 'Unknown')
    
    markdown_report = f"# {company_name} ({ticker}) - Financial Analysis Report\n\n"
    markdown_report += f"**Region:** {region} | **Reporting Unit:** {unit}\n\n"
    
    markdown_report += "## Financial Summary\n\n"
    if not df_raw.empty:
        # Sort by date descending for the report
        df_sorted = df_raw.sort_values(by='report_date', ascending=False)
        markdown_report += df_sorted.head(5).to_markdown(index=False)
    else:
        markdown_report += "No financial data available.\n"
    
    markdown_report += "\n\n## Valuation Models\n"
    markdown_report += "*(Include valuation model outputs here in future enhancements)*\n"
    
    markdown_report += "\n## Market Data\n"
    df_market = get_market_history(ticker)
    if not df_market.empty:
        df_market_sorted = df_market.sort_values(by='date', ascending=False)
        latest_market = df_market_sorted.head(1).to_dict('records')[0]
        markdown_report += f"**Latest Date:** {latest_market.get('date', 'N/A')}\n"
        markdown_report += f"**Close Price:** {latest_market.get('close', 'N/A')}\n"
        markdown_report += f"**Market Cap:** {latest_market.get('market_cap', 'N/A')}\n"
        markdown_report += f"**PE (TTM):** {latest_market.get('pe_ttm', 'N/A')}\n"
        
        markdown_report += "\n### Recent Market History (Last 5 Days)\n"
        markdown_report += df_market_sorted.head(5)[['date', 'close', 'volume', 'pe_ttm']].to_markdown(index=False)
    else:
        markdown_report += "No market data available.\n"

    return markdown_report

def init_gemini_chat(api_key, model_name="gemini-1.5-pro"):
    """
    Initializes the Gemini chat session.
    """
    if not api_key:
        return None
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    chat = model.start_chat(history=[])
    return chat

def render_ai_tab(ticker, df_raw):
    """
    Renders the AI Analysis tab content.
    """
    st.header(f"🤖 AI Analysis: {ticker}")
    
    # 1. Configuration
    with st.expander("⚙️ AI Configuration", expanded=False):
        api_key = st.text_input("Gemini API Key", type="password", help="Get your key from Google AI Studio")
        model_name = st.selectbox("Select Model", ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash-exp"], index=0)
        
        if api_key:
             st.session_state['gemini_api_key'] = api_key
             st.session_state['gemini_model_name'] = model_name
    
    # Check for API key in session state
    if 'gemini_api_key' not in st.session_state:
        st.info("Please enter your Gemini API Key in the configuration section above to start the AI chat.")
    
    # 2. Markdown Generation
    st.subheader("📄 Financial Report Structure")
    
    if st.button("Generate Markdown Report"):
        report_md = generate_report_markdown(ticker, df_raw)
        st.session_state['report_md'] = report_md
        
    if 'report_md' in st.session_state:
        st.text_area("Markdown Report", st.session_state['report_md'], height=300)
        st.download_button(
            label="Download Markdown",
            data=st.session_state['report_md'],
            file_name=f"{ticker}_analysis_report.md",
            mime="text/markdown"
        )
    
    st.divider()

    # 3. AI Chat Interface
    st.subheader("💬 Chat with AI")
    
    if 'gemini_api_key' in st.session_state:
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        # Initialize chat session if not exists or if model changed (simplified handling)
        if "chat_session" not in st.session_state:
             st.session_state.chat_session = init_gemini_chat(st.session_state['gemini_api_key'], st.session_state['gemini_model_name'])
             # Inject context if report is generated
             if 'report_md' in st.session_state:
                 try:
                    st.session_state.chat_session.send_message(f"Here is the financial report for {ticker}:\n\n{st.session_state['report_md']}\n\nPlease use this context to answer my questions.")
                 except Exception as e:
                     st.error(f"Failed to initialize context: {e}")

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask AI about this company..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            try:
                if st.session_state.chat_session:
                    response = st.session_state.chat_session.send_message(prompt)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                    with st.chat_message("assistant"):
                        st.markdown(response.text)
            except Exception as e:
                st.error(f"Error communicating with Gemini: {e}")
