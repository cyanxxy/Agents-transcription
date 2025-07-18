import streamlit as st

def apply_custom_styles():
    """Applies custom CSS styles to the Streamlit app."""
    custom_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        body, .stApp, .stButton > button, .stTextInput input, .stTextArea textarea, .stSelectbox select, .stRadio div, .stFileUploader label, .streamlit-expanderHeader, .stMarkdown p, .stCaption p {
            font-family: 'Inter', sans-serif !important;
        }

        /* Headings */
        h1 {
            text-align: center;
            margin-bottom: 30px !important; /* Ensure override if needed */
            color: #1E88E5 !important;
            font-weight: 700 !important;
            font-size: 2.5rem; /* Example size, adjust as needed */
        }
        h4 { /* For section titles */
            font-weight: 600 !important;
            color: #333333 !important; /* Darker for better contrast */
            margin-bottom: 15px !important;
            font-size: 1.25rem;
        }

        /* General Layout & Containers */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .styled-container {
            background-color: #f8f9fa; /* Light gray background */
            padding: 20px; /* Increased padding */
            border-radius: 8px;
            border: 1px solid #dee2e6; /* Slightly softer border */
            margin-bottom: 25px; /* Increased consistent bottom margin */
            box-shadow: 0 2px 4px rgba(0,0,0,0.05); /* Subtle shadow */
        }

        /* Buttons */
        div.stButton > button { /* General button style */
            width: 100%; /* Make buttons take full width of their container */
            border-radius: 6px !important; /* Slightly more rounded */
            font-weight: 500 !important;
            padding: 0.6rem 1rem !important;
            transition: background-color 0.2s ease-in-out, box-shadow 0.2s ease-in-out, border-color 0.2s ease-in-out !important;
            border: 1px solid transparent; /* Base border */
        }

        /* Primary buttons */
        div.stButton > button[kind="primary"] {
            background-color: #1E88E5 !important;
            color: white !important;
            border-color: #1E88E5 !important;
        }
        div.stButton > button[kind="primary"]:hover {
            background-color: #1A75C4 !important; /* Darker blue on hover */
            border-color: #1A75C4 !important;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
        }
        div.stButton > button[kind="primary"]:active {
            background-color: #1665A5 !important; /* Even darker on active */
            border-color: #1665A5 !important;
        }

        /* Secondary/Default buttons (Streamlit often uses default styling if not primary) */
        /* This targets buttons that are NOT primary. More specific selectors might be needed if you have other button types. */
        div.stButton > button:not([kind="primary"]) {
            background-color: #ffffff !important;
            color: #1E88E5 !important;
            border: 1px solid #1E88E5 !important;
        }
        div.stButton > button:not([kind="primary"]):hover {
            background-color: #f0f7ff !important; /* Light blueish tint on hover */
            border-color: #1A75C4 !important;
            color: #1A75C4 !important;
        }

        /* File Uploader Label */
        div[data-testid="stFileUploader"] > label {
            font-weight: 500 !important;
            color: #333333 !important;
            font-size: 1rem; /* Consistent font size */
        }

        /* Radio Buttons */
        .stRadio > label > div:first-child { /* Targets the label text of the radio group */
            font-weight: 600 !important; /* Makes 'Select Transcription Model' title bold */
            color: #333333 !important;
            margin-bottom: 10px !important; /* Space below title */
        }
        .stRadio > div[role="radiogroup"] > label { /* Target individual radio items */
            background-color: #f0f3f5;
            padding: 8px 12px;
            border-radius: 6px;
            margin-right: 10px; /* Spacing for horizontal layout */
            transition: background-color 0.2s ease, border-color 0.2s ease;
            border: 1px solid transparent;
        }
        .stRadio > div[role="radiogroup"] > label:hover {
            background-color: #e0e5e9;
        }
        .stRadio > div[role="radiogroup"] > label input[type="radio"]:checked + div {
            font-weight: 600 !important; /* Make selected text bolder */
        }
        .stRadio > div[role="radiogroup"] > label[data-is-checked="true"] {
             background-color: #ddeaff !important; /* Light blue for selected */
             border: 1px solid #1E88E5 !important;
             color: #1E88E5 !important;
        }

        /* Alerts */
        .stAlert {
            padding: 0.75rem;
            border-radius: 4px;
            margin-bottom: 1rem;
        }

        /* Tabs */
        .stTabs [aria-selected="true"] {
            background-color: #1E88E5; /* Blue accent for selected tab */
            color: white;
            font-weight: 600;
            border-radius: 6px 6px 0 0; /* Match tab radius */
        }

        /* Expander */
        .streamlit-expanderHeader {
            font-size: 1rem;
            font-weight: 500;
        }

        /* Transcript */
        .timestamp {
            font-weight: bold;
            color: #007bff; /* Blue for timestamps */
        }
        .speaker {
            font-weight: bold;
            color: #28a745; /* Green for speakers */
        }
        .special-event {
            font-style: italic;
            color: #5a6268; /* Darker gray for special events */
        }
        .transcript-container p, .stMarkdown p { /* General paragraph styling within transcript display */
            margin-bottom: 0.75em !important;
            line-height: 1.7 !important;
        }

        /* Hide default Streamlit footer */
        footer {display: none;}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def format_transcript_line(line: str) -> str:
    """Format a transcript line with styled timestamps and speakers"""
    if '[' in line and ']' in line:
        timestamp = line[line.find('['): line.find(']') + 1]
        remaining = line[line.find(']') + 1:].strip()
        
        if '[MUSIC]' in line or '[JINGLE]' in line or 'Sound' in line:
            return f'<span class="timestamp">{timestamp}</span> <span class="special-event">{remaining}</span>'
        
        if ':' in remaining:
            speaker, text = remaining.split(':', 1)
            return f'<span class="timestamp">{timestamp}</span> <span class="speaker">{speaker}</span>:{text}'
        
    return line
