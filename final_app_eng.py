import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import streamlit as st
import os
from dotenv import load_dotenv
import base64
import time
# Load .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(
    page_title="TechPro Data Science and Mentorship Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
# Load database
def load_database(file_path):
    return pd.read_excel(file_path)

# Keyword check
def is_relevant_topic(question):
    keywords = ["data science", "mentoring", "machine learning", "deep learning", "EDA", "SQL", "Python", "bootcamp"]
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in keywords)

# Create and train TF-IDF model
def train_tfidf_model(questions):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(questions)
    return tfidf_vectorizer, tfidf_matrix
    
# Get response from OpenAI GPT-3.5 Turbo
def get_response_from_gpt(question):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant that answers questions related to data science, mentoring, and related technical topics only. Do not respond to questions outside these topics."},
                {"role": "user", "content": question}
            ],
            temperature=0.7
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"An error occurred with the GPT service: {e}"
        
# Find the best matching answer for user's question
def find_best_match(user_question, tfidf_vectorizer, tfidf_matrix, df, threshold=0.6):
    user_tfidf = tfidf_vectorizer.transform([user_question])
    similarity_scores = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    best_match_idx = similarity_scores.argmax()
    best_score = similarity_scores[best_match_idx]
    if best_score >= threshold:
        return df.iloc[best_match_idx]['Answer'], best_score
    else:
        return None, best_score

### STREAMLIT CODE

# Custom CSS for Streamlit
def render_custom_css():
    st.markdown("""
        <style>
        [data-testid="stSidebar"] img {
            width: 300px;
            margin: 0 auto;
            display: block;
        }
        .main-header {
            background: linear-gradient(90deg, #e1eaf2 0%, #56c456 100%);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-top: 20px;
        }
        .main-header h1 {
            color: #003366;
            font-size: 46px;
            font-weight: bold;
        }
        .chat-message {
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
            flex-direction: column;
        }
        .user-message {
            background-color: #e3f2fd;
            margin-left: 2rem;
            border-left: 5px solid #2196f3;
        }
        .bot-message {
            background-color: #f5f5f5;
            margin-right: 2rem;
            border-left: 5px solid #56c456;
        }
        .assistant-answer {
            background: #e1f5fe;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            font-size: 16px;
        }
        .user-question {
            background: #f1f8e9;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            font-size: 16px;
            font-weight: bold;
        }
        .stButton>button {
            background-color: #56c456;
            color: white;
            font-size: 16px;
            border-radius: 5px;
            transition: all 0.3s ease;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .stButton>button:active {
            transform: translateY(0);
        }
        </style>
    """, unsafe_allow_html=True)

def render_sidebar():
    logo_path = "logoforchatbot.png"

    if os.path.exists(logo_path):
        st.sidebar.image(logo_path, use_column_width=True)
    else:
        st.sidebar.warning("Logo not found! Please check the file path.")

    # Information box
    st.sidebar.markdown("""
    <style>
        .info-box {
            background-color: #f0f8ff;
            padding: 15px;
            border-radius: 10px;
            border-left: 6px solid #56c456;
            font-size: 14px;
            margin-bottom: 20px;
        }
    </style>
    
    <div class="info-box">
        <h3>Hello! 👋</h3>
        <p>
            I'm <b>Techwise</b>, here to guide you through data science and mentorship topics! 🤓
        </p>
        <p>
            🎓 I can answer questions about bootcamps, data science, machine learning, and Python. I can also provide mentorship tips.
        <p>
             Let's learn together! 😊
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Minimalist file upload area
    uploaded_file = st.sidebar.file_uploader(
        label="",
        type=["xlsx"],
    )
    # Show message if file is uploaded
    if uploaded_file:
        st.sidebar.markdown("""
        <div style="text-align: center; padding: 10px; border-radius: 5px; background-color: #e8f5e9; border: 1px solid #56c456;">
            <p style="margin: 0; font-size: 14px; color: #333;"><b>✅ File uploaded!</b></p>
        </div>
        """, unsafe_allow_html=True)
        faq_data = load_database(uploaded_file)
    else:
        st.sidebar.markdown("""
        <div style="text-align: center; padding: 10px; border-radius: 5px; background-color: #f9f9f9; border: 1px dashed #cccccc;">
            <p style="margin: 0; font-size: 14px; color: #666;">📂 <b>Upload your file</b></p>
        </div>
        """, unsafe_allow_html=True)
        faq_data = None

    # Show Frequently Asked Questions if file is uploaded
    if faq_data is not None:
        st.sidebar.markdown("""
        <style>
            .faq-box {
                background-color: #e1f5fe;
                padding: 10px;
                border-radius: 10px;
                border-left: 4px solid #2196f3;
                margin-top: 20px;
            }
        </style>
        <div class="faq-box">
            <h3 style="margin: 0; color: #2196f3;">📚 Frequently Asked Questions</h3>
        </div>
        """, unsafe_allow_html=True)
        questions = faq_data["Questions"].tolist()
        selected_question = st.sidebar.selectbox("Select a question:", [""] + questions)
        return faq_data, selected_question

    # Reset Questions Button
    if st.sidebar.button("🔄 Reset Questions"):
        st.session_state.chat_history = []
        st.sidebar.success("Questions reset!")

    return faq_data, None

# Main workflow
def main():
    st.markdown("""
    <style>
    .social-icons {
        text-align: center; /* Center icons horizontally */
        margin-top: 20px; /* Add top margin */
    }
    .social-icons a {
        margin: 0 15px; /* Add 15px margin to left and right */
        display: inline-block; /* Make icons appear in a row */
    }
    </style>
    <div class="social-icons">
        <a href="https://www.linkedin.com/in/evolkan/" target="_blank">
            <img src="https://img.icons8.com/ios-filled/25/000000/linkedin.png" alt="LinkedIn">
        </a>
        <a href="https://www.kaggle.com/esravo" target="_blank">
            <img src="https://cdn.jsdelivr.net/npm/simple-icons@v9/icons/kaggle.svg" alt="Kaggle" width="50" height="50">
        </a>
        <a href="https://github.com/esravolkan" target="_blank">
            <img src="https://img.icons8.com/ios-filled/25/000000/github.png" alt="GitHub">
        </a>
    </div>
    """, unsafe_allow_html=True)

        
    st.markdown('<div class="main-header"><h1>Techwise: Data Science and Mentorship Assistant 👩‍💻</h1></div>', unsafe_allow_html=True)
    render_custom_css()
    
    # Add Tip to main screen (under the header)
    with st.expander("💡 **Tip: For better results...**"):
        st.markdown("""
    - Try to use a clear and understandable language when asking your questions. Express yourself concisely so I can help you in the best way possible. 😊
    - When asking a technical question, it's important to use the correct terminology.
    - Example: "What is EDA?" or "How do for loops work in Python?"

    🔔 **Note**: If you're going to ask a question not in the database, 
    please try to write a question containing the following keywords:
    
    - **Data science** (e.g., "How can I access data science articles?")
    - **Mentoring** (e.g., "How to develop mentoring skills?")
    - **Bootcamp** (e.g., "How should I prepare for a bootcamp?")
    - Technical terms like **Python**, **EDA**, **SQL**, **Machine Learning**, **Deep Learning**.

    📌 Remember: The more accurate and clear your question is, the faster and more accurate answer I can provide! 🎯
    """)
        
    # Load sidebar
    faq_data, selected_question = render_sidebar()

    if faq_data is not None:
        try:
            tfidf_vectorizer, tfidf_matrix = train_tfidf_model(faq_data['Questions'])
            user_question = st.text_input(
                "Select a question or write your own:",
                value=selected_question if selected_question else ""
            )
            
            if st.button("Get Answer"):
                if not user_question.strip():
                    st.warning("Please write a question.")
                    return

                with st.spinner("Preparing answer..."):
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    st.markdown(f"""
                        <div class="chat-message user-message">
                            <strong>Question:</strong> {user_question}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    matched_answer, match_score = find_best_match(
                        user_question, tfidf_vectorizer, tfidf_matrix, faq_data
                    )
                    
                    if matched_answer:
                        st.markdown(f"""
                            <div class="chat-message bot-message">
                                <strong>Answer:</strong> {matched_answer}
                            </div>
                        """, unsafe_allow_html=True)
                        
                        st.session_state.chat_history.append({
                            "question": user_question,
                            "answer": matched_answer,
                            "type": "database"
                        })
                    elif is_relevant_topic(user_question):
                        gpt_answer = get_response_from_gpt(user_question)
                        st.markdown(f"""
                            <div class="chat-message bot-message">
                                <strong>GPT-3.5 Answer:</strong> {gpt_answer}
                            </div>
                        """, unsafe_allow_html=True)
                        
                        st.session_state.chat_history.append({
                            "question": user_question,
                            "answer": gpt_answer,
                            "type": "gpt"
                        })
                    else:
                        st.warning("No answer can be provided for this topic. Please focus on Data Science, Mentorship, or Bootcamp topics.")
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()