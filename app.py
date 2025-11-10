# ==========================================
# 🩺 app.py — Medical RAG Chatbot with Profile Page + Login + Chat History
# ==========================================

from flask import Flask, render_template, request, redirect, session, jsonify, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from dotenv import load_dotenv
import os

# LangChain imports
from src.helper import get_huggingface_embeddings
from src.prompt import system_prompt
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Pinecone
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone as PineconeClient

# ==========================================
# 🚀 Flask Configuration
# ==========================================
app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
db = SQLAlchemy(app)

# ==========================================
# 🧱 Database Models
# ==========================================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    sessions = db.relationship("ChatSession", backref="user", lazy=True)


class ChatSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200))
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    chats = db.relationship("Chat", backref="session", lazy=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Chat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey("chat_session.id"), nullable=False)
    sender = db.Column(db.String(10))  # 'user' or 'bot'
    message = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


# ==========================================
# 🔑 Load Environment Variables
# ==========================================
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ==========================================
# 🧠 Initialize RAG (Gemini + Pinecone)
# ==========================================
try:
    pc = PineconeClient(api_key=PINECONE_API_KEY)
    index_name = "medicalbot"
    embeddings = get_huggingface_embeddings()
    docsearch = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.3
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    print("✅ RAG Model initialized successfully!")
except Exception as e:
    print(f"⚠️ Error initializing RAG: {e}")

# ==========================================
# 🌐 Flask Routes
# ==========================================
@app.route("/")
def home():
    if "user_id" in session:
        return redirect(url_for("chat"))
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        hashed_pw = generate_password_hash(password)

        if User.query.filter_by(username=username).first():
            return "⚠️ Username already exists!"

        new_user = User(username=username, password=hashed_pw)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session["user_id"] = user.id
            session["username"] = user.username
            return redirect(url_for("chat"))
        else:
            return "❌ Invalid username or password!"
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ==========================================
# 👤 Profile Page Route
# ==========================================
@app.route("/profile")
def profile():
    if "user_id" not in session:
        return redirect(url_for("login"))
    
    user_id = session["user_id"]
    user = User.query.get(user_id)
    if not user:
        return redirect(url_for("login"))
    
    total_chats = ChatSession.query.filter_by(user_id=user_id).count()
    return render_template("profile.html", user=user, total_chats=total_chats)


# ==========================================
# 💬 Chat Routes
# ==========================================
@app.route("/chat")
def chat():
    if "user_id" not in session:
        return redirect(url_for("login"))

    user_id = session["user_id"]
    sessions = ChatSession.query.filter_by(user_id=user_id).order_by(ChatSession.created_at.desc()).all()

    if not sessions:
        new_session = ChatSession(user_id=user_id, title="New Chat")
        db.session.add(new_session)
        db.session.commit()
        sessions = [new_session]

    active_session = sessions[0]
    chats = Chat.query.filter_by(session_id=active_session.id).all()

    return render_template("chat.html", username=session["username"], sessions=sessions, active_session=active_session, chats=chats)


@app.route("/switch_session/<int:session_id>")
def switch_session(session_id):
    if "user_id" not in session:
        return redirect(url_for("login"))
    session_obj = ChatSession.query.get(session_id)
    chats = Chat.query.filter_by(session_id=session_id).all()
    sessions = ChatSession.query.filter_by(user_id=session["user_id"]).all()
    return render_template("chat.html", username=session["username"], sessions=sessions, active_session=session_obj, chats=chats)


@app.route("/new_session")
def new_session():
    if "user_id" not in session:
        return redirect(url_for("login"))
    new_sess = ChatSession(user_id=session["user_id"], title="New Chat")
    db.session.add(new_sess)
    db.session.commit()
    return redirect(url_for("chat"))


@app.route("/get", methods=["POST"])
def get_response():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 403

    user_msg = request.form["msg"]
    user_id = session["user_id"]

    session_obj = ChatSession.query.filter_by(user_id=user_id).order_by(ChatSession.created_at.desc()).first()
    chat_user = Chat(session_id=session_obj.id, sender="user", message=user_msg)
    db.session.add(chat_user)
    db.session.commit()

    try:
        response = rag_chain.invoke({"input": user_msg})
        answer = response.get("answer", "I couldn't find that information.")
    except Exception as e:
        answer = f"⚠️ Error: {e}"

    chat_bot = Chat(session_id=session_obj.id, sender="bot", message=answer)
    db.session.add(chat_bot)
    if session_obj.title == "New Chat":
        session_obj.title = f"Chat about {user_msg[:30]}"
    db.session.commit()

    return str(answer)


# ==========================================
# ▶️ Run Flask
# ==========================================
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    print("🚀 Starting Medical Chatbot...")
    app.run(host="0.0.0.0", port=8080, debug=True)
