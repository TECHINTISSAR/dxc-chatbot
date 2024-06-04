import os
import io
import pymongo
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from datetime import datetime, timedelta
import tempfile
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.chains import load_summarize_chain
from langchain.prompts import PromptTemplate
import hashlib


def setup_environment_and_db():
    load_dotenv()
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mydb = myclient["dxc_chat_bot2"]
    col1 = mydb["chunks"]
    col2 = mydb["conversation"]
    col3 = mydb["col_documents_pdf"]
    return col1, col2, col3

def get_pdf_text(pdf_bytes_list):
    text = ""
    for pdf_bytes in pdf_bytes_list:
        pdf_stream = io.BytesIO(pdf_bytes)
        pdf_reader = PdfReader(pdf_stream)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore
from streamlit_option_menu import option_menu

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question, chat_history_collection):
    response = st.session_state.conversation({'question': user_question})
    
    # Get the current date and time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Convert response object into a dictionary
    response_dict = {
        "question": user_question,
        "response": {
            "chat_history": [msg.content for msg in response['chat_history']],
            "timestamp": current_time
        }
    }

    chat_history_collection.insert_one(response_dict)

    for i, message in enumerate(response['chat_history']):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            # Handle long responses by splitting them into smaller parts
            full_message = message.content
            max_length = 1000  # Define the maximum length of a single message part
            parts = [full_message[i:i+max_length] for i in range(0, len(full_message), max_length)]
            for part in parts:
                st.write(bot_template.replace("{{MSG}}", part), unsafe_allow_html=True)

def display_chat_history(chat_history_collection):
    st.subheader("Chat History")
    for entry in chat_history_collection.find():
        st.markdown('<div class="chat-history-frame">', unsafe_allow_html=True)
        if "timestamp" in entry["response"]:
            st.write("Timestamp:", entry["response"]["timestamp"])
        else:
            st.write("Timestamp: N/A")
        st.write("User:", entry["question"])
        st.write("Bot:", entry["response"]["chat_history"][-1])
        st.markdown('</div>', unsafe_allow_html=True)

def cleanup_old_history(chat_history_collection):
    threshold_date = datetime.now() - timedelta(days=5)
    result = chat_history_collection.delete_many({"response.timestamp": {"$lt": threshold_date.strftime("%Y-%m-%d %H:%M:%S")}})
    print(f"Deleted {result.deleted_count} old chat history entries.")

def download_chat_history(chat_history_collection):
    chat_history_text = ""
    for entry in chat_history_collection.find():
        chat_history_text += "User: " + entry["question"] + "\n"
        chat_history_text += "Bot: " + entry["response"]["chat_history"][-1] + "\n"
        if "timestamp" in entry["response"]:
            chat_history_text += "Timestamp: " + entry["response"]["timestamp"] + "\n\n"
        else:
            chat_history_text += "Timestamp: N/A\n\n"
    return chat_history_text

def process_and_save_pdf_info(pdf_docs, col_documents_pdf):
    for pdf in pdf_docs:
        existing_doc = col_documents_pdf.find_one({'file_name': pdf.name})
        if existing_doc is None:
            col_documents_pdf.insert_one({
                'file_name': pdf.name,
                'processing_date': datetime.now(),
            })

def summarize_pdfs_from_folder(pdfs_folder):
    summaries = []
    for pdf_file in pdfs_folder:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(pdf_file.read())
        
        loader = PyPDFLoader(temp_path)
        text = loader.load_and_split()
        prompt_template = """Écrivez un résumé concis du texte suivant délimité par des triples
        guillemets inversés.
        Renvoyez votre réponse sous forme de points, couvrant les points clés du texte.
        اكتب ملخصًا موجزًا للنص التالي المحدد بواسطة علامات الاقتباس الثلاثية العكسية.
أعد إجابتك على شكل نقاط، تغطي النقاط الرئيسية للنص.

        Please, write a concise summary of the following text delimited by triple backquotes.
        Return your response in bullet points which covers the key points of the text.
        ```{text}```
        RÉSUMÉ EN POINTS / BULLET POINT SUMMARY/ ملخص بالنقاط:
        """

        prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
        
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106")
        chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
        summary = chain.run(text)
        summaries.append(summary)

        os.remove(temp_path)
    return summaries

def get_pdf_text_from_mongo(col_documents_pdf):
    text = ""
    for pdf_doc in col_documents_pdf.find():
        pdf_bytes = pdf_doc['pdf']
        pdf_stream = io.BytesIO(pdf_bytes)
        pdf_reader = PdfReader(pdf_stream)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_conversation_chain_from_mongo(col_documents_pdf):
    raw_text = get_pdf_text_from_mongo(col_documents_pdf)
    text_chunks = get_text_chunks(raw_text)
    
    vectorstore = get_vectorstore(text_chunks)
    llm = ChatOpenAI()
    
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

client = pymongo.MongoClient("mongodb://localhost:27017/DATA")
db = client["MYdatabase"]
users = db["Users"]

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def check_password(entered_password, stored_password):
    return hash_password(entered_password) == stored_password

def sign_up_with_email_and_password(email, password, username=None):
    if users.find_one({"email": email}):
        return "User already exists"
    else:
        hashed_password = hash_password(password)
        user = {
            "email": email,
            "password": hashed_password,
            "username": username
        }
        users.insert_one(user)
        return "User registered successfully"

def sign_in_with_email_and_password(email, password):
    user = users.find_one({"email": email})
    if user and check_password(password, user["password"]):
        return {"email": user["email"], "username": user.get("username")}
    else:
        return "Invalid credentials"

def check_email_in_database(email):
    return users.find_one({"email": email}) is not None

def update_password(email, new_password):
    if check_email_in_database(email):
        hashed_password = hash_password(new_password)
        result = users.update_one(
            {"email": email},
            {"$set": {"password": hashed_password}}
        )
        if result.modified_count > 0:
            return "Password has been reset successfully"
        else:
            return "No update was made, password might be the same or update failed"
    return "Email not found in our records"

def main():
    col_chunks, col_conversation, col_documents_pdf = setup_environment_and_db()
    cleanup_old_history(col_conversation)

    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":keyboard:")
    st.markdown(css, unsafe_allow_html=True)

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        st.image('dc.jpg')
        st.markdown("<h2 class='title'>Bienvenue chez DXC LAWBOT</h2>", unsafe_allow_html=True)

        choice = st.selectbox('Se connecter/Créer un Compte', ['Se connecter', 'Créer un Compte'])
        email = st.text_input('Email Address')
        password = st.text_input('Password', type='password')

        if choice == 'Créer un Compte':
            username = st.text_input("Saisir votre nom")
            if st.button('Créer un Compte'):
                message = sign_up_with_email_and_password(email, password, username)
                st.success(message)
                if message == "Votre compte est crée":
                    st.balloons()
        elif choice == 'Se connecter':
            if st.button('Se connecter'):
                result = sign_in_with_email_and_password(email, password)
                if isinstance(result, dict):
                    st.success('Connexion réussi')
                    st.write(f"Bienvenue, {result['username']}!")
                    st.session_state.logged_in = True
                    st.session_state.username = result['username']
                else:
                    st.error(result)

    if st.session_state.logged_in:
        with st.sidebar:
            st.image('dc.jpg')
            app = option_menu(
                menu_title='Pondering',
                options=['Ask a Question', 'View History', 'View PDF History'],
                icons=['question-circle', 'clock-history', 'file-earmark-pdf'],
                menu_icon='chat-text-fill',
                default_index=0,
                styles={
                    "container": {"padding": "5!important", "background-color": '#d8bdd8'},
                    "icon": {"color": "white", "font-size": "23px"},
                    "nav-link": {"color": "white", "font-size": "20px", "text-align": "left", "margin": "0px", "--hover-color": "purple"},
                    "nav-link-selected": {"background-color": "##C285C2"},
                }
            )

        if app == 'Ask a Question':
            sub_option = option_menu(
                menu_title='',
                options=['Upload new documents', 'Select existing documents','Initialize Conversation with Stored PDFs'],
                icons=['upload', 'file'],
                menu_icon='',
                default_index=0,
                styles={
                    "container": {"padding": "0!important", "background-color": '#d8bdd8'},
                    "icon": {"color": "white", "font-size": "20px"},
                    "nav-link": {"color": "white", "font-size": "18px", "text-align": "left", "margin": "0px", "--hover-color": "purple"},
                    "nav-link-selected": {"background-color": "##C285C2"},
                }
            )

            st.header("Chat with PDFs :keyboard:")
            
            with st.container():
                cols = st.columns([5, 1, 1])
                user_question = st.chat_input("Ask a question about your documents:")
                generate_summary_clicked = cols[1].button("Résumé")
                process_clicked = cols[2].button("Process")

            if user_question:
                handle_userinput(user_question, col_conversation)

            # Initialize pdf_docs to avoid UnboundLocalError
            pdf_docs = None
            pdf_bytes = None
            existing_files = []

            # Secondary menu for uploading/selecting documents
            if sub_option == "Upload new documents":
                st.sidebar.subheader("Upload new documents")
                pdf_docs = st.sidebar.file_uploader("Upload new PDFs here:", accept_multiple_files=True)
            elif sub_option == "Select existing documents":
                st.sidebar.subheader("Or select existing documents")
                existing_files = [doc["filename"] for doc in col_documents_pdf.find({}, {"filename": 1})]
                selected_files = st.sidebar.multiselect("Select documents to ask questions about:", existing_files)
                pdf_bytes = [col_documents_pdf.find_one({"filename": filename})["pdf"] for filename in selected_files] if existing_files else None

            st.markdown('<div class="button-container">', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if generate_summary_clicked:
                st.write("Summaries:")
                if pdf_docs:
                    summaries = summarize_pdfs_from_folder(pdf_docs)
                else:
                    pdf_bytes_list = [io.BytesIO(pdf) for pdf in pdf_bytes]
                    summaries = summarize_pdfs_from_folder(pdf_bytes_list)
                for i, summary in enumerate(summaries):
                    st.write(f"Summary for PDF {i + 1}:")
                    st.write(summary)

            if process_clicked:
                with st.spinner("Processing"):
                    pdf_bytes_list = []
                    if pdf_docs:
                        for pdf in pdf_docs:
                            filename = pdf.name
                            if filename not in existing_files:
                                pdf_content = pdf.read()
                                pdf_bytes_list.append(pdf_content)
                                col_documents_pdf.insert_one({"filename": filename, "pdf": pdf_content})
                                existing_files.append(filename)

                    if pdf_bytes:
                        for filename in selected_files:
                            pdf_content = col_documents_pdf.find_one({"filename": filename})["pdf"]
                            pdf_bytes_list.append(pdf_content)

                    if pdf_bytes_list:
                        raw_text = get_pdf_text(pdf_bytes_list)
                        text_chunks = get_text_chunks(raw_text)

                        for chunk in text_chunks:
                            col_chunks.insert_one({"chunk": chunk})

                        vectorstore = get_vectorstore(text_chunks)
                        st.session_state.conversation = get_conversation_chain(vectorstore)

                        st.success("Processing completed.")

            st.markdown("---")

            # Section pour initialiser la conversation avec les PDF stockés
            if sub_option == "Initialize Conversation with Stored PDFs":
                    with st.spinner("Initializing conversation..."):
                        st.session_state.conversation = get_conversation_chain_from_mongo(col_documents_pdf)
                        st.success("Conversation initialized with stored PDFs")

        elif app == 'View History':
            st.header("Chat History")
            display_chat_history(col_conversation)

        elif app == 'View PDF History':
            st.header("PDF History")
            pdf_entries = col_documents_pdf.find()
            for pdf in pdf_entries:
                st.markdown('<div class="chat-history-frame">', unsafe_allow_html=True)
                st.write("File Name:", pdf['filename'])
                if 'processing_date' in pdf:
                    date_uploaded = pdf['processing_date'].strftime("%Y-%m-%d %H:%M:%S")
                    st.write("Date Uploaded:", date_uploaded)

                pdf_bytes = pdf['pdf']
                st.download_button(label="Download PDF", data=pdf_bytes, file_name=pdf['filename'], mime='application/pdf')
                st.markdown('</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
