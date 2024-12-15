import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import pyperclip

# Load model and tokenizer
@st.cache_resource
def load_model():
    model_path = "T5_Abstractive-20241208T134228Z-001\\T5_Abstractive"

    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    return model, tokenizer

model, tokenizer = load_model()

# Summarization function
def summarize_text(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True)
    summary_ids = model.generate(
        input_ids,
        max_length=150,  # Maximum length of summary
        min_length=50,   # Minimum length of summary
        length_penalty=2.0,
        num_beams=4,     # Beam search for better results
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Streamlit UI
st.title("Website Peringkasan Teks")
st.write("Masukkan suatu teks atau unggah file .txt untuk mendapatkan versi ringkasnya.")

# Sidebar for file upload
st.sidebar.header("Unggah File .txt")
uploaded_file = st.sidebar.file_uploader("format file: .txt", type=["txt"])

# Input text box
if uploaded_file is not None:
    try:
        # Read and decode the uploaded file
        input_text = uploaded_file.read().decode("utf-8")
        with st.sidebar:
            st.write("Isi file ditampikan di text area")
    except Exception as e:
        st.sidebar.error(f"Terjadi kesalahan saat membaca file: {e}")
        input_text = ""
else:
    input_text = ""

input_text = st.text_area("text area", input_text, height=300)

# Initialize Session State for Summary
if "summary" not in st.session_state:
    st.session_state.summary = ""

# Summarize button
if st.button("Ringkas"):
    if input_text.strip():
        with st.spinner("Menghasilkan ringkasan..."):
            st.session_state.summary = summarize_text(input_text)  # Store in session state
        st.success("Ringkasan berhasil dibuat!")
    else:
        st.error("Silahkan masukkan teks atau unggah file untuk diringkas.")

# Display the summary if available
if st.session_state.summary:
    st.subheader("Ringkasan:")
    ringkasan = st.session_state.summary
    st.write(st.session_state.summary)

    # Copy to clipboard button
    if st.button("📋 Salin Ringkasan"):
        try:
            pyperclip.copy(ringkasan)  # Salin ke clipboard
            st.success("Ringkasan berhasil disalin ke clipboard!")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat menyalin: {e}")