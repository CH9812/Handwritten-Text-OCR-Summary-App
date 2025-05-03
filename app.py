import streamlit as st
import base64
import json
import time
from mistralai import Mistral
from transformers import pipeline

# Setup Streamlit page
st.set_page_config(layout="wide", page_title="OCR & Summarizer", page_icon="🖥️")
st.title("📄 Handwritten Text OCR + Summary App")

# Step 1: API Key
api_key = st.text_input("Enter your Mistral API Key", type="password")
if not api_key:
    st.warning("Please enter your API key to continue.")
    st.stop()

# Step 2: Input type
file_type = st.radio("Select file type", ["PDF", "Image"])
input_mode = st.radio("Select input mode", ["URL", "Upload"])

# Step 3: Upload or URL input
urls, files = [], []
if input_mode == "URL":
    urls = st.text_area("Enter one or more URLs (new line for each)").splitlines()
else:
    files = st.file_uploader("Upload one or more files", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True)

# Step 4: Summarization pipeline using Hugging Face (Free, local)
summarizer = pipeline("summarization", model="t5-small")

# Step 5: Process files
if st.button("Run OCR + Summarize"):
    client = Mistral(api_key=api_key)
    
    inputs = urls if input_mode == "URL" else files
    for idx, item in enumerate(inputs):
        st.subheader(f"📄 Document {idx + 1}")
        preview_src, document = None, None

        if input_mode == "Upload":
            bytes_data = item.read()
            mime_type = item.type
            encoded = base64.b64encode(bytes_data).decode("utf-8")
            document = {
                "type": "image_url" if file_type == "Image" else "document_url",
                "image_url" if file_type == "Image" else "document_url": f"data:{mime_type};base64,{encoded}"
            }
            preview_src = f"data:{mime_type};base64,{encoded}"
        else:
            url = item.strip()
            document = {
                "type": "image_url" if file_type == "Image" else "document_url",
                "image_url" if file_type == "Image" else "document_url": url
            }
            preview_src = url

        # Show preview
        if file_type == "PDF":
            st.markdown(f'<iframe src="{preview_src}" width="100%" height="600"></iframe>', unsafe_allow_html=True)
        else:
            st.image(preview_src, caption="Input Image", use_column_width=True)

        with st.spinner("Extracting OCR text..."):
            try:
                response = client.ocr.process(
                    model="mistral-ocr-latest",
                    document=document,
                    include_image_base64=False,
                )
                pages = response.pages if hasattr(response, "pages") else []
                ocr_text = "\n\n".join(page.markdown for page in pages)
            except Exception as e:
                st.error(f"OCR failed: {e}")
                continue

        st.subheader("📌 OCR Result")
        st.text_area("Extracted Text", ocr_text, height=300)

        with st.spinner("Generating Summary..."):
            try:
                # Summarize in chunks if text is long
                if len(ocr_text.strip()) < 10:
                    summary_text = "Not enough text to summarize."
                else:
                    chunks = [ocr_text[i:i+800] for i in range(0, len(ocr_text), 800)]
                    summary_parts = [summarizer(chunk, max_length=120, min_length=30, do_sample=False)[0]["summary_text"] for chunk in chunks]
                    summary_text = " ".join(summary_parts)
            except Exception as e:
                st.error(f"Summarization failed: {e}")
                continue

        st.subheader("📚 Summary")
        st.text_area("Generated Summary", summary_text, height=200)

        # Prepare downloads
        output_json = json.dumps({"ocr_text": ocr_text, "summary": summary_text}, indent=2)
        st.download_button("📥 Download OCR & Summary (JSON)", output_json, file_name=f"ocr_summary_{idx+1}.json", mime="application/json")
        st.download_button("📥 Download OCR Only", ocr_text, file_name=f"ocr_text_{idx+1}.txt")
        st.download_button("📥 Download Summary Only", summary_text, file_name=f"summary_{idx+1}.txt")
