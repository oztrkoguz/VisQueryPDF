import fitz 
import clip
import torch
from PIL import Image
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""

    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()

    return text


def summarize_text(text, llm):
    summary_prompt = PromptTemplate.from_template(
        "Summarize the following text in exactly 1 sentence:\n\n{text}\n\nProvide the summary below:"
    )

    summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

    summary = summary_chain.run({"text": text})
    
    return summary

def compute_image_text_similarity(image_path, text_summary, model, preprocess, device):
    image = Image.open(image_path)
    image = preprocess(image).unsqueeze(0).to(device)

    text_inputs = clip.tokenize([text_summary]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)


    with torch.no_grad():
        image_features = model.encode_image(image)
    image_features /= image_features.norm(dim=-1, keepdim=True)


    similarity = (image_features @ text_features.T).cpu().numpy()
    
    return similarity





 