import torch
import clip
from langchain.llms import Ollama
from describe_image_agents import extract_images_from_pdf,describe_image
from generating_questions_agent import generate_questions
from rag_agent import rag_query
from ebbeding_similarity import extract_text_from_pdf,summarize_text,compute_image_text_similarity
 
pdf_path = r"example\document.pdf"
output_folder = "images"

llm = Ollama(model="llama3")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 1.method

#step1

images_paths = extract_images_from_pdf(pdf_path, output_folder)
descriptions = describe_image(images_paths)

print("Descriptions of images:\n")
print(descriptions)

#step2

questions = generate_questions(descriptions,llm)

print("Generate questions:\n")
print(questions)

#step3
answer_list = []
for question in questions:
    answer_list.append(rag_query(pdf_path, question,llm))

print("Rag result:\n")
print(answer_list)


# 2.method

#step1
text = extract_text_from_pdf(pdf_path)

#step2
summarize = summarize_text(text,llm)
print("summarize result:\n")
print(summarize)

#step3
image_path = r"images\image_0_0.png"
similarity = compute_image_text_similarity(image_path, summarize, model, preprocess, device)
print("similarity result:\n")
print(similarity)

