import fitz 
import io
from PIL import Image
import ollama
import os

def extract_images_from_pdf(pdf_path, output_folder):
    pdf_document = fitz.open(pdf_path)
    images = []
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            
            # Save the image to the output folder
            image_path = os.path.join(output_folder, f"image_{page_num}_{img_index}.png")
            image.save(image_path, "PNG")
            images.append(image_path)
    
    return images

def describe_image(images_paths):
    descriptions_list = ""  
    for image_path in images_paths:
        res = ollama.chat(
            model="llava",
            messages=[
                {
                    'role': 'user',
                    'content': 'tell in detail the story shown in the picture',
                    'images': [image_path]
                }
            ]
        )
        descriptions_list += res['message']['content'] + "\n"  
    return descriptions_list
