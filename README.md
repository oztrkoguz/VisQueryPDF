# VisQueryPDF
### The aim of the project is to verify the alignment between images and texts in documents.
### 2 different methods were discussed.
## 1.Method 

![yontem1](https://github.com/oztrkoguz/VisQueryPDF/assets/101019436/65b62ab9-c98a-44db-bee1-abd71e6d0714)

Images automatically extracted from the document were described using a VLM agent structure. Using the description results, questions were generated with a question generation agent. Subsequently, these questions were posed to the document using the RAG system, and answers were verified.

## 2.Method

![Adsz-2024-06-29-0711](https://github.com/oztrkoguz/VisQueryPDF/assets/101019436/2a0dd56e-8839-446c-b42b-3758c577cf86)

Images and texts are automatically extracted from the document. Text data undergoes processing using a summarization agent to obtain a concise summary. Subsequently, embeddings of images and texts are extracted using the CLIP model, and their similarities are compared.
