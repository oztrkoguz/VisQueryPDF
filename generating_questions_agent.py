from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def generate_questions(text, llm):
    question_prompt = PromptTemplate.from_template(
        "Generate exactly five questions based on the following text:\n\n{text}\n\nList the questions without numbering, each question on a new line."
    )
    
    question_chain = LLMChain(llm=llm, prompt=question_prompt)
    
    questions = question_chain.run({"text": text})
    
    questions_list = [line.strip() for line in questions.strip().split('\n') if line.strip() and not line.startswith('Here are') and not line.startswith('Let me know')]

    return questions_list[:5]

