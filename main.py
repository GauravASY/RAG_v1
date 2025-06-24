from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import gradio as gr
from helper import *

load_dotenv()


def model_init():
    model = ChatOllama(
        model="llama3:latest",
        temperature= 0.2
    )
    
    return model

model = model_init()

prompt = ChatPromptTemplate.from_messages([
    ('system' ,  "You are a specialized assistant for answering questions based ONLY on the provided context from PDF documents. Your instructions are to be followed exactly. 1. Review the 'Context' below. 2. If the 'Context' contains the information to answer the 'Question', provide a helpful answer based solely on that context. 3. If the 'Context' does NOT contain the information to answer the 'Question', you MUST respond with the exact phrase: 'I can not process the request'. 4. Do NOT use any of your internal knowledge. Do NOT attempt to answer if the information is not in the 'Context'. Context: {context} "),
    MessagesPlaceholder(variable_name = "history"),
    ('user' , '{query}')
])



def chat(message, history):
    messages = []
    for context in history:
        if context['role'] == 'user':
            messages.append(HumanMessage(content=context['content']))
        else :
            messages.append(AIMessage(content=context['content']))
    
    chain = prompt | model
    response = ""        
    if len(message['files']) == 0 :
        retrieved_data = retrieve_documents(message['text'])
        context = "".join(d.page_content for d in retrieved_data)
      
        
        for chunk in chain.stream({
            "context" : context,
            "history" : messages,
            "query" : message['text']
        }):
            response += chunk.content
            yield response
        
    else : 
        split_docs = load_document(message['files'])
        print("Splitting done!")
       
        if message['text'] == "" :
            yield "PDF uploaded successfully" 
        else :
            retrieved_data = retrieve_documents(message['text'])
            context = "".join(d.page_content for d in retrieved_data)
            
            for chunk in chain.stream({
                "context" : context,
                "history" : messages,
                "query" : message['text']
            }):
                response += chunk.content
                yield response
    
    
gr.ChatInterface(
    fn = chat,
    theme="ocean",
    type='messages',
    multimodal=True,
    textbox=gr.MultimodalTextbox(file_count="multiple", file_types=[".pdf"], sources=["upload"])
).launch()

