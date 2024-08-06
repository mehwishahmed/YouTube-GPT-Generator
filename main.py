import os 
from apikey import apikey 

import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 

os.environ['OPENAI_API_KEY'] = apikey

# App framework
st.title('🖥️Python Project idea generator💡')
prompt = st.text_input('Insert your topics of interest here:') 

# Prompt templates
title_template = PromptTemplate(
    input_variables = ['topic'], 
    template='Write me a list of Python projects going from a beginner level to advanced based on {topic} and section each set of 1-2 projects by labels such as beginner, intermediate, andadvanced above the mini list of the projects. and limit to max 8 projects in total please. '
)

script_template = PromptTemplate(
    input_variables = ['title', 'wikipedia_research'], 
    template='Write me a list of Python projects going from a beginner level to advanced based on this topic TOPIC: {title} '
)

# Memory 
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')


# Llms
llm = OpenAI(temperature=0.9) 
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

wiki = WikipediaAPIWrapper()

# Show stuff to the screen if there's a prompt
if prompt: 
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt) 
    script = script_chain.run(title=title, wikipedia_research=wiki_research)

    st.write(title) 
    st.write(script) 