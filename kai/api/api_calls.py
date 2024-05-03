import os
from openai import OpenAI
import openai

def openai_repr(system_instructions, prompt):
    '''
    Query the OpenAI API to get the response to the prompt. Prompt something like
    '''

    # Set the OpenAI API key -- REPLACE THIS WITH YOUR OWN
    

    # Define the GPT model
    gpt_model = 'gpt-3.5-turbo'
    
    messages = [
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": prompt}
    ]

    client = OpenAI() # pass api_key
   
    # Generate response using Chat Completion API
    response = client.chat.completions.create(
        model=gpt_model,
        messages=messages,
        temperature=0.2
    )

    return response
