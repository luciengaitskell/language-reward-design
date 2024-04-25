def openai_repr(prompt):
    '''
    Query the OpenAI API to get the response to the prompt. Prompt something like
    
     messages = [
        {"role": "system", "content": "You are a helpful, analytic assistant who can summarize text well."},
        {"role": "user", "content": prompt}
    ]
    '''
    import openai

    # Set the OpenAI API key -- REPLACE THIS WITH YOUR OWN
    openai.api_key = None # YOUR KEY HERE

    # Define the GPT model
    gpt_model = 'gpt-3.5-turbo'

    # Generate response using Chat Completion API
    response = openai.ChatCompletion.create(
        model=gpt_model,
        messages=prompt,
        temperature=0.2
    )

    return response
