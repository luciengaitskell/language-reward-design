 messages = [
        {"role": "system", "content": "You are a helpful, analytic assistant who can summarize text well."},
        {"role": "user", "content": prompt}
    ]

    # Return the messages list
    #return messages

def openai_repr(model, docs, topic, doc_to_topic):
    '''
    Gets the ChatGPT representation of a topic.

    Args:
        model (object): The bertopic model.
        docs (object): The training corpus.
        topic (int): The topic number.
        doc_to_topic (object): The DataFrame relating a document to a topic.

    Returns:
        object: The openai response.
    '''
    import openai

    # Set the OpenAI API key -- REPLACE THIS WITH YOUR OWN
    openai.api_key = None # YOUR KEY HERE

    # Define the GPT model
    gpt_model = 'gpt-3.5-turbo'

    # Get document information
    doc_info = model.get_document_info(docs)

    # Filter documents by topic
    topic_df = doc_info.loc[doc_info["Topic"] == topic]

    # Get topic representation and number
    topic_words = topic_df["Representation"].tolist()[0]
    topic_num = topic_df["Topic"].tolist()[0]

    # Get documents related to the topic
    topic_docs = doc_to_topic.loc[doc_to_topic["Topic"] == topic_num]["Document"].tolist()

    # Create prompt for GPT model
    prompt = gpt_prompt(topic_docs, topic_words)

    # Generate response using Chat Completion API
    response = openai.ChatCompletion.create(
        model=gpt_model,
        messages=prompt,
        temperature=0.2
    )

    return response
