def format_docs(docs):
    return "\n---\n".join('Title: ' + doc.metadata['title'] + '\n' + doc.page_content for doc in docs)

def validate_input(data):
    if not isinstance(data, dict):
        raise ValueError("Input must be a dictionary.")
    return True

def extract_query_params(query):
    return {key: value for key, value in query.items() if value is not None}