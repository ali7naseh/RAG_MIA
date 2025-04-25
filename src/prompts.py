MULTIPLE_PROMPT = 'You are a helpful assistant, below is a query from a user and some relevant contexts. \
Answer the question given the information in those contexts. Your answer should be short and concise. \
If you cannot find the answer to the question, just say "I don\'t know". \
\n\nContexts: [context] \n\nQuery: [question] \n\nAnswer:'

GROUND_TRUTH_PROMPT = 'You are a helpful assistant. Based solely on the provided contexts, generate a concise and accurate response to the question from the user. \
\n\nContexts: [context] \n\nQuestion: [question]? Answer only with "Yes," "No," or "I don\'t know". \n\nAnswer:'


MULTIPLE_PROMPT_NO_DOC = 'You are a helpful assistant, below is a query from a user. \
Your answer should be short and concise. \
If you cannot give the answer to the question, just say "I don\'t know". \
\n\nQuery: [question] \n\nAnswer:'


def wrap_prompt(question, context,
                prompt_id: int=1,
                atk: bool = False,
                context_free_response: bool = False) -> str:

    if atk==True:
        PROMPT = GROUND_TRUTH_PROMPT
    else:
        if context_free_response:
            # Ignore context
            input_prompt = MULTIPLE_PROMPT_NO_DOC.replace('[question]', question)
            return input_prompt
        else:
            PROMPT = MULTIPLE_PROMPT

    if prompt_id == 4:
        assert type(context) == list
        context_str = "\n".join(context)
        input_prompt = PROMPT.replace('[question]', question).replace('[context]', context_str)
    else:
        input_prompt = PROMPT.replace('[question]', question).replace('[context]', context)
    return input_prompt

