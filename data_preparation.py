import json
import os
import numpy as np
import re
from PyPDF2 import PdfReader
from sklearn.model_selection import train_test_split


def data_prep():
    """
    Prepare training examples, parsed document and document summary
    """

    # Reading the document

    reader = PdfReader("generative_agent.pdf")

    number_of_pages = len(reader.pages)

    # Generate question-answer pairs

    pages_sentences = []
    pages_content = []
    pages_clean_text = []
    NUM_OVERLAP_SENTENCES = 10

    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        # Remove redundant text on each page to keep the text clean.
        page_text = page_text.replace('arXiv, April, 2023, J. S.  Park, J. C.  O’Brien, C. J.  Cai, M.  Morris, P.  Liang, M. S', '')
        page_text = page_text.replace('arXiv, April, 2023, J.S. Park, J.C. O’Brien, C.J. Cai, M. Morris, P. Liang, M.S. Bernstein', '')
        page_text = page_text.replace('[cs.HC]  7 Apr 2023', '')
        page_text = page_text.replace('arXiv:2304.03442v1', '')

        # Split page text into sentences
        sentences = re.split(r'[.]', page_text)
        pages_clean_text.append(page_text)
        pages_sentences.append(sentences)

    for page_num, page in enumerate(pages_clean_text):
        prev_page_overlapped_sentences = "" if page_num == 0 else '. '.join(pages_sentences[page_num - 1][-NUM_OVERLAP_SENTENCES:])
        next_page_overlapped_sentences = "" if page_num == len(pages_clean_text) - 1 else '. '.join(pages_sentences[page_num + 1][:NUM_OVERLAP_SENTENCES])
        pages_content.append(prev_page_overlapped_sentences + '\n' + page + '\n' + next_page_overlapped_sentences)
  

    # Write data to text file
    with open(os.path.join('data_test', 'generative_agent.txt'), 'w') as f:
        for page in pages_content:
            f.write(page + '\n ---- \n')

    # Generating Q&A using LLM model
    from openai import OpenAI
    client = OpenAI(api_key='')

    qas = []
    messages = []
    summaries = []

    for page_num, page in enumerate(reader.pages):
        sentences = re.split(r'[.?!]', page.extract_text())

    for page_num, page in enumerate(pages_content):
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages= [
                {"role": "system", "content": "You act as a training data generator for a GPT 3.5 fine-tuning job. Your job is to generate questions and answer those questions for the content of a research paper." 
                " The whole paper is broken into multiple sections. Therefore, all of the following content is from the same research paper and presented to you in order."
                " You only generate questions, those are and highly-relevant to the research paper, and the correct answers to the corresponding questions. "
                " The questions and answers in following format: "\
                "Question 1: \n Answer 1: \n\nQuestion 2: \n Answer 2: Question 3: \n Answer 3: " 
                "\n\nQuestion 4: \n Answer 4: \n\nQuestion 4: \n Answer 2: Question 5: \n Answer 5: "
                "\n\nQuestion 6: \n Answer 6: \n\nQuestion 7: \n Answer 7: Question 8: \n Answer 8: "
                "\n\nQuestion 9: \n Answer 9: Question 10: \n Answer 10: "},
                {"role": "user", "content": f"{page}"}
            ]
            )
        qas.append({'page': page_num, 'completion': completion.choices[0].message.content})

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages = [
                {"role": "system", "content": "You act as a training data generator for a GPT 3.5 fine-tuning job. Summarise the page content as precisely as possible. " \
                "{\"summary\": \"\"}"},
                {"role": "user", "content": f"{page}"}
            ]
            
            )
        summaries.append(completion.choices[0].message.content)

    # 3. Concat the pages summaries, generate Q&As on all summaries.
    
    page_num = 1
    str_summaries = ""
    for page_summary in summaries:
        str_summaries += f"\nPage {page_num}'s summary: {page_summary}"
        page_num += 1

    completion = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=[
            {"role": "system", "content": "You act as a training data generator for a GPT 3.5 fine-tuning job. Your job is to generate questions and answer those questions for the content of a research paper. The whole paper is broken into multiple sections. Each page's summary will be provided."
            " Provide questions and answers those are involved content from different pages. "
            " The questions and answers in following format: "\
            "\n\nQuestion 1: \n Answer 1: \n\nQuestion 2: \n Answer 2: Question 3: \n Answer 3: " 
            "\n\nQuestion 4: \n Answer 4: \n\nQuestion 5: \n Answer 5: "
            "\n\nQuestion 6: \n Answer 6: \n\nQuestion 7: \n Answer 7: \n\nQuestion 8: \n Answer 8: "
            "\n\nQuestion 9: \n Answer 9: Question 10: \n Answer 10: "
            "\n\nQuestion 11: \n Answer 11: \n\nQuestion 12: \n Answer 12: Question 13: \n Answer 13: " 
            "\n\nQuestion 14: \n Answer 14: \n\nQuestion 15: \n Answer 15: "
            "\n\nQuestion 16: \n Answer 16: \n\nQuestion 17: \n Answer 17: \n\nQuestion 18: \n Answer 18: "
            "\n\nQuestion 19: \n Answer 19: Question 20: \n Answer 20: "},
            {"role": "user", "content": "Summary: " + str_summaries}
        ]
    )

    qas.append({'page': -1, 'completion': completion.choices[0].message.content})

    with open(os.path.join('data_test', 'page_summary.txt'), 'w') as f:
        f.writelines(str_summaries)
    
    qa_examples = ""

    for qa in qas:
        page_num = qa['page']
        data_str = qa['completion'].replace('\n\nAnswer', '\nAnswer').replace(': \n', ': ')
        data_list = data_str.split('\n\n')
        for pair in data_list:
            if pair[:8] != 'Question':
                continue
            q, a = pair.split('Answer')
            q, a = q.split(': ')[-1].replace('?\n', '?'), a.split(': ')[-1]
            qa_examples += json.dumps({"messages": [{"role": "system", "content": "Assistant is a large language model trained by OpenAI.\n\n"
                                                            "Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on topics related to provided documents."
                                                            "As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.\n\n"
                                                            "Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on the topics."}, 
                                                        {"role": "user", "content": f"{q}"}, 
                                                        {"role": "assistant", "content": f"{a}"}]}) + '\n'


    with open(os.path.join('data_test', 'QA_pairs.jsonl'), 'w') as f:
        f.writelines(qa_examples[:-1])

    # Finally, split data sets


    with open(os.path.join('data_test', 'QA_pairs.jsonl'), 'r') as f:
        lines = f.readlines()

    train, test = train_test_split(np.array(lines), test_size=0.2)
    train, val = train_test_split(np.array(train), test_size=0.2)

    with open(os.path.join('data_test', 'QA_pairs_train.jsonl'), 'w') as f:
        f.writelines(train)

    with open(os.path.join('data_test', 'QA_pairs_val.jsonl'), 'w') as f:
        f.writelines(val)

    with open(os.path.join('data_test', 'QA_pairs_test.jsonl'), 'w') as f:
        f.writelines(test)


if __name__ == "__main__":
    data_prep()