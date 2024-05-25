import streamlit as st
import pandas as pd
import re
import io
import stanza
import spacy
import pymorphy3
from flair.models import SequenceTagger
from flair.data import Sentence
from transformers import AutoModelForTokenClassification, AutoTokenizer
from contextlib import redirect_stdout
import os
import time
import openai
from openai import OpenAI
import subprocess
import string 
import torch

client = openai.Client(api_key=os.environ.get("OPENAI_API_KEY"))
# Load secrets
#openai_api_key = st.secrets["OPENAI_API_KEY"]
assistant = client.beta.assistants.retrieve("asst_7Lbs35tXNgg5HwAkjQKU5xZs")
#os.environ["OPENAI_API_KEY"] = openai_api_key
#client = OpenAI()

# Function to install the SpaCy model if not present
@st.cache_resource
def install_spacy_model():
    try:
        nlp_spacy = spacy.load("uk_core_news_sm")
    except OSError:
        subprocess.run(["python", "-m", "spacy", "download", "uk_core_news_sm"], check=True)
        nlp_spacy = spacy.load("uk_core_news_sm")
    return nlp_spacy

# Initialize models
@st.cache_resource
def load_models():
    stanza.download('uk')
    nlp_stanza = stanza.Pipeline('uk')
    
    nlp_spacy = install_spacy_model()

    tagger_flair = SequenceTagger.load("dchaplinsky/flair-uk-pos")

    morph = pymorphy3.MorphAnalyzer(lang='uk')

    model_roberta = AutoModelForTokenClassification.from_pretrained("KoichiYasuoka/roberta-base-ukrainian-upos")
    tokenizer_roberta = AutoTokenizer.from_pretrained("KoichiYasuoka/roberta-base-ukrainian-upos")
    
    return nlp_stanza, nlp_spacy, tagger_flair, morph, model_roberta, tokenizer_roberta

nlp_stanza, nlp_spacy, tagger_flair, morph, model_roberta, tokenizer_roberta = load_models()

def stanza_pos(text):
    doc = nlp_stanza(text)
    return [(word.text, word.pos) for sentence in doc.sentences for word in sentence.words]

def spacy_pos(text):
    doc = nlp_spacy(text)
    return [(token.text, token.pos_) for token in doc]

def tag_ukrainian_text(text):
    tagged_words = [(word, morph.parse(word)[0].tag.POS if word not in string.punctuation else None)
                    for word in re.findall(r"[\w']+|[.,!?;]", text)]
    return tagged_words

#def tag_flair_text(input_text):
 #   sentence = Sentence(input_text)
 #   tagger_flair.predict(sentence)
 #   return [(token.text, token.get_tag("pos").value) for token in sentence]

def tag_flair_text(input_text):
  sentence = Sentence(input_text)
  tagger_flair.predict(sentence)
  output = ""
  for token in sentence:
    output += f"{token.text}: {token.tag}\n"
  return output.strip()

def tag_roberta_text(input_text):
    inputs = tokenizer_roberta(input_text, return_tensors="pt")
    outputs = model_roberta(**inputs).logits
    predictions = torch.argmax(outputs, dim=2)
    id2label = model_roberta.config.id2label
    tokens = tokenizer_roberta.convert_ids_to_tokens(inputs["input_ids"][0])
    tags = [id2label[p.item()] for p in predictions[0]]
    return [(token.replace("##", ""), tag.replace("B-", "").replace("I-", ""))
            for token, tag in zip(tokens, tags) if token not in ["[CLS]", "[SEP]"]]

#assistant = client.beta.assistants.retrieve("asst_7Lbs35tXNgg5HwAkjQKU5xZs")
#user_message = str(input_text)
#thread = client.beta.threads.create()
#message = client.beta.threads.messages.create(thread_id=thread.id, role="user", content=user_message)

#run = client.beta.threads.runs.create(thread_id = thread.id,assistant_id = assistant.id)

#run_status = client.beta.threads.runs.retrieve(thread_id = thread.id,run_id = run.id)

def loop_until_completed(clnt: object, thrd: object, run_obj: object) -> None:
    """
    Poll the Assistant runtime until the run is completed or failed
    """
    while run_obj.status not in ["completed", "failed", "requires_action"]:
        run_obj = clnt.beta.threads.runs.retrieve(
            thread_id = thrd.id,
            run_id = run_obj.id)
        time.sleep(10)
        print(run_obj.status)


#loop_until_completed(client, thread, run_status)

def print_thread_messages(clnt: object, thrd: object, content_value: bool=True) -> None:
    """
    Prints OpenAI thread messages to the console.
    """
    messages = clnt.beta.threads.messages.list(
        thread_id = thrd.id)
    print(messages.data[0].content[0].text.value)

st.title("POS Tag Discrepancy Highlighter")

input_text = st.text_area("Enter the text to analyze:")

    # OpenAI API interaction
assistant = client.beta.assistants.retrieve("asst_7Lbs35tXNgg5HwAkjQKU5xZs")


def capture_printed_output():
    captured_output = io.StringIO()
    with redirect_stdout(captured_output):
        # Print results
        print("\nGPT 4o:")
        print(print_thread_messages(client, thread))

        print("\nStanza:")
        stanza_pos_tags = stanza_pos(input_text)
        for token, pos in stanza_pos_tags:
            print(f"{token}: {pos}")

        print("\nSpaCy:")
        spacy_pos_tags = spacy_pos(input_text)
        for token, pos in spacy_pos_tags:
            print(f"{token}: {pos}")

        print("\nPymorphy3:")
        ukrainian_text = input_text
        tagged_text = tag_ukrainian_text(ukrainian_text)
        for word, pos in tagged_text:
            print(f"{word}: {pos}")

        print("\nFlair:")
        tagged_text = tag_flair_text(input_text)
        print(tagged_text)

        print("\nRoBERTa:")
        formatted_output = [f"{token}: {tag}" for token, tag in tag_roberta_text(input_text)]
       # print(tag_roberta_text(input_text))
        for item in formatted_output:
            print(item)
    
    return captured_output.getvalue()

def parse_output(output):
    results = {"Token": []}
    models = ["GPT 4o", "Stanza", "SpaCy", "Pymorphy3", "Flair", "RoBERTa"]
    for model in models:
        results[model] = []

    current_model = None
    token_dict = {model: [] for model in models}
    token_positions = []

    for line in output.splitlines():
        if not line.strip():
            continue
        if any(model in line for model in models):
            current_model = line.strip().replace(":", "")
        elif current_model:
            match = re.match(r"(.+): (.+)", line)
            if match:
                token, pos = match.groups()
                token_dict[current_model].append((token, pos))
                if current_model == "GPT 4o":
                    token_positions.append(token.lower())

    common_tokens = set(token.lower() for token, _ in token_dict[models[0]])
    for model in models[1:]:
        model_tokens = set(token.lower() for token, _ in token_dict[model])
        common_tokens &= model_tokens

    sorted_common_tokens = [token for token in token_positions if token in common_tokens]

    for token in sorted_common_tokens:
        results["Token"].append(token)
        for model in models:
            found = False
            for token_model, pos in token_dict[model]:
                if token == token_model.lower():
                    results[model].append(pos)
                    found = True
                    break
            if not found:
                results[model].append(None)

    return pd.DataFrame(results)

def highlight_discrepancies(row):
    tags = row[1:].values
    tag_counts = pd.Series(tags).value_counts()

    # Check for PUNCT tag
    if 'PUNCT' in tags:
        return [''] * len(row)

    # All models agree
    if len(tag_counts) == 1:
        return [''] * len(row)

    # 5 out of 6 agree
    if tag_counts.iloc[0] == 5:
        return [''] + ['background-color: yellow' if x != tag_counts.index[0] else '' for x in tags]

    # 4 out of 6 agree
    if tag_counts.iloc[0] == 4:
        return [''] + ['background-color: yellow' if x != tag_counts.index[0] else '' for x in tags]

    # 3 or fewer agree
    return [''] + ['background-color: yellow'] * len(tags)


if st.button("Analyze"):
    # Perform POS tagging with all models
    #stanza_pos_tags = stanza_pos(input_text)
    #spacy_pos_tags = spacy_pos(input_text)
    #pymorphy_pos_tags = tag_ukrainian_text(input_text)
    #flair_pos_tags = tag_flair_text(input_text)
    #roberta_pos_tags = tag_roberta_text(input_text)
    user_message = str(input_text)
    thread = client.beta.threads.create()
    message = client.beta.threads.messages.create(thread_id=thread.id, role="user", content=user_message)
    run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant.id)
    run_status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
    loop_until_completed(client, thread, run_status)
    # OpenAI API interaction
    #user_message = str(input_text)
    #thread = client.beta.threads.create()
    #message = client.beta.threads.messages.create(thread_id=thread.id, role="user", content=user_message)
    #run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant.id)
    #run_status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run_obj.id)
    #loop_until_completed(client, thread, run_status)
    
    # Capture and process output
    captured_output = capture_printed_output()
    os.write(1, f"{captured_output}\n".encode())
    df = parse_output(captured_output)
    highlighted_df = df.style.apply(highlight_discrepancies, axis=1)
    st.dataframe(highlighted_df)
