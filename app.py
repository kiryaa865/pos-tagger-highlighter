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
st.markdown(
    """
    <style>
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
    .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
    .viewerBadge_text__1JaDK {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)
client = openai.Client(api_key=os.environ.get("OPENAI_API_KEY"))
# Load secrets
#openai_api_key = st.secrets["OPENAI_API_KEY"]
assistant = client.beta.assistants.retrieve("asst_j3as6b0uolhI7cLVU8MS40cK")
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

    word_tokens = []
    word_tags = []
    current_word = ""
    current_tag = ""

    for token, tag in zip(tokens, tags):
        if token in ["[CLS]", "[SEP]"]:
            continue  # Skip special tokens
        if token.startswith("##"):
            current_word += token[2:]
        else:
            if current_word:
                word_tokens.append(current_word)
                word_tags.append(current_tag.replace("B-", "").replace("I-", ""))  # Remove B- and I- prefixes
            current_word = token
            current_tag = tag

    if current_word:
        word_tokens.append(current_word)
        word_tags.append(current_tag.replace("B-", "").replace("I-", ""))

    # Debugging output
    print("Tokens:", tokens)
    print("Tags:", tags)
    print("Word Tokens:", word_tokens)
    print("Word Tags:", word_tags)

    return [f"{token}: {tag}" for token, tag in zip(word_tokens, word_tags)]


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

st.title("Інструмент для порівняння POS-тегування")

input_text = st.text_area("Уведіть текст для аналізу:")

    # OpenAI API interaction
assistant = client.beta.assistants.retrieve("asst_j3as6b0uolhI7cLVU8MS40cK")


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
        formatted_output = tag_roberta_text(input_text)
        for item in formatted_output:
           print(item)
    
    return captured_output.getvalue()


def parse_output(output):
    results = {"Token": []}
    models = ["Fine-tuned GPT-4", "Stanza", "SpaCy", "Pymorphy3", "Flair", "Fine-tuned RoBERTa"]
    model_map = {
        "GPT 4o": "Fine-tuned GPT-4",
        "RoBERTa": "Fine-tuned RoBERTa"
    }

    for model in models:
        results[model] = []

    current_model = None
    token_dict = {model: [] for model in models}
    token_positions = []

    for line in output.splitlines():
        if not line.strip():
            continue
        if any(model in line for model in model_map.keys()):
            current_model = model_map[line.strip().replace(":", "")]
        elif any(model in line for model in models):
            current_model = line.strip().replace(":", "")
        elif current_model:
            match = re.match(r"(.+): (.+)", line)
            if match:
                token, pos = match.groups()
                token_dict[current_model].append((token, pos))
                if current_model == "Fine-tuned GPT-4":
                    token_positions.append(token.lower())

    common_tokens = set(token.lower() for token, _ in token_dict[models[0]])
    for model in models[1:]:
        model_tokens = set(token.lower() for token, _ in token_dict[model])
        common_tokens &= model_tokens

    sorted_common_tokens = [token for token in token_positions if token in common_tokens]

    gpt_capitalized_tokens = {}
    for token, pos in token_dict["Fine-tuned GPT-4"]:
        gpt_capitalized_tokens[token.lower()] = token

    for token in sorted_common_tokens:
        capitalized_token = gpt_capitalized_tokens.get(token, token)
        results["Token"].append(capitalized_token)
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
    tags = [tag.strip() if tag is not None else None for tag in tags]  # Strip whitespace from tags
    tag_counts = pd.Series(tags).value_counts()

    # Debug print statements
    print(f"Token: {row['Token']}")
    print(f"Tags: {tags}")
    print(f"Tag Counts: {tag_counts}")

    # Check for PUNCT tag
    if 'PUNCT' in tags:
        print("PUNCT found, no highlighting.")
        return [''] * len(row)

    # All models agree
    if len(tag_counts) == 1:
        print("All models agree, no highlighting.")
        return [''] * len(row)

    # 5 out of 6 agree
    if tag_counts.iloc[0] == 5:
        most_common_tag = tag_counts.index[0]
        result = [''] + ['background-color: yellow' if x != most_common_tag else '' for x in tags]
        print(f"5 out of 6 agree, highlighting: {result}")
        return result

    # 4 out of 6 agree
    if tag_counts.iloc[0] == 4:
        most_common_tag = tag_counts.index[0]
        result = [''] + ['background-color: yellow' if x != most_common_tag else '' for x in tags]
        print(f"4 out of 6 agree, highlighting: {result}")
        return result

    # 3 or fewer agree
    result = [''] + ['background-color: yellow'] * len(tags)
    print(f"3 or fewer agree, highlighting the whole row: {result}")
    return result

if 'show_results' not in st.session_state:
    st.session_state.show_results = False
if 'show_guide_button' not in st.session_state:
    st.session_state.show_guide_button = False
if 'show_guide' not in st.session_state:
    st.session_state.show_guide = False

def show_tags_guide():
    tags_explanations = {
        "NOUN": "іменник",
        "PUNCT": "пунктуація",
        "VERB": "дієслово",
        "ADJ/ADJF": "прикметник",
        "ADP": "прийменник",
        "ADV": "прислівник",
        "PRON": "займенник",
        "CCONJ": "сполучник",
        "DET": "детермінатив",
        "PART": "частка",
        "PROPN": "власна назва",
        "SCONJ": "підрядний сполучник",
        "NUM": "числівник",
        "AUX": "допоміжне дієслово",
        "X": "інше",
        "INTJ": "вигук",
        "SYM": "символ",
        "ADJS": "прикметник коротка форма",
        "COMP": "порівняльна ступінь",
        "INFN": "інфінітив",
        "PRTF": "дієприкметник",
        "PRTS": "коротка форма дієприкметника",
        "GRND": "дієприслівник",
        "NUMR": "числівник",
        "ADVB": "прислівник",
        "NPRO": "займенник",
        "PRED": "предикатив",
        "PREP": "прийменник",
        "CONJ": "сполучник",
        "PRCL": "частка"
    }
    data = [{"Тег": tag, "Пояснення": explanation} for tag, explanation in tags_explanations.items()]
    st.table(data)
    
if st.button("Почати"):
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
    st.session_state.highlighted_df = highlighted_df
    st.session_state.show_guide = False
    st.session_state.show_results = True

# Show the dataframe if the "Почати" button has been clicked
if st.session_state.show_results:
    st.dataframe(st.session_state.highlighted_df)
    if st.button("Показати довідник з тегами"):
        st.session_state.show_guide = True

# Show the guide if the "Показати довідник із тегами" button has been clicked
if st.session_state.show_guide:
    show_tags_guide()



