# Importing all features for work

import streamlit as st
from PIL import Image
import nltk
import requests
import json
import nltk.data
from nltk.stem import  WordNetLemmatizer, PorterStemmer
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import glob
import torch
import pickle
import zipfile
from IPython.display import display
from IPython.display import Image as IPImage
import os
from tqdm.autonotebook import tqdm
import base64
import io
torch.set_num_threads(4)

stem = PorterStemmer()
Lemm = WordNetLemmatizer()
nltk.download('wordnet')

im = Image.open('logo.png')
# Adding Image to web app
st.set_page_config(page_title="AIEV", page_icon = im)

def synonym_extractor(phrase):
     from nltk.corpus import wordnet
     synonyms = []

     for syn in wordnet.synsets(phrase):
          for l in syn.lemmas():
            if((l.name().lower()) != phrase.lower()):
               synonyms.append(l.name().lower())
     return synonyms

# Function for getting antonyms from nltk

def antonym_extractor(phrase):
     from nltk.corpus import wordnet
     antonyms = []

     for syn in wordnet.synsets(phrase):
          for I in syn.lemmas():
            if I.antonyms():
              if((I.antonyms()[0].name().lower()) != phrase.lower()):
                 antonyms.append(I.antonyms()[0].name().lower())

    # print(set(antonyms))
     return antonyms

# Function for getting synonyms from API

def synonym_request(word):
    # API endpoint for the dictionary
    url = f'https://api.dictionaryapi.dev/api/v2/entries/en/{word}'

    try:
        # Send GET request to the API
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()

            # Retrieve the meanings
            meanings = data[0]['meanings']

            # Check each meaning for synonyms
            for meaning in meanings:
                if 'synonyms' in meaning:
                    synonyms = meaning['synonyms']

                    # Check if the synonym list is not empty
                    if synonyms:
                        return synonyms
                        #print(f'Synonyms of {word}:')

                        break  # Exit the loop after printing the first non-empty synonym list

            # If no non-empty synonym list was found
            else:
                return "None"
                print(f'No synonyms found for {word} .')
        else:
            return 'None'
            print("Error: Request failed with status code", response.status_code)

    except requests.exceptions.RequestException as e:
        return "None"
        print("Error: ", e)


# Function for getting antonyms from API

def antonym_request(word):
    # API endpoint for the dictionary
    url = f'https://api.dictionaryapi.dev/api/v2/entries/en/{word}'

    try:
        # Send GET request to the API
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()

            # Retrieve the meanings
            meanings = data[0]['meanings']

            # Check each meaning for antonyms
            for meaning in meanings:
                if 'antonyms' in meaning:
                    antonyms = meaning['antonyms']

                    # Check if the antonym list is not empty
                    if antonyms:
                        for antonym in antonyms:
                            print(antonym, "these are antonyms")
                        return antonyms
                        # print(f'Antonyms of : {word}')
                        
                        break  # Exit the loop after printing the first non-empty antonym list

            # If no non-empty antonym list was found
            else:
                return "None"
                print(f'No antonyms found for  {word}.')
        else:
            return "None"
            print("Error: Request failed with status code", response.status_code)

    except requests.exceptions.RequestException as e:
        return "None"
        print("Error: ", e)

# Function for getting definition from API

def definition_request(word):

    url = f'https://api.dictionaryapi.dev/api/v2/entries/en/{word}'
    try:
    # Send GET request to the API
        response = requests.get(url)

    # Check if the request was successful
        if response.status_code == 200:
            data = response.json()

        # Retrieve the first definition from the response
            definition = data[0]['meanings'][0]['definitions'][0]['definition']

        # Print the definition
            print(f'Definition of {word}:')
            print(definition)
            return definition
        else:
            print("Error: Request failed with status code", response.status_code)
            return "None"

    except requests.exceptions.RequestException as e:
        print("Error: ", e)
        return "None"

# Function for getting example sentences from API

def example_request(word):
    url = f'https://api.dictionaryapi.dev/api/v2/entries/en/{word}'
    try:
        # Send GET request to the API
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()

            # Retrieve the example sentences from the response
            examples = []
            for meaning in data[0]['meanings']:
                for definition in meaning['definitions']:
                    if 'example' in definition:
                        examples.append(definition['example'])

            # Print the example sentences
            array = []

            if examples:
                max = 0
                count = 0
                for i in examples:
                  if len(i) > max:
                    max = len(i)
                    result = i

                return result
            else:
               #print(f'No example sentences found for {word}.')
               return "None"
    except requests.exceptions.RequestException as e:
        return "None"

# Function for getting final_synonym_list by using synonyms either from API and nltk

def final_synonym(word):
    synonyms_from_api = synonym_request(word)
    synonyms_from_nltk = synonym_extractor(word)
    final_synonym_list = []
    for i in synonyms_from_api:
        i = i.replace('-', ' ')
        i = i.replace('_', ' ')
        final_synonym_list.append(i)
    for i in synonyms_from_nltk:
        i = i.replace('-', ' ')
        i = i.replace('_', ' ')
        final_synonym_list.append(i)
    print(set(final_synonym_list))
    return set(final_synonym_list)

# Function for getting final_antonym_list by using synonyms either from API and nltk

def final_antonym(word):
    antonyms_from_api = antonym_request(word)
    antonyms_from_nltk = antonym_extractor(word)
    final_antonyms_list = []
    for i in antonyms_from_api:
        i = i.replace('-', ' ')
        i = i.replace('_', ' ')
        final_antonyms_list.append(i)
    for i in antonyms_from_nltk:
        i = i.replace('-', ' ')
        i = i.replace('_', ' ')
        final_antonyms_list.append(i)
    print(set(final_antonyms_list))
    return set(final_antonyms_list)

# Function for getting example sentences of the given word's antonyms


def example_for_antonym(list):
    res = []
    for i in list:
      text = example_request(i)
      if type(text) !=  type(None):
          text1 = text.replace("\xa0", "")
          text2 = text1.replace('\u2003', ' ')
          #print(type(text1))
          res.append(text2)
    return res


# Function for getting example sentences of the given word's synonym


def example_for_synonym(list):
    res = []
    for i in list:
      text = example_request(i.replace('-', ' '))
      if type(text) !=  type(None):
          text1 = text.replace("\xa0", "")
          text2 = text1.replace('\u2003', ' ')
          #print(type(text1))
          res.append(text2)
    return res


# Functions for searching images based on example sentences

#First, we load the respective CLIP model
model = SentenceTransformer('clip-ViT-B-32')

# Next, we get about 25k images from Unsplash
img_folder = 'photos/'
if not os.path.exists(img_folder) or len(os.listdir(img_folder)) == 0:
    os.makedirs(img_folder, exist_ok=True)

    photo_filename = 'unsplash-25k-photos.zip'
    if not os.path.exists(photo_filename):   #Download dataset if does not exist
        util.http_get('http://sbert.net/datasets/'+photo_filename, photo_filename)

    #Extract all images
    with zipfile.ZipFile(photo_filename, 'r') as zf:
        for member in tqdm(zf.infolist(), desc='Extracting'):
            zf.extract(member, img_folder)

use_precomputed_embeddings = True

if use_precomputed_embeddings:
    emb_filename = 'unsplash-25k-photos-embeddings.pkl'
    if not os.path.exists(emb_filename):   #Download dataset if does not exist
        util.http_get('http://sbert.net/datasets/'+emb_filename, emb_filename)

    with open(emb_filename, 'rb') as fIn:
        img_names, img_emb = pickle.load(fIn)
    print("Images:", len(img_names))
else:
    img_names = list(glob.glob('unsplash/photos/*.jpg'))
    print("Images:", len(img_names))
    img_emb = model.encode([Image.open(filepath) for filepath in img_names], batch_size=128, convert_to_tensor=True, show_progress_bar=True)

# Next, we define a search function.
def search(query, k=1):
    # First, we encode the query (which can either be an image or a text string)
    query_emb = model.encode([query], convert_to_tensor=True, show_progress_bar=False)

    # Then, we use the util.semantic_search function, which computes the cosine-similarity
    # between the query embedding and all image embeddings.
    # It then returns the top_k highest ranked images, which we output
    hits = util.semantic_search(query_emb, img_emb, top_k=k)[0]

    # st.text(query)
    for hit in hits:
        image_path = os.path.join(img_folder, img_names[hit['corpus_id']])
        image = Image.open(image_path)
        st.image(image, width=200)


main_style = """
    <style>
        body {
            background-color: #F5F5F5;
        }
        .text_style {
            font-size: 25px;
            color: green;
        }
        .header {
            font-size:20px;
        }
        .def{
            font-size:17px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .synonym {
            font-size:17px;
            font-weight: bold;
            margin-bottom: 10px;
            color: blue;
        }
        .antonym {
            font-size:17px;
            font-weight: bold;
            margin-bottom: 10px;
            color: red;
        }
        .example {
            margin-left: 20px;
            margin-bottom: 5px;
        }
        .example-container {
            border: 1px solid #ccc;
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 10px;
        }
        .example-image {
            width: 100%;
            margin-bottom: 10px;
        }
    </style>
"""


hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
padding = 0
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)




st.write(main_style, unsafe_allow_html=True)

# Define a folder to save the images
image_folder = 'image_folder'
os.makedirs(image_folder, exist_ok=True)

# Define a list to store the paths of saved images
saved_image_paths = []

col1, col2 = st.columns([6,2])
with col1:
    st.title("AI English Vocabulary")
    st.markdown("<p class='text_style'>Write a word from the English Vocabulary</p>", unsafe_allow_html=True)
with col2:
    logo_image = 'logo2.png'  # Replace with the path to your logo image
    # st.image(logo_image, output_format='PNG',use_column_width = True, style = 'margin-right: 20px'     )
    with open(logo_image, 'rb') as f:
        logo_bytes = f.read()

# Convert the logo image to base64
    logo_base64 = base64.b64encode(logo_bytes).decode('utf-8')

    # Apply margin using CSS
    st.markdown(
        f"""
        <style>
            .logo-container {{
                margin-bottom: 20px;
                margin-right: 20px;
            }}
        </style>
        <div class="logo-container">
            <img src="data:image/png;base64,{logo_base64}" alt="Logo" style="width: 100%;">
        </div>
        """,
        unsafe_allow_html=True
    )



text_input = st.text_input("", placeholder='Enter your word')
if text_input:
    st.write(f"<p class = 'header'> Your word is {text_input.lower()}  </p>", unsafe_allow_html =True )
    st.write(f"<p class = 'def'> The definition of {text_input.lower()} - {definition_request(text_input.lower())}  </p>", unsafe_allow_html = True)
    final_synonym_list = final_synonym(text_input.lower())
    final_antonym_list = final_antonym(text_input.lower())
    final_synonym_list = list(final_synonym_list)
    final_antonym_list = list(final_antonym_list)
    count = 0
    helper = 0
    res = []
    st.markdown(f"<p class='synonym'>Synonyms of {text_input.lower()} with their examples.</p>", unsafe_allow_html=True)
    while helper < 5 and count < len(final_synonym_list):
        sentences_list = example_request(final_synonym_list[count])
        
        if type(sentences_list) != type(None) and len(final_synonym_list[count]) >= 3:  
            if sentences_list not in res:
                with st.container():
                    # Apply CSS styling to the container
                    st.markdown("""
                        <style>
                            .example-container {
                                border: 1px solid #ccc;
                                border-radius: 5px;
                                padding: 10px;
                                margin-bottom: 10px;
                            }
                            .example-text {
                                font-size: 17px;
                                font-weight: bold;
                                margin-bottom: 5px;
                            }
                        </style>
                    """, unsafe_allow_html=True)
                    
                    # Add the Markdown content inside the container
                    st.markdown(f"<div class='example-container'><div class='example-text'>{final_synonym_list[count]}: {sentences_list}.</div>", unsafe_allow_html=True)
                    
                    # Perform the search
                    with st.expander("Image Results"):
                        if sentences_list == "None":
                            logo_image = 'test11.png'  # Replace with the path to your logo image
                            # st.image(logo_image, output_format='PNG',use_column_width = True, style = 'margin-right: 20px'     )
                            with open(logo_image, 'rb') as f:
                                logo_bytes = f.read()

                            # Convert the logo image to base64
                                logo_base64 = base64.b64encode(logo_bytes).decode('utf-8')
                        else:
                            search(sentences_list)
                    st.markdown('</div>', unsafe_allow_html = True)
                    # Update the helper and count variables
                    res.append(sentences_list)
                    helper += 1
            
        count += 1

    count = 0
    helper = 0
    res = []
    st.markdown(f"<p class='antonym'>Antonyms of {text_input.lower()} with their examples.</p>", unsafe_allow_html=True)
    while helper < 5 and count < len(final_antonym_list):
        sentences_list = example_request(final_antonym_list[count])
        
        if type(sentences_list) != type(None) and len(final_antonym_list[count]) >= 3:  
            # sentences_list = sentences_list.split('.')
            if sentences_list not in res:
                with st.container():
                    # Apply CSS styling to the container
                    st.markdown("""
                        <style>
                            .example-container {
                                border: 1px solid #ccc;
                                border-radius: 5px;
                                padding: 10px;
                                margin-bottom: 10px;
                            }
                            .example-text {
                                font-size: 17px;
                                font-weight: bold;
                                margin-bottom: 5px;
                            }
                        </style>
                    """, unsafe_allow_html=True)
                    
                    # Add the Markdown content inside the container
                    st.markdown(f"<div class='example-container'><div class='example-text'>{final_antonym_list[count]}: {sentences_list}.</div>", unsafe_allow_html=True)
                    
                    # Perform the search
                    with st.expander("Image Results"):
                        if sentences_list == "None":
                            logo_image = 'test11.png'  # Replace with the path to your logo image
                            # st.image(logo_image, output_format='PNG',use_column_width = True, style = 'margin-right: 20px'     )
                            with open(logo_image, 'rb') as f:
                                logo_bytes = f.read()

# Convert the logo image to base64
                            logo_base64 = base64.b64encode(logo_bytes).decode('utf-8')
                        else:
                            search(sentences_list)
                    st.markdown('</div>', unsafe_allow_html = True)
                    # Update the helper and count variables
                    res.append(sentences_list)
                    helper += 1
            
        count += 1









