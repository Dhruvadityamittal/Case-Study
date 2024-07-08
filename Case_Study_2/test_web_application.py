# Streamlit web application for Question Classification

# Import necessary libraries and modules
import streamlit as st
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"  # Check if CUDA is available and set the device accordingly
model_path = 'Saved_Models/'  # Path to the saved models
from models import QuestionClassifier  # Import the QuestionClassifier model
from transformers import AutoTokenizer, AutoModel  # Import tokenizer and model from transformers library
import re  # Import regular expressions module

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')  # Download stopwords for nltk

# Set of English stopwords
stop = set(stopwords.words('english'))

if __name__ == "__main__":
    # HTML template for the app's header
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Question Quality Classifier</h2>
    </div>
    <br>
    """
    st.markdown(html_temp, unsafe_allow_html=True)  # Display the header

    # Dropdown for model selection
    model_name = st.selectbox(
        'Select the Base Model',
        ["BERT-BASE", "BERT_LARGE"]
    )
    
    # Dropdown to select whether to remove stop words
    REMOVE_STOP_WORDS = st.selectbox(
        'Select if want to remove Stop Words',
        ["True", "False"]
    )
    
    # Input fields for title, body, and tag
    title = st.text_input("Title", "")
    body = st.text_input("Body", "")
    tag = st.text_input("Tag", "")
    
    # Combine and preprocess the input text
    text = title + " " + body + " " + body
    text = text.lower()
    text = re.sub(r"<([^>]+)>", r"\1 ", text)  # Remove HTML tags
    text = ''.join(text.splitlines())  # Remove newlines
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters

    # Remove stop words if selected
    if REMOVE_STOP_WORDS == "True":
        text = [w.strip() for w in text.split(' ') if not w in stop and len(w.strip()) > 3]
        text = ' '.join(text)
    
    # Predict button
    if st.button("Predict", type="primary"):
        
        # Tokenize and encode the input text based on the selected model
        if model_name == 'BERT-BASE':
            tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
            tokenized_review = tokenizer.encode_plus(
                text,
                max_length=300,
                add_special_tokens=True,  # Add [CLS] and [SEP] tokens
                pad_to_max_length=True,  # Pad to the maximum length
                return_attention_mask=True,  # Return the attention mask
                return_token_type_ids=False,  # Do not return token type IDs
                return_tensors='pt'  # Return PyTorch tensors
            )
            token = {
                'review': text,
                'input_ids': tokenized_review['input_ids'].flatten(),  # Flatten the input IDs tensor
                'attention_mask': tokenized_review['attention_mask'].flatten(),  # Flatten the attention mask tensor
            }
            pretrained_model = AutoModel.from_pretrained("google-bert/bert-base-uncased")
            
            # Initialize and load the model
            model = QuestionClassifier(3, 1, 300, pretrained_model).to(device)
            checkpoint = torch.load(f'{model_path}/best_model_STW_{REMOVE_STOP_WORDS}_MAXLEN_300_TK_bert_Fold_{3}.pth', map_location=torch.device(device))
            del checkpoint['model_state_dict']['pretrained_model.embeddings.position_ids']
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Make prediction
            prediction = model(input_ids=token['input_ids'].view(-1, 300).to(device), attention_mask=token['attention_mask'].view(-1, 300).to(device))
            _, preds = torch.max(prediction, dim=1)
            labels_inverse_mapping = {0: 'HQ', 1: 'LQ_CLOSE', 2: 'LQ_EDIT'}
            st.write("Question Quality is: ", labels_inverse_mapping[preds.item()])
            
        if model_name == 'BERT_LARGE':
            tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
            tokenized_review = tokenizer.encode_plus(
                text,
                max_length=300,
                add_special_tokens=True,  # Add [CLS] and [SEP] tokens
                pad_to_max_length=True,  # Pad to the maximum length
                return_attention_mask=True,  # Return the attention mask
                return_token_type_ids=False,  # Do not return token type IDs
                return_tensors='pt'  # Return PyTorch tensors
            )
            token = {
                'review': text,
                'input_ids': tokenized_review['input_ids'].flatten(),  # Flatten the input IDs tensor
                'attention_mask': tokenized_review['attention_mask'].flatten(),  # Flatten the attention mask tensor
            }
            pretrained_model = AutoModel.from_pretrained("bert-large-uncased")
            
            # Initialize and load the model
            model = QuestionClassifier(3, 1, 300, pretrained_model).to(device)
            checkpoint = torch.load(f'{model_path}/best_model_STW_{REMOVE_STOP_WORDS}_MAXLEN_300_TK_bert_large_Fold_{3}.pth')  # , map_location=torch.device(device))
            del checkpoint['model_state_dict']['pretrained_model.embeddings.position_ids']
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Make prediction
            prediction = model(input_ids=token['input_ids'].view(-1, 300).to(device), attention_mask=token['attention_mask'].view(-1, 300).to(device))
            _, preds = torch.max(prediction, dim=1)
            labels_inverse_mapping = {0: 'HQ', 1: 'LQ_CLOSE', 2: 'LQ_EDIT'}
            st.write("Question Quality is: ", labels_inverse_mapping[preds.item()])
