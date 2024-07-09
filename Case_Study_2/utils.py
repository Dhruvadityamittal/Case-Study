# Utils file for data processing.
from nltk import pos_tag
import nltk
from wordcloud import WordCloud, STOPWORDS
import re, string, unicodedata
from nltk.corpus import stopwords
import torch
from sklearn.metrics import accuracy_score,  f1_score

# Download the stopwords from NLTK
nltk.download('stopwords')

# Set of English stopwords
stop = set(stopwords.words('english'))

# Add punctuation to the stopwords set
punctuation = list(string.punctuation)
stop.update(punctuation)

def merge_text(df):
    # Combine the 'Title' and 'Body' columns into a single 'text' column
    df['text'] = df['Title'] + " " + df['Body']  + " " + df['Tags'] 

    # Drop columns that are not needed for modeling
    # cols_to_drop = ['Id', 'Tags', 'CreationDate', 'Title', 'Body']

    columns_to_keep = ['text', 'Y']
    df = df[columns_to_keep]
    # df.drop(cols_to_drop, axis=1, inplace=True)

    # Rename the target column to 'class' for clarity
    df = df.rename(columns={"Y": "class"})

    # Print the total number of samples in the dataframe
    print("Total number of samples:", len(df))
    return df

def clean_text(text, remove_stopwords):
    # Convert text to lowercase
    text = text.lower()
    # Remove non-alphabetic characters (keeping only letters and spaces)
    text = re.sub(r"<([^>]+)>", r"\1 ", text)  # Pattern for tags
    
    text = ''.join(text.splitlines())
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    
    # print(remove_stopwords)
    if(remove_stopwords):
        text = [w.strip() for w in text.split(' ') if not w in stop and len(w.strip())>3]
        return ' '.join(text)
    
    return text

def get_cleaned_text(df, remove_stopwords):
    # Apply the clean_text function to the 'text' column of the dataframe
    df['text'] = df['text'].apply(clean_text, remove_stopwords=remove_stopwords)
    return df


def get_model_evaluations(model, dataloader,loss_fn,  device):
    
    
    total = 0
    total_loss = 0
    
    all_preds = []
    all_actuals  = []
    for batch_idx, d in enumerate(dataloader):
        # if(batch_idx>1):
        #     break
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        sentiments = d["sentiments"].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, sentiments)
        
        if(len(all_preds)==0) :
            all_preds = preds
            all_actuals = torch.tensor(sentiments).to(device)
        else:
            
            all_preds = torch.concat((all_preds, preds))
            all_actuals = torch.concat((all_actuals, sentiments))

        total += len(preds)
        total_loss += loss.item()

    
    f1_scr = f1_score(all_actuals.cpu(), all_preds.cpu(), average = 'macro')
    acc_score = accuracy_score(all_actuals.cpu(), all_preds.cpu())
    loss = total/total_loss
    return loss, acc_score, f1_scr
        # correct_predictions_val = torch.sum(preds == sentiments)
        # total_correct_predictions_val += correct_predictions_val.item()

        
