from torch.utils.data import Dataset
import torch

class stackoverflowDataloader(Dataset):
    def __init__(self, df, tokenizer, MAX_LEN):
        self.df = df
        self.tokenizer = tokenizer
        self.MAX_LEN = MAX_LEN

    def __len__(self):
        # Return the total number of samples
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get the text and class label for the specified index
        text = str(self.df['text'].to_numpy()[idx])
        quality = self.df['class'].to_numpy()[idx]
        
        # Tokenize the text using the provided tokenizer
        tokenized_review = self.tokenizer.encode_plus(
            text,
            max_length=self.MAX_LEN,
            add_special_tokens=True,          # Add [CLS] and [SEP] tokens
            pad_to_max_length=True,           # Pad to the maximum length
            return_attention_mask=True,       # Return the attention mask
            return_token_type_ids=False,      # Do not return token type IDs
            return_tensors='pt'               # Return PyTorch tensors
        )

 
        
        # Return a dictionary containing the review text, input IDs, attention mask, and class label
        return {
            'review': text,
            'input_ids': tokenized_review['input_ids'].flatten(),      # Flatten the input IDs tensor
            'attention_mask': tokenized_review['attention_mask'].flatten(),  # Flatten the attention mask tensor
            'sentiments': torch.tensor(quality, dtype=torch.long)      # Convert class label to a tensor
        }
