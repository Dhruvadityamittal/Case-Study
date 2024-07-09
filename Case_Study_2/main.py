import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, KFold
import torch
from transformers import AutoTokenizer, AutoModel,GPT2Model
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import numpy as np
import os
import warnings
from tqdm import tqdm
from data_handling import stackoverflowDataloader
from models import QuestionClassifier
from utils import merge_text, get_cleaned_text, get_model_evaluations

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Set device to GPU if available, else use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Argument parser to take command-line arguments
parser = argparse.ArgumentParser(description='Quality Classification of Stack Overflow Questions - Dhruv Aditya Mittal')
parser.add_argument('--BATCH_SIZE', default=16, type=int, help='Batch Size')
parser.add_argument('--EPOCHS', default=20, type=int, help='Number of training epochs.')
parser.add_argument('--MAX_LEN', default=300, type=int, help='Maximum Length of input sequence')
parser.add_argument('--TOKENIZER', type=str, default='bert', help='Tokenizer to use')
parser.add_argument('--REMOVE_STOP_WORDS', type=bool, default=True, help='Remove stop words from text')
parser.add_argument('--FOLDS', default=5, type=int, help='Number of Folds for cross-validation')
parser.add_argument('--EARLY_STOPPING_COUNT', default=5, type=int, help='Early Stopping Count')

args = parser.parse_args()

# Load training and validation data
train_df = pd.read_csv(r"C:\Users\Dhruv\Downloads\cluepoints_case\cluepoints_case\Case_Study_2\Data\stack_overflow_questions_train.csv")
val_df = pd.read_csv(r"C:\Users\Dhruv\Downloads\cluepoints_case\cluepoints_case\Case_Study_2\Data\stack_overflow_questions_valid.csv")

# Merge and clean text data
train_df = merge_text(train_df)
val_df = merge_text(val_df)
train_df = get_cleaned_text(train_df, args.REMOVE_STOP_WORDS)
val_df = get_cleaned_text(val_df, args.REMOVE_STOP_WORDS)

# Combine training and validation data for cross-validation
full_df = pd.concat([train_df, val_df])

# Encode class labels into numeric values
label_encoder = preprocessing.LabelEncoder()
full_df['class'] = label_encoder.fit_transform(full_df['class'])

# print(label_encoder.classes_)
# print(label_encoder.fit_transform(label_encoder.classes_))


# Paths to save results and models
result_path = f"Results/best_model_STW_{args.REMOVE_STOP_WORDS}_MAXLEN_{args.MAX_LEN}_TK_{args.TOKENIZER}_Folds_{args.FOLDS}_results.txt"
model_path = 'Saved_Models/'

# Lists to store evaluation metrics for each fold
best_val_accuracies, best_train_val_accuracies, test_accuracies = [], [], []
best_val_f1s, best_train_val_f1s, test_f1s = [], [], []

# Initialize tokenizer
if(args.TOKENIZER=='bert'):
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    pretrained_model = AutoModel.from_pretrained("google-bert/bert-base-uncased")
if(args.TOKENIZER=='bert_large'):
    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
    pretrained_model = AutoModel.from_pretrained("bert-large-uncased")





# Log the configuration
print(f"\n*************** Tokenizer: {args.TOKENIZER}, MAX LEN: {args.MAX_LEN}, Remove Stop Words: {args.REMOVE_STOP_WORDS} ***************\n")
with open(result_path, "w") as file:
    file.write(f"\n*************** Tokenizer: {args.TOKENIZER}, MAX LEN: {args.MAX_LEN}, Remove Stop Words: {args.REMOVE_STOP_WORDS} ***************\n")

# Set up K-Fold cross-validation
kf = KFold(n_splits=args.FOLDS, shuffle=True, random_state=1)

# Loop through each fold
for fold, (train_idx, val_idx) in enumerate(kf.split(full_df)):
    print(f"\n----------------------------------------Fold: {fold + 1}-----------------------------------------------------\n")
    
    # Split data into training and validation sets for the current fold
    fold_train_df, fold_val_df = full_df.iloc[train_idx], full_df.iloc[val_idx]
    fold_val_df, fold_test_df = train_test_split(fold_val_df, test_size=0.5, random_state=42)

    # Create dataloaders for training, validation, and test sets
    train_data = stackoverflowDataloader(fold_train_df, tokenizer, args.MAX_LEN)
    val_data = stackoverflowDataloader(fold_val_df, tokenizer, args.MAX_LEN)
    test_data = stackoverflowDataloader(fold_test_df, tokenizer, args.MAX_LEN)

    train_loader = DataLoader(train_data, batch_size=args.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.BATCH_SIZE, shuffle=False)

    # Initialize the model
    model = QuestionClassifier(len(fold_train_df['class'].unique()), args.BATCH_SIZE, args.MAX_LEN, pretrained_model).to(device)

    # Create directory for saving models
    os.makedirs(model_path, exist_ok=True)

    # Define optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss().to(device)

    # Initialize variables for tracking the best model
    es_count, best_val_acc, best_train_val_acc, best_val_f1, best_train_val_f1 = 0, 0, 0, 0, 0

    # Training loop
    for epoch in range(args.EPOCHS):
        pbar = tqdm(enumerate(train_loader))
        model.train()

        total_correct_predictions, total_train_loss, total_train = 0, 0, 0
        all_preds, all_actuals = [], []

        for batch_idx, batch_data in pbar:
            # if batch_idx > 1:
            #     break
            input_ids = batch_data["input_ids"].to(device)
            attention_mask = batch_data["attention_mask"].to(device)
            sentiments = batch_data["sentiments"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, sentiments)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if len(all_preds) == 0:
                all_preds = preds
                all_actuals = sentiments
            else:
                all_preds = torch.cat((all_preds, preds))
                all_actuals = torch.cat((all_actuals, sentiments))

            total_train += len(preds)
            total_train_loss += loss.item()

            # Update progress bar
            pbar.set_description(
                'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.4f}, Acc: {:.4f}, F1: {:.4f}'.format(
                    epoch, batch_idx + 1, len(train_loader), 100. * batch_idx / len(train_loader), loss.item(), 
                    accuracy_score(sentiments.cpu(), preds.cpu()), f1_score(sentiments.cpu(), preds.cpu(), average='macro')
                )
            )

        # Calculate training metrics
        train_loss = total_train_loss / total_train
        train_acc = accuracy_score(all_actuals.cpu(), all_preds.cpu())
        train_f1 = f1_score(all_actuals.cpu(), all_preds.cpu(), average='macro')

        # Evaluate model on validation set
        model.eval()
        val_loss, val_acc, val_f1 = get_model_evaluations(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch}/{args.EPOCHS} Train Loss: {train_loss:.4f}, Accuracy Train: {train_acc:.4f} F1 Train: {train_f1:.4f}, Val Loss: {val_loss:.4f}, Accuracy Val: {val_acc:.4f}, F1 Val: {val_f1:.4f}")

        # Save the best model based on validation accuracy
        if best_val_acc < val_acc:
            best_model = model
            best_val_acc = val_acc
            best_train_val_acc = train_acc
            best_val_f1 = val_f1
            best_train_val_f1 = train_f1

            # Save the model state
            torch.save({'model_state_dict': model.state_dict()}, f'{model_path}/best_model_STW_{args.REMOVE_STOP_WORDS}_MAXLEN_{args.MAX_LEN}_TK_{args.TOKENIZER}_Fold_{fold}.pth')
            es_count = 0
        else:
            print(f"Early Stopping Count: {es_count}")
            es_count += 1

        # Early stopping if validation accuracy does not improve for specified epochs
        if es_count == args.EARLY_STOPPING_COUNT or epoch + 1 == args.EPOCHS:
            if es_count == args.EARLY_STOPPING_COUNT:
                print("Early Stopping")
            best_val_accuracies.append(best_val_acc)
            best_train_val_accuracies.append(best_train_val_acc)
            best_val_f1s.append(best_val_f1)
            best_train_val_f1s.append(best_train_val_f1)
            break

    # Evaluate the best model on the test set
    test_loss, test_acc, test_f1 = get_model_evaluations(best_model, test_loader, loss_fn, device)
    test_accuracies.append(test_acc)
    test_f1s.append(test_f1)

    # Log the results for the current fold
    with open(result_path, "a") as file:
        file.write(f"\n----------------------------------------Fold: {fold + 1}-----------------------------------------------------\n")
        file.write(f"Best Validation Accuracy: {best_val_acc:.4f}, Training Accuracy for Best Validation epoch: {best_train_val_acc:.4f}, Test Accuracy: {test_acc:.4f}\n")
        file.write(f"Best Validation F1: {best_val_f1:.4f}, Training F1 for Best Validation epoch: {best_train_val_f1:.4f}, Test F1: {test_f1:.4f}\n")

    print(f"\nFold {fold+1}: Best Validation Accuracy: {best_val_acc:.4f}, Training Accuracy for Best Validation epoch: {best_train_val_acc:.4f}, Test Accuracy: {test_acc:.4f}")
    print(f"Fold {fold+1}: Best Validation F1: {best_val_f1:.4f}, Training F1 for Best Validation epoch: {best_train_val_f1:.4f}, Test F1: {test_f1:.4f}")

# Summary of results across all folds
result_summary = (
    "\nXXXXXXXXXXXXXXXXXXXXXXXXX Results Summary XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
    f"Train Accuracy: {np.mean(best_train_val_accuracies):.4f} ± {np.std(best_train_val_accuracies):.4f}, F1: {np.mean(best_train_val_f1s):.4f} ± {np.std(best_train_val_f1s):.4f}\n"
    f"Val Accuracy: {np.mean(best_val_accuracies):.4f} ± {np.std(best_val_accuracies):.4f}, F1: {np.mean(best_val_f1s):.4f} ± {np.std(best_val_f1s):.4f}\n"
    f"Test Accuracy: {np.mean(test_accuracies):.4f} ± {np.std(test_accuracies):.4f}, F1: {np.mean(test_f1s):.4f} ± {np.std(test_f1s):.4f}\n"
    "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
)

# Print and save the summary of results
print(result_summary)
with open(result_path, "a") as file:
    file.write(result_summary)
