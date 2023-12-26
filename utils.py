import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import pipeline
from transformers import AutoModel, BertTokenizerFast
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW
from sklearn.utils import class_weight
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score,confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

class BERT_Arch(nn.Module):

    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        
        self.bert = bert 
        
        # dropout layer
        self.dropout = nn.Dropout(0.1)
      
        # relu activation function
        self.relu =  nn.ReLU()

        # dense layer 1
        self.fc1 = nn.Linear(768,512)
      
        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(512,13)

        #softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    #define the forward pass
    def forward(self, sent_id, mask):
        
        #pass the inputs to the model  
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
      
        x = self.fc1(cls_hs)

        x = self.relu(x)

        x = self.dropout(x)

        # output layer
        x = self.fc2(x)
      
        # apply softmax activation
        x = self.softmax(x)

        return x


# function to train the model
def train(model, train_dataloader, cross_entropy, optimizer, device='cpu'):
    
    model.train()
    total_loss, total_accuracy = 0, 0
    
    # iterate over batches
    progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Training")

    for step,batch in progress_bar:
        
        # push the batch to gpu
        batch = [r.to(device) for r in batch]
        #batch = [r for r in batch]
        sent_id, mask, labels = batch
        
        # clear previously calculated gradients 
        model.zero_grad()        

        # get model predictions for the current batch
        preds = model(sent_id, mask)

        # compute the loss between actual and predicted values
        loss = cross_entropy(preds, labels)

        # add on to the total loss
        total_loss = total_loss + loss.item()

        # backward pass to calculate the gradients
        loss.backward()

        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update parameters
        optimizer.step()

        preds = preds.detach().cpu().numpy()
        labels = labels.cpu().numpy()
        total_accuracy += accuracy_score(np.argmax(preds, axis=1), labels)

    avg_loss = total_loss / len(train_dataloader)
    avg_accuracy = total_accuracy / len(train_dataloader)
    return avg_loss, avg_accuracy

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return time.strftime("%H:%M:%S", time.gmtime(elapsed_rounded))

def evaluate(model, val_dataloader, cross_entropy, device='cpu'):
    print("\nEvaluating...")
  
    # Move model to the right device
    model = model.to(device)
    model.eval()

    total_loss, total_accuracy = 0, 0
    
    # Tracking variables
    predictions , true_labels = [], []
    
    # Measure the evaluation time
    t0 = time.time()

    progress_bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc="Evaluating")

    # Evaluate data for one epoch
    for step, batch in progress_bar:
        
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        sent_id, mask, labels = batch

        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            preds = model(sent_id, mask)

            # Calculate the loss between actual and predicted values
            loss = cross_entropy(preds,labels)
            total_loss += loss.item()

            # Move preds to the CPU
            preds = preds.detach().cpu().numpy()
            labels = labels.to('cpu').numpy()

            total_accuracy += accuracy_score(np.argmax(preds, axis=1), labels)

    # Calculate the average loss over all of the batches.
    avg_loss = total_loss / len(val_dataloader)
    
    # Calculate the average accuracy over all predictions.
    avg_accuracy = total_accuracy / len(val_dataloader)

    # Measure how long the evaluation took.
    evaluation_time = format_time(time.time() - t0)

    print("  Evaluation took: {:}".format(evaluation_time))

    # Clear some memory
    if device == 'cuda':
        torch.cuda.empty_cache()

    return avg_loss, avg_accuracy, predictions