
#training the holistic model
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
import pickle
import time
import random
import os
import sys
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import argparse
from transformers import BertTokenizer
from tqdm import tqdm
import torch.nn.functional as F
import nltk
import re
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


start = time.time()

def tokenization_for_BERT(df, path="/media2/special/Sadat/Convabuse/Data/", filename="put_the_filename_here", saveit="No"):

    if "tokenized" not in df.columns:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        df["tokenized"] = df["user_text_only"].apply(lambda sent:tokenizer.encode(sent, add_special_tokens=True, 
                                                                                max_length=512, truncation=True,
                                                                                padding='max_length', 
                                                                                return_attention_mask=False))

        if saveit!="No":
            df.to_pickle(path + filename + ".pkl")
        

    return df







# Create the BertClassfier class
class BertClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, hidden_size=50, dropout=0): 
        #The freeze_bert is set to false to make sure our model DOES do some fine tuning
        #on the BERT layers
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, hidden_size, 1 #Just one hidden layer with 50 units in it
        # Instantiate BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # We'd like to build a fully connected neural network for classification task. We choose to keep the droput to 
        #zero for now. Later on, we will see if the dropouts can be adjusted to avoid overfitting. 

        self.fc1 = nn.Linear(D_in, H)
        self.fc2 = nn.Linear(H, D_out)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        
    def forward(self, input_ids, attention_mask):
        '''
        This function takes input as the training set and attention mask and 
        gives the output as porbability values.
        Inputs-->
        input_ids: the training set tensor. MUST be of size [batch_size, tokenization_length]
        attention_mask: The 1/0 indication of input_ids. MUST be of size [batch_size, tokenization_length]
        output-->
        logits: Output values of shape [batch_size, number_of_labels]. Now keep it in mind, this is NOT
        softmax, it is only logits.
        '''
        # Feed input to BERT
        bert_cls_outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)[0][:, 0, :]
        

        # Feed input to classifier to compute logits
        out1 = self.fc1(bert_cls_outputs)
        out1 = self.relu(out1)
        out1 = self.dropout(out1)
        logits = self.fc2(out1)
        return logits

def prepare_train_and_valid(df):

    tokenized = np.array(list(df["tokenized"]))
    attention_masks = np.where(tokenized>0, 1, 0)
    ## labels
    labels = np.array(list(df["hard_label"]))
    soft = np.array(list(df["SOFT"]))
    tokenized, attention_masks, labels, soft = torch.tensor(tokenized), torch.tensor(attention_masks), torch.tensor(labels), torch.tensor(soft)
    return tokenized, attention_masks, labels, soft


def initialize_model(epochs, train_dataloader, device, H, D_in=768, dropout=0.25, classes=2):
    """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    """
    # Instantiate Bert Classifier
    bert_classifier = BertClassifier(hidden_size=H, dropout=dropout)
    # Tell PyTorch to run the model on GPU
    bert_classifier.to(device)

    # Create the optimizer
    optimizer = AdamW(bert_classifier.parameters(),
                      lr=args.lr,    # Default learning rate
                      eps=1e-8,    # Default epsilon value
                      weight_decay=args.weight_decay
                      )

    # Total number of training steps
    total_steps = len(train_dataloader) * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0, # Default value
                                                num_training_steps=total_steps)
    return bert_classifier, optimizer, scheduler

def create_dataloader(features, labels, attention_masks, soft, batch_size, mode="Train"):
    # Create the DataLoader for our training set
    '''
    This function will create a dataloader for our training set. The dataloader will help to feed the randomly 
    sampled data on each batch. The batch size is selected to be 16, is simply as instructed in the original
    paper. 
    '''
    data = TensorDataset(features, attention_masks, labels, soft)
    if mode=="Train":
        sampler = RandomSampler(data)
    else:
        sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return dataloader

def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def train_model(model, train_dataloader,val_dataloader, epochs, evaluation, device, optimizer, scheduler):
    """Train the BertClassifier model.
    """
    loss_fn = nn.MSELoss()
    # Start training loop
    print("Start training...\n")
    loss_record = pd.DataFrame()
    train_loss = []
    valid_loss = []
    val_acc = []
    val_f1 = []
    best_loss = np.inf
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*70)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # Put the model into the training mode
        model.train()

        # For each batch of training data...
        for step, batch in enumerate((train_dataloader)):
            batch_counts +=1
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels, soft = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids, b_attn_mask)

            # Compute loss and accumulate the loss values
            loss = loss_fn(logits.view(logits.shape[0],), soft.type(torch.float))
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        print("-"*70)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            #val_loss, val_accuracy, f1, ce = evaluate(model, val_dataloader, device)
            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            
            # print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            # print("-"*70)

        ce, f1, val_loss, df = evaluate(model, val_dataloader, device)
        # if val_loss<best_loss:
        #     best_loss = val_loss
        #     df.to_pickle(args.log_dir + "Annotator_" + str(ANN) + "_split_" + str(args.split) + ".pkl")
        
        print(ce, f1)
        train_loss.append(avg_train_loss)
        valid_loss.append(val_loss)
        

        if val_loss<best_loss:
            best_loss = val_loss
            try:
                df.to_pickle(args.log_dir + "temp/probs.pkl")
            except:
                os.mkdir(args.log_dir + "temp")
                df.to_pickle(args.log_dir + "temp/probs.pkl")

        with open(args.log_dir+ "/" + str(args.ver) + "_BERT_MSEloss.txt", 'a') as r:
            res = "#epoch: " + str(epoch_i) + "  #train_loss " + str(round(avg_train_loss,4)) +  "  #valid_loss "  + str(round(val_loss ,4)) \
             +  "  #valid_f1 "  + str(round(f1 ,4)) +  "  #valid_ce "  + str(round(ce ,4)) 
            r.write(res)
            r.write("\n")

        print("\n")
        if args.model_and_pediction_save!="No":
            model_name = "BERT_Fold_" + str(args.split) +"_epoch_" + str(epoch_i) + ".pth"
            path = args.model_path_dir + model_name
            torch.save(model.state_dict(), path)

    # loss_record["train_loss"] = train_loss
    # loss_record["valid_loss"] = valid_loss
    # loss_record["val_acc"] = val_acc
    # loss_record["val_f1"] = val_f1


    return loss_record

def cross_entropy(targets, predictions, epsilon = 1e-12):                                
    predictions = np.clip(predictions, epsilon, 1. - epsilon)                                      
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions+1e-9))/N
    return ce

def evaluate(model, dataloader, device):
    """Perform a forward pass on the trained BERT model to predict probabilities
    on the test set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    loss_fn = nn.MSELoss()
    model.eval()

    all_probs = []
    all_labels = []

    # For each batch in our test set...
    total_loss = 0
    all_logits = []
    all_soft = []
    for batch in (dataloader):
        # Load batch to GPU
        b_input_ids, b_attn_mask, labels, soft = tuple(t.to(device) for t in batch)
        all_labels = all_labels + (labels.tolist())
        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
            loss = loss_fn(logits.view(logits.shape[0],), soft)
            total_loss += loss.item()
            
        all_logits.append(logits)
        all_soft.append(soft)
    
    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)
    all_soft = torch.cat(all_soft, dim=0)

    # Apply softmax to calculate probabilities
    #probs = F.softmax(all_logits, dim=1).cpu().numpy() 
    avg_loss = total_loss / len(dataloader)
    T = list(all_soft.cpu().numpy())
    P = list(all_logits.cpu().numpy().reshape(all_logits.shape[0],))
    df = pd.DataFrame()
    df["soft_lab"] = T
    df["probs"] = P
    ce = cross_entropy(T, P)
    T = [0 if t<.5 else 1 for t in T]
    P_disc = [0 if t<.5 else 1 for t in P]
    f1 = f1_score(T, P_disc, average='micro')
    ## For saving
    probs_1 = []
    probs_0 = []

    for m in P:
        if m>1:
            probs_1.append(1)
            probs_0.append(0)
        elif m<0:
            probs_1.append(0)
            probs_0.append(1)
        else:
            probs_1.append(m)
            probs_0.append(1-m)
    
    record_results = pd.DataFrame()
    record_results["p_disc"] = P_disc
    record_results["probs_0"] = probs_0
    record_results["probs_1"] = probs_1 
    # record_results["T_disc"] = T_disc
    # record_results["T"] = T

    record_results.to_csv(args.log_dir + "convabuse.tsv", sep='\t', header=False, index=False)
    return ce, f1*100, avg_loss, df



set_seed(42)

parser = argparse.ArgumentParser(description='BERT model arguments')

parser.add_argument("--data_dir", 
                    type=str, 
                    default="/media2/special/Sadat/Convabuse/Data/",
                     help="Input data path.")
parser.add_argument("--log_dir",
                     type=str, 
                     default="/media2/special/Sadat/Convabuse/Result/",
                     help="Store result path.")
parser.add_argument("--model_path_dir",
                     type=str, 
                     default="/media2/special/Sadat/Convabuse/Result/",
                     help="Store result path.")
parser.add_argument("--batch_size", type=int, default=8, help="what is the batch size?")
parser.add_argument("--device", type=str, default="cuda:0", help="what is the device?")
parser.add_argument("--dropout", type=np.float32, default=0.1, help="what is the dropout in FC layer?")
parser.add_argument("--epochs", type=int, default=4, help="what is the epoch size?")
parser.add_argument("--hidden_size", type=int, default=32, help="what is the hidden layer size?")
parser.add_argument("--sample", type=str, default="none", help="what kind of sampling you want? Over, Under or none")
parser.add_argument("--model_and_pediction_save", type=str, default="No", help="Do you want to save the model and the results as well?")
parser.add_argument("--fast_track", type=np.float32, default=1.00, help="Do you want a fast track train ?")
parser.add_argument("--lr", type=np.float32, default=5e-5, help="The learning rate ?")
parser.add_argument("--weight_decay", type=np.float32, default=0.01, help="The weight decay ?")
parser.add_argument("--ver", type=str, default="0", help="the version?")


args = parser.parse_args()




device = args.device
# first, we'll see if we have CUDA available
if torch.cuda.is_available():       
    device = torch.device(device)
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

train = pd.read_pickle(args.data_dir + "Convabuse_train.pkl")
train = tokenization_for_BERT(train, path=args.data_dir, filename= "Convabuse_train.pkl" , saveit="Yes").sample(frac=args.fast_track)
train["SOFT"] = train.soft_label.apply(lambda x:x['1'])
print(train.shape)

dev = pd.read_pickle(args.data_dir + "Convabuse_dev.pkl")
dev = tokenization_for_BERT(dev, path=args.data_dir, filename= "Convabuse_dev.pkl" , saveit="Yes")
dev["SOFT"] = dev.soft_label.apply(lambda x:x['1'])

tok_tr, mask_tr, lab_tr, soft_tr = prepare_train_and_valid(train)
train_dataloader = create_dataloader(tok_tr, lab_tr, mask_tr, soft_tr, batch_size=args.batch_size)

tok_ts, mask_ts, lab_ts, soft_ts = prepare_train_and_valid(dev)
valid_dataloader = create_dataloader(tok_ts, lab_ts, mask_ts, soft_ts, batch_size=args.batch_size, mode="Test")

#valid_dataloader = create_dataloader(tokenized_valid, labels_valid, attention_masks_valid, batch_size=args.batch_size)


with open(args.log_dir+ str(args.ver) + "_BERT_MSEloss.txt", 'a') as r:
    rs1 = "\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n---- Model Properties -----\n"
    rs2 = " Batch Size: " + str(args.batch_size) + "\n Learning Rate: " + str(args.lr) + "\n Dropout :" + str(args.dropout) + "\n Hidden Size: " \
    + str(args.hidden_size) +  "\n Weight Decay: " + str(args.weight_decay) +  "\n\n"
    r.write(rs1 + rs2)
    r.write("\n")

bert_classifier, optimizer, scheduler = initialize_model(epochs=args.epochs, train_dataloader=train_dataloader, \
device=args.device, H=args.hidden_size,  D_in=768, dropout=args.dropout, classes=1)


df = train_model(bert_classifier, train_dataloader, valid_dataloader, epochs=args.epochs, evaluation=True, device=args.device,
        optimizer=optimizer, scheduler=scheduler)

print(df)

end = time.time()

with open(args.log_dir+ "/" + str(args.ver) + "_BERT_MSEloss.txt", 'a') as r:
    timerq = str(end-start)
    r.write("################ TIME ###############\n")
    r.write(timerq)
    r.write(" seconds\n\n\n")
