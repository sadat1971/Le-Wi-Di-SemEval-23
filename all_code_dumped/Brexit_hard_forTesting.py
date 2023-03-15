
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

def tokenization_for_BERT(df, path="/media2/special/Sadat/Brexit_v2/Data/", filename="put_the_filename_here", saveit="No"):

    if "tokenized" not in df.columns:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        df["tokenized"] = df["text"].apply(lambda sent:tokenizer.encode(sent, add_special_tokens=True, 
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
        D_in, H, D_out = 768, hidden_size, 2 #Just one hidden layer with 50 units in it
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
    labels = np.array(list(df["annot_lab"]))
    soft = np.array(list(df["Soft_lab_1"]))
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
                      lr=5e-5,    # Default learning rate
                      eps=1e-8    # Default epsilon value
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


def train_model(model, train_dataloader,val_dataloader, epochs, evaluation, device, optimizer, scheduler, class_weights):
    """Train the BertClassifier model.
    """
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
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
            loss = loss_fn(logits, b_labels)
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
            val_loss, val_accuracy, f1, ce = evaluate(model, val_dataloader, device)
            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-"*70)

        val_loss, val_accuracy, val_f1sc, df = evaluate(model, val_dataloader, device)
        df.to_pickle(args.log_dir + "/temp/annotator_" + str(ANN)  + ".pkl")
            


        train_loss.append(avg_train_loss)
        valid_loss.append(val_loss)
        val_acc.append(val_accuracy)
        val_f1.append(val_f1sc)
        with open(args.log_dir+ "/" + "BERT_hard_results.txt", 'a') as r:
            res = "#epoch: " + str(epoch_i) + "  #train_loss " + str(round(avg_train_loss,4)) +  "  #valid_loss "  + str(round(val_loss ,4)) \
             +  "  #valid_acc "  + str(round(val_accuracy ,4)) +  "  #valid_f1 "  + str(round(val_f1sc ,4)) 
            r.write(res)
            r.write("\n")

        print("\n")
        if args.model_and_pediction_save!="No":
            model_name = "BERT_" + args.train_domain +"_epoch_" + str(epoch_i) + ".pth"
            path = args.model_path_dir + model_name
            torch.save(model.state_dict(), path)

    loss_record["train_loss"] = train_loss
    loss_record["valid_loss"] = valid_loss
    loss_record["val_acc"] = val_acc
    loss_record["val_f1"] = val_f1


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
    loss_fn = nn.CrossEntropyLoss()
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
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            
        all_logits.append(logits)
        all_soft.append(soft)
    
    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)
    all_soft = torch.cat(all_soft, dim=0)

    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy() 

    df = pd.DataFrame()
    df["GT"] = all_labels
    df["probs_0"] = probs[:,0]
    df["probs_1"] = probs[:,1]
    df["prediction"] = df["probs_0"].apply(lambda x:0 if x>0.5 else 1)

    df.to_pickle(args.log_dir + "ann_res/Ann_nometa_" + str(ANN) + ".pkl")
    avg_loss = total_loss / len(dataloader)
    ce = cross_entropy(targets=all_soft.cpu().numpy(), predictions=probs[:,1], epsilon = 1e-12)
    print("acc is: ", accuracy_score(df["GT"], df["prediction"]))
    print("f1 is", f1_score(df["GT"], df["prediction"], average='micro'))
    return avg_loss, accuracy_score(df["GT"], df["prediction"]), f1_score(df["GT"], df["prediction"], average='micro'), df



def sample_as_instructed(df, instruction="none"):
  if instruction=="Over":
    n = RandomOverSampler(random_state=42)
    dfo = n.fit_resample(df, df["soft_lab"])[0]
    return dfo
  
  elif instruction=="Under":
    n = RandomUnderSampler(random_state=42)
    dfu = n.fit_resample(df, df["soft_lab"])[0]
    return dfu
  else:
    return df

set_seed(42)

parser = argparse.ArgumentParser(description='BERT model arguments')

parser.add_argument("--data_dir", 
                    type=str, 
                    default="/media2/special/Sadat/Brexit_v2/Data/",
                     help="Input data path.")
parser.add_argument("--log_dir",
                     type=str, 
                     default="/media2/special/Sadat/Brexit_v2/Result/",
                     help="Store result path.")
parser.add_argument("--model_path_dir",
                     type=str, 
                     default="/media2/special/Sadat/Brexit_v2/Data/Result/",
                     help="Store result path.")
parser.add_argument("--batch_size", type=int, default=8, help="what is the batch size?")
parser.add_argument("--device", type=str, default="cuda:0", help="what is the device?")
parser.add_argument("--dropout", type=np.float32, default=0.1, help="what is the dropout in FC layer?")
parser.add_argument("--valid_with", type=str, default="test", help="Validate with test data?")
parser.add_argument("--epochs", type=int, default=4, help="what is the epoch size?")
parser.add_argument("--hidden_size", type=int, default=32, help="what is the hidden layer size?")
parser.add_argument("--sample", type=str, default="none", help="what kind of sampling you want? Over, Under or none")
parser.add_argument("--model_and_pediction_save", type=str, default="No", help="Do you want to save the model and the results as well?")
parser.add_argument("--fast_track", type=np.float32, default=1.00, help="Do you want a fast track train ?")


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

train = pd.read_pickle(args.data_dir + "Brexit_train.pkl")
train = tokenization_for_BERT(train, path=args.data_dir, filename="Brexit_train" , saveit="Yes")
train["Soft_lab_1"] = train.soft_label.apply(lambda x:x['1'])


dev = pd.read_pickle(args.data_dir + "Brexit_dev.pkl")
dev = tokenization_for_BERT(dev, path=args.data_dir, filename="Brexit_dev" , saveit="Yes")
dev["Soft_lab_1"] = dev.soft_label.apply(lambda x:x['1'])

train = pd.concat([train, dev], axis=0)
train.reset_index(inplace=True, drop=True)


test = pd.read_pickle(args.data_dir + "Brexit_test.pkl")
test = tokenization_for_BERT(test, path=args.data_dir, filename="Brexit_test" , saveit="Yes")
test["Soft_lab_1"] = test.soft_label.apply(lambda x:x['1'])

annotator_wise_epochs = [2,3,3,2,3,2]
for ANN in range(6):
    train["annot_lab"] = train.annotations.apply(lambda x:x[ANN])
    test["annot_lab"] = test.annotations.apply(lambda x:x[ANN])
    tok_tr, mask_tr, lab_tr, soft_tr = prepare_train_and_valid(train)
    train_dataloader = create_dataloader(tok_tr, lab_tr, mask_tr, soft_tr, batch_size=args.batch_size)

    tok_ts, mask_ts, lab_ts, soft_ts = prepare_train_and_valid(test)
    test_dataloader = create_dataloader(tok_ts, lab_ts, mask_ts, soft_ts, batch_size=args.batch_size, mode="test")
    valid_dataloader = test_dataloader

    #valid_dataloader = create_dataloader(tokenized_valid, labels_valid, attention_masks_valid, batch_size=args.batch_size)

    weights = [train[train.annot_lab==1].shape[0]/train.shape[0],  train[train.annot_lab==0].shape[0]/train.shape[0]]
    class_weights = torch.FloatTensor(weights).to(device)


    with open(args.log_dir+ "/" + "Fortesting_BERT_hard_results.txt", 'a') as r:
        r.write("---- annotator {} model -----".format(ANN))
        r.write("\n")

    bert_classifier, optimizer, scheduler = initialize_model(epochs=annotator_wise_epochs[ANN], train_dataloader=train_dataloader, \
    device=args.device, H=args.hidden_size,  D_in=768, dropout=args.dropout, classes=2)


    df = train_model(bert_classifier, train_dataloader, valid_dataloader, epochs=args.epochs, evaluation=True, device=args.device,
            optimizer=optimizer, scheduler=scheduler, class_weights=class_weights)

    print(df)

    end = time.time()

    with open(args.log_dir+ "/" + "Fortesting_BERT_hard_results.txt", 'a') as r:
        timerq = str(end-start)
        r.write("################ TIME ###############\n")
        r.write(timerq)
        r.write("\n\n\n")


### Compute the f1 and ce score:

otherinfo= {
    0:{"11":0.45,
       "10":0.17,
       "01":0.28,
       "00":0.00},
    
    1:{"11":0.29,
       "10":0.09,
       "01":0.52,
       "00":0.03},
2:{"11":0.48,
       "10":0.03,
       "01":0.37,
       "00":0.02},
3:{"11":0.92,
       "10":0.44,
       "01":0.00,
       "00":0.00},
4:{"11":0.81,
       "10":0.55,
       "01":0.50,
       "00":0.01},
5:{"11":0.87,
       "10":0.49,
       "01":0.2,
       "00":0.03}     
             }

def compute_aggregated_performance(preds, probs, soft_lab, hard_lab):
    all_preds = []
    all_probs = []
    for t in range(len(preds[0])):
        sum_of_preds = 0
        sum_of_probs = 0
        for ann in range(6):
            sum_of_preds += preds[ann][t]
            sum_of_probs += probs[ann][t]
        all_probs.append(sum_of_probs/6)
        all_preds.append(sum_of_preds/6)
    all_aggr_preds = [0 if i<.5 else 1 for i in all_preds] 
    pred_wice_ce = cross_entropy(soft_lab, all_preds)
    prob_wice_ce = cross_entropy(soft_lab, all_probs)
    f1_micro = f1_score(hard_lab, all_aggr_preds, average = "micro")
    f1_bin = f1_score(hard_lab, all_aggr_preds, average = "binary")
    return pred_wice_ce, prob_wice_ce, f1_micro, f1_bin

def compute_new_probs(of, ag, smax_prob, ann_num, wt):
    otherinfo_prob=[]
    for o, a in zip(of, ag):
        otherinfo_prob.append(otherinfo[ann_num][str(o)+str(a)])
    
    return [(i+ wt*j)/(2) for i, j in zip(smax_prob, otherinfo_prob)]


wt=2
probs_pred = pd.DataFrame()
test =pd.read_pickle(args.data_dir + "Brexit_test.pkl")
test["Soft_lab_1"] = test.soft_label.apply(lambda x:x['1'])
soft_lab = test["Soft_lab_1"].values
probs = []
preds = []
probs_otherinfo = []
for ann in range(0,6):
    Ann1 = pd.read_pickle(args.log_dir + "/temp/annotator_" +str(ann) + ".pkl")
    probs_pred["Ann_" +str(ann)+"_probs"] = Ann1.probs_1
    probs_pred["Ann_" +str(ann)+"_preds"] = Ann1.prediction
    probs.append(Ann1.probs_1.tolist())
    preds.append(Ann1.prediction.tolist())
    of = test.Offensive.apply(lambda x:x[ann]).tolist()
    ag = test.Aggressive.apply(lambda x:x[ann]).tolist()
    probs_otherinfo.append(compute_new_probs(of, ag, Ann1.probs_1.tolist(), ann, wt))
#####
preds_otherinfo = []
for A in probs_otherinfo:
    annot = []
    for p in A:
        if p>=.5:
            annot.append(1)
        else:
            annot.append(0)
    preds_otherinfo.append(annot)

df_spcl = pd.DataFrame()
df_spcl["spcl"] = probs_otherinfo
df_spcl.to_pickle(args.log_dir + "ann_res/df_spcl.pkl")


df2=pd.DataFrame()
for i in range(1,7):
    ann_name = "ANN_" + str(i)
    df2[ann_name] = probs_otherinfo[i-1]
df2.to_pickle(args.log_dir + "ann_res/meta_added.pkl")
#####
pred_wise, prob_wise, f1m, f1b = compute_aggregated_performance(preds_otherinfo, probs_otherinfo, soft_lab, test.hard_label.tolist())
print("(metadata) For weight {}, f1 micro is {:.4f}, f1 binary is {:.4f}  the CE loss is softmax score wise is {:.4f} and average prediction wise {:.4f}".format(wt, f1m, f1b, prob_wise,pred_wise))

pred_wise, prob_wise, f1m, f1b = compute_aggregated_performance(preds, probs, soft_lab, test.hard_label.tolist())
print("(No metadata) For weight {}, f1 micro is {:.4f}, f1 binary is {:.4f}  the CE loss is softmax score wise is {:.4f} and average prediction wise {:.4f}".format(wt, f1m, f1b, prob_wise,pred_wise))