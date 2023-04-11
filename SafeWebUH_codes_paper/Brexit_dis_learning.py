
#training the holistic model
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
import time
import random
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import argparse
from transformers import BertTokenizer
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LinearRegression

start = time.time()

## Load the json files and convert to pickle
train = pd.read_json(path + "HS-Brexit_dataset/HS-Brexit_train.json", orient='index')
dev  = pd.read_json(path + "HS-Brexit_dataset/HS-Brexit_dev.json", orient='index')
test  = pd.read_json(path + "data_post-competition/data_post-competition/HS-Brexit_dataset/HS-Brexit_test.json", orient='index')
train.to_pickle(path + "Brexit_v2/Data/Brexit_train.pkl")
dev.to_pickle(path + "Brexit_v2/Data/Brexit_dev.pkl")
test.to_pickle(path + "Brexit_v2/Data/Brexit_test.pkl")


def tokenization_for_BERT(df, path="/path/to/pickle/files/", filename="put_the_filename_here", saveit="No"):

    if "tokenized" not in df.columns:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        df["tokenized"] = df["text"].apply(lambda sent:tokenizer.encode(sent, add_special_tokens=True, 
                                                                                max_length=512, truncation=True,
                                                                                padding='max_length', 
                                                                                return_attention_mask=False))

        if saveit!="No":
            df.to_pickle(path + filename + ".pkl")
        

    return df







# Create the BertReg class
class BertReg(nn.Module):
    """Bert Model for Regression Tasks.
    """
    def __init__(self, hidden_size=50, dropout=0): 

        super(BertReg, self).__init__()
        # Specify hidden size of BERT, hidden size of our regressor, and number of labels
        D_in, H, D_out = 768, hidden_size, 1 
        # Instantiate BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # We'd like to build a fully connected neural network for regressing on the soft label

        self.fc1 = nn.Linear(D_in, H)
        self.fc2 = nn.Linear(H, D_out)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        
    def forward(self, input_ids, attention_mask):
        '''
        This function takes input as the training set and attention mask and 
        gives the output as regression values.
        Inputs-->
        input_ids: the training set tensor. MUST be of size [batch_size, tokenization_length]
        attention_mask: The 1/0 indication of input_ids. MUST be of size [batch_size, tokenization_length]
        output-->
        prediction on soft label
        '''
        # Feed input to BERT
        bert_cls_outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)[0][:, 0, :]
        

        # Feed input to regressor to compute PREDICTIONS
        out1 = self.fc1(bert_cls_outputs)
        out1 = self.relu(out1)
        out1 = self.dropout(out1)
        PREDICTIONS = self.fc2(out1)
        return PREDICTIONS

def prepare_train_and_valid(df, mode="Train"):

    tokenized = np.array(list(df["tokenized"]))
    attention_masks = np.where(tokenized>0, 1, 0)
    ## labels
    labels = np.array(list(df["hard_label"]))
    soft = np.array(list(df["SOFT"]))
    if mode=="Test":
        tokenized, attention_masks = torch.tensor(tokenized), torch.tensor(attention_masks)
        return tokenized, attention_masks
    else:
        tokenized, attention_masks, labels, soft = torch.tensor(tokenized), torch.tensor(attention_masks), torch.tensor(labels), torch.tensor(soft)
        return tokenized, attention_masks, labels, soft


def initialize_model(epochs, train_dataloader, device, H, D_in=768, dropout=0.25, classes=2):
    """Initialize the Bert regressor, the optimizer and the learning rate scheduler.
    """
    # Instantiate Bert regressor
    bert_regressor = BertReg(hidden_size=H, dropout=dropout)
    # Tell PyTorch to run the model on GPU
    bert_regressor.to(device)

    # Create the optimizer
    optimizer = AdamW(bert_regressor.parameters(),
                      lr=5e-5,    # Default learning rate
                      eps=1e-8    # Default epsilon value
                      )

    # Total number of training steps
    total_steps = len(train_dataloader) * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0, # Default value
                                                num_training_steps=total_steps)
    return bert_regressor, optimizer, scheduler

def create_dataloader(features, labels, attention_masks, soft, batch_size, mode="Train"):
    # Create the DataLoader for our training set
    '''
    This function will create a dataloader for our training set. 
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
    """Train the BertReg model.
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

            # Perform a forward pass. This will return PREDICTIONS.
            PREDICTIONS = model(b_input_ids, b_attn_mask)

            # Compute loss and accumulate the loss values
            loss = loss_fn(PREDICTIONS.view(PREDICTIONS.shape[0],), soft.type(torch.float))
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
            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            
            print("-"*70)

    df = evaluate(model, val_dataloader, device)
            


    return df

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
    all_PREDICTIONS = []
    all_soft = []
    for batch in (dataloader):
        # Load batch to GPU
        b_input_ids, b_attn_mask, labels, soft = tuple(t.to(device) for t in batch)
        all_labels = all_labels + (labels.tolist())
        with torch.no_grad():
            PREDICTIONS = model(b_input_ids, b_attn_mask)

            
        all_PREDICTIONS.append(PREDICTIONS)
        all_soft.append(soft)
    
    # Concatenate PREDICTIONS from each batch
    all_PREDICTIONS = torch.cat(all_PREDICTIONS, dim=0)
    all_soft = torch.cat(all_soft, dim=0)

    P = list(all_PREDICTIONS.cpu().numpy().reshape(all_PREDICTIONS.shape[0],))
    P_disc = [0 if t<.5 else 1 for t in P]
    df = pd.DataFrame()
    df["probs"] = P
    df["prediction"] = P_disc
    return df



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
                    default="/path/to/data/",
                     help="Input data path.")
parser.add_argument("--log_dir",
                     type=str, 
                     default="/path/to/results/",
                     help="Store result path.")
parser.add_argument("--model_path_dir",
                     type=str, 
                     default="/path/to/results/",
                     help="Store result path.")
parser.add_argument("--batch_size", type=int, default=8, help="what is the batch size?")
parser.add_argument("--device", type=str, default="cuda:0", help="what is the device?")
parser.add_argument("--dropout", type=np.float32, default=0.1, help="what is the dropout in FC layer?")
parser.add_argument("--epochs", type=int, default=4, help="what is the epoch size?")
parser.add_argument("--hidden_size", type=int, default=32, help="what is the hidden layer size?")
parser.add_argument("--lr", type=np.float32, default=5e-5, help="The learning rate ?")
parser.add_argument("--weight_decay", type=np.float32, default=0.01, help="The weight decay ?")
parser.add_argument("--use_metadata", type=str, default="yes", help="Do you want to use metadata?")


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
train["SOFT"] = train.soft_label.apply(lambda x:x['1'])

dev = pd.read_pickle(args.data_dir + "Brexit_dev.pkl")
dev = tokenization_for_BERT(dev, path=args.data_dir, filename="Brexit_dev" , saveit="Yes")
dev["SOFT"] = dev.soft_label.apply(lambda x:x['1'])


test = pd.read_pickle(args.data_dir + "Brexit_test.pkl")
test = tokenization_for_BERT(test, path=args.data_dir, filename= "Brexit_test.pkl" , saveit="Yes")
test["SOFT"] = test.soft_label.apply(lambda x:x['1'])

train = pd.concat([train, dev], axis=0)

tok_tr, mask_tr, lab_tr, soft_tr = prepare_train_and_valid(train)
train_dataloader = create_dataloader(tok_tr, lab_tr, mask_tr, soft_tr, batch_size=args.batch_size)

tok_ts, mask_ts = prepare_train_and_valid(test, mode="Test")
valid_dataloader = create_dataloader(tok_ts, tok_ts, mask_ts, mask_ts, batch_size=args.batch_size, mode="Test")



bert_regressor, optimizer, scheduler = initialize_model(epochs=args.epochs, train_dataloader=train_dataloader, \
device=args.device, H=args.hidden_size,  D_in=768, dropout=args.dropout, classes=1)


df = train_model(bert_regressor, train_dataloader, valid_dataloader, epochs=args.epochs, evaluation=True, device=args.device,
        optimizer=optimizer, scheduler=scheduler)

print(df)

end = time.time()

with open(args.log_dir+ "/" + "BERT_dis_learning_results.txt", 'a') as r:
    timerq = str(end-start)
    r.write("################ TIME ###############\n")
    r.write(timerq)
    r.write("\n\n\n")




### Using metadata on top of the predicted values

## The idea is simple: Make a linear reression model to predict the soft
# label based on the aggressive and offensive ratings. Then just create an weighted 
# average. We found the right weight simply by results observed in the dev set


test =pd.read_pickle(args.path + "Brexit_v2/Data/Brexit_test.pkl")
result =pd.read_pickle(args.path + "Brexit_v2/Result/temp/testresult" + ".pkl")


soft = list(train.soft_label.apply(lambda x:x['1']))
aggr = [sum(a)/len(a) for a in train.Aggressive.tolist()]
offen = [sum(a)/len(a) for a in train.Offensive.tolist()]
X_other = []
for a, o in zip(aggr, offen):
    X_other.append([a,o])
X_other = np.array(X_other)
y = np.array(soft)
reg = LinearRegression().fit(X_other, y)
# print(reg.coef_) #0.58299109, 0.33627756
# print(reg.intercept_) #  - 0.0023297653922081385

def compute_reg_score(soft, hard, probs, agg, off, wt, save_result=True, produce_result=True):
    
    if args.use_metadata=="yes":
        other_score = np.array(agg)*reg.coef_[0] + np.array(off)*reg.coef_[1] - reg.intercept_
        target_soft = np.clip((np.array(probs) + other_score*wt)/2, 0, 1)
    else:
        target_soft = np.array(probs)

   
    prediction = [1 if p>=.5 else 0 for p in target_soft]
    record_results = pd.DataFrame()
    if save_result:
        
        record_results["p_disc"] = prediction
        record_results["probs_0"] = 1-target_soft
        record_results["probs_1"] = target_soft 
        record_results.to_csv(args.log_dir + "HS-Brexit_results.tsv", sep='\t', header=False, index=False)
    if produce_result:
        ce = cross_entropy(soft, target_soft)
        f1_micro = f1_score(hard, prediction, average = "micro")

        print("ce is {}, and f1 is {}".format(ce, f1_micro))
    return record_results


soft = list(test.soft_label.apply(lambda x:x['1']))
hard = test.hard_label.tolist()
probs= result.probs.tolist()
agg = [sum(a)/len(a) for a in test.Aggressive.tolist()]
off = [sum(a)/len(a) for a in test.Offensive.tolist()]

df = compute_reg_score(soft, hard, probs, agg, off, 1, True, True)

