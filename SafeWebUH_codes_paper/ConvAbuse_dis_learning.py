from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
import time
import random
import json
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import argparse
from transformers import BertTokenizer
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.model_selection import train_test_split



start = time.time()


# Process the data: Load and convert to pickle


train = pd.read_json(path + "data_practicephase_cleardev/data_practicephase_cleardev/ConvAbuse_dataset/ConvAbuse_train.json", orient='index')
dev = pd.read_json(path + "data_practicephase_cleardev/data_practicephase_cleardev/ConvAbuse_dataset/ConvAbuse_dev.json", orient='index')
test = pd.read_json(path + "data_post-competition/data_post-competition/ConvAbuse_dataset/ConvAbuse_test.json", orient='index')


# Data Contains both user and bot text. We will use the user text only
def convert_to_user_texts(x):
    t = json.loads(x)
    return t["prev_user"] + " " + t["user"]




train["user_text_only"] = train.text.apply(lambda x:convert_to_user_texts(x))
dev["user_text_only"] = dev.text.apply(lambda x:convert_to_user_texts(x))
test["user_text_only"] = test.text.apply(lambda x:convert_to_user_texts(x))


def tokenization_for_BERT(df, path="/path/to/data/Convabuse/Data/", filename="put_the_filename_here", saveit="No"):


    if "tokenized" not in df.columns:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        df["tokenized"] = df["user_text_only"].apply(lambda sent:tokenizer.encode(sent, add_special_tokens=True,
                                                                                max_length=512, truncation=True,
                                                                                padding='max_length',
                                                                                return_attention_mask=False))


        if saveit!="No":
            df.to_pickle(path + filename + ".pkl")
       


    return df


class BertClassifier(nn.Module):


    def __init__(self, hidden_size=50, dropout=0):
        super(BertClassifier, self).__init__()
        D_in, H, D_out = 768, hidden_size, 1
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc1 = nn.Linear(D_in, H)
        self.fc2 = nn.Linear(H, D_out)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
       
    def forward(self, input_ids, attention_mask):


        # Feed input to BERT
        bert_cls_outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)[0][:, 0, :]
       


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
    #print("=========+++++++++++  Best Epoch: ", best_epoch)




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
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
            loss = loss_fn(logits.view(logits.shape[0],), soft)
            total_loss += loss.item()
           
        all_logits.append(logits)
        all_soft.append(soft)
   
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
    P = [0 if t<.5 else 1 for t in P]
    f1 = f1_score(T, P, average='micro')


    return ce, f1*100, avg_loss, df


set_seed(42)


parser = argparse.ArgumentParser(description='BERT model arguments')


parser.add_argument("--data_dir",
                    type=str,
                    default="/path/to/data/Convabuse/Data/",
                     help="Input data path.")
parser.add_argument("--log_dir",
                     type=str,
                     default="/path/to/data/Convabuse/Result/SafeWebUH/",
                     help="Store result path.")
parser.add_argument("--model_path_dir",
                     type=str,
                     default="/path/to/data/Convabuse/Result/",
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


#train = pd.read_pickle(args.data_dir + "Convabuse_train.pkl")
train = tokenization_for_BERT(train, path=args.data_dir, filename= "Convabuse_train.pkl" , saveit="Yes")
train["SOFT"] = train.soft_label.apply(lambda x:x['1'])
print(train.shape)


#dev = pd.read_pickle(args.data_dir + "Convabuse_dev.pkl")
dev = tokenization_for_BERT(dev, path=args.data_dir, filename= "Convabuse_dev.pkl" , saveit="Yes")
dev["SOFT"] = dev.soft_label.apply(lambda x:x['1'])


train = pd.concat([train, dev], axis=0).sample(frac=args.fast_track)
train.reset_index(inplace=True, drop=True)


#test = pd.read_pickle(args.data_dir + "Convabuse_test.pkl")
test = tokenization_for_BERT(test, path=args.data_dir, filename= "Convabuse_test.pkl" , saveit="Yes")
test["SOFT"] = test.soft_label.apply(lambda x:x['1'])




tok_tr, mask_tr, lab_tr, soft_tr = prepare_train_and_valid(train)
train_dataloader = create_dataloader(tok_tr, lab_tr, mask_tr, soft_tr, batch_size=args.batch_size)


tok_ts, mask_ts, lab_ts, soft_ts = prepare_train_and_valid(test)
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




### following lines of codes are for metadata only
## If you are using metadata


def find_freq(L):
    freq_dic=dict()
    uniqs = list(set(L))
    for u in uniqs:
        freq_dic[u] = L.count(u)
    return freq_dic
def find_avg(L):
    return(sum(L)/len(L))


def convert_to_user_texts(x):
    t = json.loads(x)
    return t["prev_user"] + " " + t["user"]
def convert_to_tsystem(x):
    return find_avg([int(i) for i in x["other_annotations"]["target.system"].split(',')])
   
def convert_to_explicitness(x):
    return find_avg([int(i) for i in x["other_annotations"]["explicit"].split(',')])

def compute_reg_score(soft, hard, probs, Tsystem, Explicit, wt, save_result=False, produce_result=True):
    
    if args.use_metadata=="yes":
        other_score = np.array(Tsystem)*reg.coef_[0] + np.array(Explicit)*reg.coef_[1] - reg.intercept_
        target_soft = np.clip((np.array(probs) + other_score*wt)/2, 0, 1)
    else:
        target_soft = np.array(probs)

   
    prediction = [1 if p>=.5 else 0 for p in target_soft]
    record_results = pd.DataFrame()
    if save_result:
        
        record_results["p_disc"] = prediction
        record_results["probs_0"] = 1-target_soft
        record_results["probs_1"] = target_soft 
        record_results.to_csv(args.log_dir + "ConvAbuse_results.tsv", sep='\t', header=False, index=False)
    if produce_result:
        ce = cross_entropy(soft, target_soft)
        f1_micro = f1_score(hard, prediction, average = "micro")

        print("ce is {}, and f1 is {}".format(ce, f1_micro))
    return record_results

def compute_aggregated_score(prediction_df, sys_wt, exp_wt):
    Target = prediction_df["target"].tolist()
    Prediction = prediction_df["probs"].tolist()
    Tsys = prediction_df["Tsystem"].tolist()
    Exp = prediction_df["Explicit"].tolist()
    computed_prediction = []
    for P, Ts, Ex in zip(Prediction, Tsys, Exp):
        computed_prediction.append((P + Ts*sys_wt + Ex*exp_wt)/(1+sys_wt + exp_wt))
   
    Target_hard = [0 if t<.5 else 1 for t in Target]
    Pred_hard = [0 if t<.5 else 1 for t in computed_prediction]  
    f1 = f1_score(Target_hard, Pred_hard,  average='micro')
    ce = cross_entropy(Target, computed_prediction)
    ## For saving
    probs_1 = []
    probs_0 = []


    for m in computed_prediction:
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
    record_results["p_disc"] = Pred_hard
    record_results["probs_0"] = probs_0
    record_results["probs_1"] = probs_1
    # record_results["T_disc"] = T_disc
    # record_results["T"] = T
    record_results.to_csv(args.log_dir + "convabuse_test.tsv", sep='\t', header=False, index=False)
    return f1, ce
   
y = list(train.soft_label.apply(lambda x:x['1']))
X_other = []
for a, o in zip(train.Tsystem.tolist(), train.Explicit.tolist()):
    X_other.append([a,o])
reg = LinearRegression().fit(X_other, y)

def compute_reg_score(soft, hard, probs, Tsystem, Explicit, wt, save_result=False, produce_result=True):
    
    if use_metadata=="yes":
        other_score = np.array(Tsystem)*reg.coef_[0] + np.array(Explicit)*reg.coef_[1] - reg.intercept_
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

if args.use_metadata=="yes":
    test = pd.read_json(args.data_dir + "data_post-competition/data_post-competition/ConvAbuse_dataset/ConvAbuse_test.json", orient='index')
    test["Tsystem"] = test.other_info.apply(lambda x:convert_to_tsystem(x))
    test["Explicit"] = test.other_info.apply(lambda x:convert_to_explicitness(x))
    test["user_text_only"] = test.text.apply(lambda x:convert_to_user_texts(x))
    probs = pd.read_pickle(args.log_dir + "temp/probs.pkl")
    df = compute_reg_score( probs.soft_lab.tolist(), test.hard_label.values, probs["probs"].values, test.Tsystem.values, test.Explicit.values, 1)

