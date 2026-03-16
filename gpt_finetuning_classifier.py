from gpt import  GPTModel, create_dataloader_v1
import torch
import tiktoken
import torch.nn as nn
import pandas as pd
from gpt_training import generate_text_sample , text_to_token, token_ids_to_text


tokenizer = tiktoken.get_encoding("gpt2")

device = "cuda" if torch.cuda.is_available() else "cpu"

########################### 1- Create Dataset  with padding short sequences msgs ##############################
import torch 
from torch.utils.data import Dataset, DataLoader


class SpamDataset(Dataset):
    def __init__(self,csv_file, tokenizer, max_length=None, pad_token = 50256):
        # Read the csv file
        self.data = pd.read_csv(csv_file)

        # Pretokenize the msgs

        self.encoded_texts = [tokenizer.encode(msg) for msg in self.data['message']]
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        
        # Truncate the encoded texts to max_length
        else:
            self.max_length = max_length
            self.encoded_texts = [msg[:self.max_length] for msg in self.encoded_texts]

        # Pads the encoded texts to max_length

        self.encoded_texts = [ msg +[pad_token] * (self.max_length - len(msg)) for msg in self.encoded_texts]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data['label'][index]
        return (torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long))
    

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            if len(encoded_text) > max_length:
                max_length = len(encoded_text)
        return max_length



###########################################################################################################

# Create datasets

train_dataset = SpamDataset("data/train.csv",
                            tokenizer,
                            max_length=None,
                            )


val_dataset = SpamDataset("data/validation.csv",
                            tokenizer,
                            max_length = train_dataset.max_length,
                            )

test_dataset = SpamDataset("data/test.csv",
                            tokenizer,
                            max_length = train_dataset.max_length,
                            )




####################### 2- Create DataLoaders #######################


from torch.utils.data import DataLoader


batch_size = 8
num_workers = 0
torch.manual_seed(123)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)


######################################################################



############################ 3- Import the model #################################

BASE_CONFIG = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "drop_rate": 0.0,       # Dropout rate
    "qkv_bias": True        # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}


CHOOSE_MODEL = "gpt2-small (124M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
INPUT_PROMPT = "Every Efforts moves you"


model = GPTModel(BASE_CONFIG)

## Loading the pre-trained weights
model.load_state_dict(torch.load("gpt2-small-124M.pth"))



####################################################################################




################################# 4- Adding Classification Head ##############################


# freeze all layers

for param in model.parameters():
    param.requires_grad= False

# Replace the existing head with 2-classes head

torch.manual_seed(123)
num_classes =2

model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)



## Allow gradient for the last transformer layer and final norm layer

for param in model.trf_blocks[-1].parameters():
    param.requires_grad =True

for param in model.final_norm.parameters():
    param.requires_grad= True


##############################################################################################



################################## 5- Functions for calc loss and accuracy ############################


def calc_acc_loader(data_loader, model, device, num_batches= None):
    model.eval()
    correct_predictions, num_examples = 0 , 0

    # If there is no num_batches assigned, calc the loss for all of them at once
    if num_batches is None:
        num_batches = len(data_loader)

    # this condition for safety, if user enter num_batches not realistic, it took the len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i , (input, target) in enumerate(data_loader):
        if i < num_batches:
            input = input.to(device)
            target = target.to(device)

            with torch.no_grad():
                logits = model(input)[:,-1,:] # last token only
            probas = torch.softmax(logits, dim=-1)
            predicted_label = torch.argmax(probas, dim=-1)
            num_examples += predicted_label.shape[0]

            # Compare Prediction between Boolean Tensors
            correct_predictions += (
            (predicted_label == target).sum().item()
            )

        else:
            break
    return correct_predictions / num_examples



# calc loss using cross entropy

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:,-1,:]
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss



# Loss function Loader

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) ==0:
        return float("nan")

    elif num_batches is None:
        num_batches = len(data_loader)
    
    else:
        num_batches = min(num_batches , len(data_loader))

    for i , (input_batch, target_batch) in enumerate(data_loader):

        #print(f"Iteration : {i}")
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss

        else:
            break
    
    return total_loss / num_batches


##################################################################################



################################ 6- Training the model ############################

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model,device ,num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def train_classifier(model,train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen , global_step = 0 , -1

    model.to(device)

    for epoch in range(num_epochs):
        model.train()

        for input_batch,target_batch in train_loader:
            
            # remove any gradient accumulation
            optimizer.zero_grad()
            
            # calc loss
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            
            # Backprpagation
            loss.backward() 

            # update weights
            optimizer.step()

            examples_seen += input_batch.shape[0]
            global_step +=1


            # Evaluation step

            if global_step % eval_freq ==0 :
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)

                # Appending loss
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                print(f"Ep {epoch+1} (step {global_step:06d}) "
                      f"Train Loss : {train_loss:.3f}, "
                      f"Val Loss : {val_loss:.3f}")
                
        # Calculating accuracy
        train_acc = calc_acc_loader(train_loader, model, device, eval_iter)
        val_acc = calc_acc_loader(val_loader, model, device, eval_iter)

        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Training Accuracy: {train_acc*100:.2f}%")
        print(f"Validation Accuracy: {val_acc*100:.2f}%")

    return train_losses, val_losses, train_accs, val_accs, examples_seen



###########################################################################################


###################### 7. Starting the process ##########################


import time

start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
num_epochs = 5

train_losses, val_losses, train_accs, val_accs, example_seen = train_classifier(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq=50, eval_iter=5)

end_time = time.time()

exec_time_in_mins = (end_time - start_time) / 60
print(f"Training Completed in {exec_time_in_mins:.2f} minutes.")


