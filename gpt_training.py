from gpt import  GPTModel, create_dataloader_v1
import torch
import tiktoken
import torch.nn as nn

GPT_CONFIG_124M = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 256,  # Context length
    "emb_dim": 768,          # Embedding dimension
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False        # Query-Key-Value bias
}



# Tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# text to token
def text_to_token(tokenizer, text):

    encoded = tokenizer.encode(text , allowed_special={"<|endoftext|>"})
    # adding batch dimenstion
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    return encoded_tensor


# token to text
def token_ids_to_text(tokenizer, token_ids):
    # Flatten the tensor by removing the batch dimenstion
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


# reading data
file_path  = "training-text.txt"

with open(file_path, "r") as f:
    text_data = f.read()



# Train test split
total_caracters = len(text_data)

train_ratio = 0.90
split_idx = int(total_caracters * train_ratio)

train_data = text_data[:split_idx]
val_data = text_data[split_idx:]



# Creating Data Loader
import torch 
from torch.utils.data import Dataset,DataLoader



train_loader = create_dataloader_v1(train_data ,batch_size=2 , max_length = GPT_CONFIG_124M["context_length"], stride = GPT_CONFIG_124M["context_length"], drop_last=True, shuffle=True)
val_loader = create_dataloader_v1(val_data ,batch_size=2 , max_length = GPT_CONFIG_124M["context_length"], stride = GPT_CONFIG_124M["context_length"], shuffle=False, drop_last=True)


# calc loos for batch
def calc_loss_batch(input , targets, model, device):

    input = input.to(device).long()
    #print(f"Input : {input}")
    targets = targets.to(device)

    logits = model(input)

    logits_flatten = logits.flatten(0,1)
    targets_flatten = targets.flatten()

    loss = torch.nn.functional.cross_entropy(logits_flatten, targets_flatten)

    return loss


# calc loss across loader

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float('nan')
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i , (x,y) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(x,y, model, device)
            total_loss += loss.item()
        else:
            break
    avg_loss = total_loss / num_batches
    return avg_loss



# model evaluation

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
        val_loss = calc_loss_loader(val_loader,model, device, eval_iter)
    model.train()
    return train_loss, val_loss

# generate sample 
def generate_text_sample(model, idx, max_tokens, context_size):

    for _ in range(max_tokens):

        # get the last max_tokens with all batches, columns

        crpped_context_size = idx[: , -context_size: ]

        # generate logits based on the last tokens

        with torch.no_grad():
            logits = model(crpped_context_size)

        # getting last token


        logits = logits[: , -1 , :]
        
        # getting proba

        proba = torch.softmax(logits, dim = -1)

        next_idx = torch.argmax(proba , dim = -1, keepdim=True)

        # adding last token_id to the existing context

        idx = torch.cat((idx, next_idx), dim = 1)

    return idx


def generate_and_print_simple(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token(tokenizer, start_context).to(device)
    with torch.no_grad():
        tokens_id = generate_text_sample(model, encoded, 50, context_size)
    
    decoded_text = token_ids_to_text(tokenizer, tokens_id)
    print(decoded_text.replace("\n" , " "))
    model.train()



# train model 

def train_model_simple(train_loader, val_loader, optimizer,model, device,num_epochs, eval_freq, eval_iter, start_context, tokenizer ):
    train_losses, val_losses, track_tokens_losses =[], [], [] 
    tokens_seen, global_step = 0, -1


    for epoch in range(num_epochs):
        model.train()
        for x , y in train_loader:

            # prevent gradient accumulation
            optimizer.zero_grad()

            loss = calc_loss_batch(x, y , model, device)

            # Backpropagation
            loss.backward()

            # Update weights

            optimizer.step()

            tokens_seen += x.numel()
            global_step +=1

            if global_step % eval_freq ==0 :
              
              train_loss, eval_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)

              train_losses.append(train_loss)
              val_losses.append(eval_loss)
              track_tokens_losses.append(tokens_seen)
              print(f"Ep {epoch +1} (step {global_step:06d}) " ,
                  f"Train Loss : {train_loss:.3f}",
                  f"Validation Loss :{eval_loss:.3f}"

                  
                  )
        generate_and_print_simple(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_tokens_losses

            

def save_model(model,optimizer):
    print("\nSave the Model & Optimizer Parameters")

    torch.save(
        {
            "model_state_dict" : model.state_dict(),
            "optimizer_state_dict" : optimizer.state_dict()
        },
        "model_and_optimimzer.pth"
    )


def main():
    torch.manual_seed(123)
    device ="cuda" if torch.cuda.is_available() else "cpu"
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay =0.1)

    num_epochs = 10
    train_losses, val_losses ,tokens_seen = train_model_simple(train_loader, val_loader, optimizer, model, device, num_epochs, eval_freq=5, eval_iter=5 , tokenizer=tokenizer, start_context="Every Efforts Moves you")
    save_model(model, optimizer)

if __name__ == "__main__":
    main()

