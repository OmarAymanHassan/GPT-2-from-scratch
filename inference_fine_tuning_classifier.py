from pathlib import Path
import sys

import tiktoken
import torch
import chainlit


from gpt import  GPTModel


BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,          # Embedding dimension
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}


device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer= tiktoken.get_encoding("gpt2")

def classify_review(text, model, tokenizer, device, max_length, pad_token =50256):
    model.eval()
    input_ids = tokenizer.encode(text)
    max_context_length = model.pos_emb.weight.shape[1]

    # take all inputs from 0 to min of max_length or max_context_length
    input_ids = input_ids[: min(max_length,max_context_length)]
    
    input_ids += [pad_token] * (max_length - len(input_ids))

    input_tensor =torch.tensor(input_ids, device=device).unsqueeze(0)


    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
        predicted_label = torch.argmax(logits, dim=-1).item()

    return "spam" if predicted_label == 1 else "not spam"
    



##############3 Run ####################3

model = GPTModel(BASE_CONFIG)
num_classes = 2
model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)

### Load the pre-trained classification weights

model.load_state_dict(torch.load("review_classifier.pth" , map_location=device))
model.to(device)
model.eval()

######## chainlit function #########

@chainlit.on_message
async def main(message: chainlit.Message):
    """
    The main Chainlit function.
    """
    user_input = message.content

    label = classify_review(user_input, model, tokenizer, device, max_length=120)

    await chainlit.Message(
        content=f"{label}",  # This returns the model response to the interface
    ).send()


