import torch
import torch.nn as nn
from torch.nn import functional as F

block_size = 8 # max context length for model
batch_size = 4
n_embd = 32
learning_rate = 1e-2
max_iter = 1000
eval_iters = 100
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(2665)

# load the text from file
with open("input.txt", 'r', encoding="utf-8") as f:
    text = f.read() 
    
chars = sorted(list(set(text)))   
vocab_size = len(chars)


# encode/decode data
char_to_token = {}
token_to_char = {}
for i, char in enumerate(chars):
    char_to_token[char] = i
    token_to_char[i] = char
    
def encode(chars):
    encoded = []
    for char in chars:
        encoded.append(char_to_token[char])
    return encoded

def decode(tokens):
    decoded = []
    for token in tokens:
        decoded.append(token_to_char[token])
    return "".join(decoded)

# print(encode("sudhan"))
# print(decode(encode("sudhan")))
# encoder/decoder works


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
test_data = data[n:]

def get_data_batch(split):
    data = train_data if split == "train" else test_data
    idxs = torch.randint(low = 0, high=(len(chars)- block_size), size=(batch_size,))
    x = torch.stack([data[idx: idx + block_size] for idx in idxs])
    y = torch.stack([data[idx + 1 : idx + block_size + 1] for idx in idxs])
    x, y = x.to(device), y.to(device)
    return x, y
    
@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_data_batch(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, x, targets = None):
        # x is B*T shape, B is batchsize and T = max_context (time)
        logits = self.token_embedding_table(x) #B*T*C C = n_embd
        
        if targets == None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
            
            
    def generate(self, token_batch, max_token_generation):
        for _ in range(max_token_generation):
            logits, loss  = self(token_batch)
            last_logits = logits[:, -1, :]
            probs = F.softmax(last_logits, dim=-1)
            predicted_tokens = torch.multinomial(probs, num_samples=1)
            token_batch = torch.cat((token_batch, predicted_tokens), dim=-1) # dim = 1/-1 gives same result, as the time dimesion is 1 here
            
        return token_batch
            
            
  
# training part          
m = BigramLanguageModel(vocab_size)
m.to(device)

optimzer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iter):
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
    x, y = get_data_batch("train")
    
    logits, loss = m(x,y)
    optimzer.zero_grad(set_to_none=True)
    loss.backward()
    optimzer.step()
        
# generation 
context = torch.zeros((1,1), dtype= torch.long, device=device)
print(decode(m.generate(token_batch=context, max_token_generation=50)[0].tolist()))

        
        
    