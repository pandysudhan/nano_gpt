import torch


block_size = 8 # max context length for model
batch_size = 4
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(2665)

# load the text from file
with open("input.txt", 'r', encoding="utf-8") as f:
    text = f.read() 
    
chars = sorted(list(set(text)))   

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
    


