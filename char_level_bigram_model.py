import torch
import torch.nn as nn
from torch.nn import functional as F

block_size = 256  # max context length for model
batch_size = 64
learning_rate = 3e-4
max_iters = 5000
eval_interval = 500
eval_iters = 200
dropout = 0.2
n_embd = 384  # embedding dimesion of a token
n_head = 6
n_layers = 6


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)

# load the text from file
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(vocab_size)

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
    idxs = torch.randint(low=0, high=(len(data) - block_size), size=(batch_size,))
    x = torch.stack([data[idx : idx + block_size] for idx in idxs])
    y = torch.stack([data[idx + 1 : idx + block_size + 1] for idx in idxs])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_data_batch(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(
            n_embd, head_size, bias=False
        )  # the key output size could be other than head_size as well, it is not used in output, confirmed!
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # attention
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # transpose only last two dimension and normalize
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )  # the tril is already in buffer, if tril[i,j] = 0, replace with -inf, as we use softmax after this
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # attention and value
        out = wei @ v

        return out


class MultiHead(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.linear_forward = nn.Linear(
            n_heads * head_size, n_embd
        )  # n_embd = n_heads*head_size anyways, but just to make sure
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        x = self.linear_forward(x)
        out = self.dropout(x)

        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.feedforward = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(),
        )

    def forward(self, x):
        out = self.feedforward(x)
        return out


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.multihead = MultiHead(n_head, n_embd // n_head)
        self.feedforward = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)  # TODO: need to learn mode
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.multihead(self.ln1(x))
        out = x + self.feedforward(self.ln2(x))

        return out


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layers)])
        self.final_ln = nn.LayerNorm(n_embd)
        self.model_head = nn.Linear(n_embd, vocab_size)

        # TODO self._init_weights

    def forward(self, idx, targets=None):
        # x is B*T shape, B is batchsize and T = max_context (time)
        B, T = idx.shape
        token = self.token_embedding_table(idx)  # B*T*C C = n_embd
        pos = self.pos_embedding_table(torch.arange(T, device=device))
        x = token + pos
        x = self.blocks(x)
        x = self.final_ln(x)
        logits = self.model_head(x)

        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, token_batch, max_token_generation):
        for _ in range(max_token_generation):
            token_batch_concated = token_batch[
                :, -block_size:
            ]  # only last "context length" token are taken
            logits, loss = self(token_batch_concated)
            last_logits = logits[:, -1, :]
            probs = F.softmax(last_logits, dim=-1)
            predicted_token = torch.multinomial(probs, num_samples=1)
            token_batch = torch.cat(
                (token_batch, predicted_token), dim=-1
            )  # dim = 1/-1 gives same result, as the time dimesion is 1 here, confirmed!

        return token_batch


# training part
m = BigramLanguageModel(vocab_size)
m.to(device)

# num of parameters in model
print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")


optimzer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:

        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    x, y = get_data_batch("train")

    logits, loss = m(x, y)
    optimzer.zero_grad(set_to_none=True)
    loss.backward()
    optimzer.step()

# generation
context = torch.zeros((2, 1), dtype=torch.long, device=device)
# open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
open("more.txt", "w").write(
    decode(m.generate(token_batch=context, max_token_generation=500)[0].tolist())
)
print(decode(m.generate(token_batch=context, max_token_generation=500)[1].tolist()))
