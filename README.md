This is a char level bigram model trained on all of Shakespeare's writing. 

Number of parameters: 10.788929 M 

model specifications:
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

sample output can be found [here](./more.md)
