with open("Projets\\FromZeroToHeroGPT\\input.txt", "r") as f:
    text=f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("Vocab size:", vocab_size)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# We tokenize here character per character to keep it simple, but other ways of encoding it are possible
# Trade-off between length sequence and vocab size

# let's now encode the entire text dataset and store it into a torch.Tensor
import torch # we use PyTorch: https://pytorch.org
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000]) # the 1000 characters we looked at earier will to the GPT look like this

#split data
n = int(0.9*len(data))
train = data[:n]
test = data[n:]

block_size = 8

x = train[:block_size]
y = train[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f'context={context} qui a pour target {target}')