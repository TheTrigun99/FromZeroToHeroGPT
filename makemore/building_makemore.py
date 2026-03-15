import numpy as np
import torch
import torch.nn as nn

words = open('names.txt','r').read().splitlines()
b = {}
#for w in words:
#    chs = ['<S>'] + list(w)+ ['<E>'] #we have more information (starts ,ends)
#    for ch1,ch2 in zip(chs,chs[1:]):
#        bigram = (ch1,ch2)
#        b[bigram] = b.get(bigram,0)+1 #adding words (count dictionnary)
import torch
N = torch.zeros( (27,27),dtype=torch.int32) #26 letters from the alphabet + 2 special
chars = sorted(list(set("".join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0

for w in words:
    chs = ['.'] + list(w)+ ['.'] #we have more information (starts ,ends)
    for ch1,ch2 in zip(chs,chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1,ix2]+=1

itos = {i:s for s,i in stoi.items()}
import matplotlib.pyplot as plt
#plt.imshow(N)
#plt.show()
p = N[0].float()
p = p / p.sum()
g = torch.Generator().manual_seed(2147483647)
#p = torch.rand(3, generator=g)
#ix = torch.multinomial(p,num_samples=1, replacement=True, generator=g).item()
P = (N+1).float()
P /= P.sum(1,keepdim=True)
for i in range(10):
    ix = 0
    name = ''
    while True:
        p = P[ix]
        ix = torch.multinomial(p,num_samples=1, replacement=True, generator=g).item()

        if ix == 0:
            break
        name = name + itos[ix]
 
log_likehood = 0
n = 0
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1,ch2 in zip(chs,chs[1:]):
        ix1,ix2 = stoi[ch1], stoi[ch2]
        prob = P[ix1, ix2]
        log_likehood += torch.log(prob)
        n += 1
nll = -log_likehood
print(f'{nll/n}')


