#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn, optim
from torch.nn import functional as F
#######sequence process
import pickle
def wordProcess(text,average_len):
    data_index=[]
    data_temp=[]
    data_processed=[]
    data=[]
    data_out=[]
    for each in text:
        data_processed.append(jieba.cut(each))
    
    stopwords = [line.strip() for line in open('stop_words.txt',encoding='UTF-8').readlines()]
    for row in data_processed:
        for word in row:
            if word not in stopwords:
                data_out.append(word)
        data.append(data_out)
        data_out=[]    
    
    vocab_to_int={}
    with open("vocab_to_int.file", "rb") as f:
        vocab_to_int = pickle.load(f)
    
    for each in data:
        for word in each:
            if word in vocab_to_int:
                data_temp.append(vocab_to_int[word])
            else:
                word = 0
        data_index.append(data_temp)
        data_temp=[]
   
    for i in range(len(data_index)):
        if len(data_index[i])>average_len:
            while (len(data_index[I])-average_len>0):
                data_index[i].pop()
        if len(data_index[i])==average_len:
            data_index[i]=data_index[i]
        else:
            m=average_len-len(data_index[i])
            for j in range(m):
                data_index[i].append(2)
    for i in range(len(data_index)):
        data_index[i] = list(map(int,data_index[i]))
    data_index=  torch.LongTensor(data_index)
    return data_index


#___________Definition of hyperparameters_____
embed_size = 32
batch_size = 128
num_layers = 2
hidden_size = 32
output_size = 48
latent_dim = 10
categorical_dim = 10
temp=0.5
learning_rate = 1e-4
word_length = vocab_length + 1
#__________gpu & dataset loading__________
embedding_matrix = np.load("embedding_matrix.npy")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainset = TextDataSet("NTM_DMIE_Data.txt")
trainloader =  torch.utils.data.DataLoader(trainset,batch_size=128,shuffle=False,drop_last=True,num_workers=8)
#__________Neural Network Body___________
class NTM_DMIE(nn.Module):
    def __init__(self,embed_size,hidden_size,output_size,num_layers,latent_dim,categorical_dim,word_length,bidirectional=True):
        super(NTM_DMIE,self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.embedding_matrix = embedding_matrix
        self.bidirectional=bidirectional
        self.num_layers=num_layers
        self.output_size = output_size
        self.vocab_length = vocab_length 
        self.embedding= nn.Embedding.from_pretrained(torch.from_numpy(embedding_matrix))
        self.lstm=nn.LSTM(embed_size, hidden_size, num_layers,bidirectional=bidirectional)
        self.encode_fc = nn.Linear(hidden_size * 4,output_size)
        self.encode_fc2 = nn.Linear(output_size,latent_dim * categorical_dim)
        self.global_fc1 = nn.Linear(latent_dim * categorical_dim,10)
        self.global_fc2 = nn.Linear(10,1)
        self.local_fc1 = nn.Linear(latent_dim * categorical_dim,10)
        self.local_fc2 = nn.Linear(10,1)
        self.decode_fc = nn.Linear(latent_dim * categorical_dim,word_length)
        
        
    
    def encode(self,x):
        embeds = self.embedding(x)
        embeds = embeds.to(torch.float32)
        embeds=embeds.permute(1, 0, 2)
        h0 = torch.zeros(self.num_layers*2,batch_size,self.hidden_size).to(device)#2 for bidirectional
        c0 = torch.zeros(self.num_layers*2,batch_size,self.hidden_size).to(device)
        lstm_out,(final_hidden_state,final_cell_state) = self.lstm(embeds, (h0,c0))#final_hidden_state.size() = (1, batch_size, hidden_size)
        final_out = torch.cat((lstm_out[0], lstm_out[-1]), -1)
        out = self.encode_fc(final_out)
        out = self.encode_fc2(out)
        out = self.sigmoid(out)
        return embeds,out
        
    def gumbelTransfer(self,out):
        out = self.encode(x)
        out_dimTrans = out.view(out.size(0),latent_dim,categorical_dim)
        z = gumbel_softmax(out_dimTrans,temp)
        return z
        
    def globalDiscriminator(self,z):
        z = self.global_fc1(z)
        global_score = self.global_fc2(z)
        return global_score
        
    def localDiscriminator(self,z,w2v):
        local_feature = torch.cat(w2v,z)
        out = self.local_fc1(local_feature)
        local_score = self.local_fc2(out)
        return local_score
        
    def decode(self,z):
        renconst_z = self.decode_fc(z)
        return renconst_z
        
        
    def forward(self,x):
        embeds,out = self.encode(x)
        z = self.gumbelTransfer(out)
        global_score = self.globalDiscriminator(z)
        local_score = self.localDiscriminator(z)
        renconst_z = self.decode(z)
        return z,global_score,local_score,reconst_z

model=NTM_DMIE(embed_size,hidden_size,output_size,num_layers,
latent_dim,categorical_dim,word_length,bidirectional=True)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 3, gamma = 0.1, last_epoch=-1)#changing the learning_rate through training


# In[ ]:


#_______________Training_______________
# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x,x,qy):
    BCE = F.binary_cross_entropy(recon_x,x,size_average=False)
    #BCE = (-x*torch.log(recon_x+1e-10)).sum(1)
    log_qy = torch.log(qy+1e-20)
    g = Variable(torch.log(torch.Tensor([1.0/categorical_dim])).to(device))
    KLD = torch.sum(qy*(log_qy - g),dim=-1).mean()
    return BCE + KLD

num_epochs = 100
for epoch in range(num_epochs):
    scheduler.step()
    train_loss = 0
    model.train()
    for i,(original_text,fake_text) in enumerate(trainloader):
        bow_matrix = 
        original_text = wordProcess(original_text)
        fake_text = wordProcess(fake_text)
        original_z,orginal_global,original_local,original_reconst = model(original_text)
        fake_z,fake_global,fake_local,fake_reconst = model(fake_text)
        global_loss = original_global - fake_global
        local_loss = original_local - fake_local
        renconst_loss = loss_function(original_reconst,bow_matrix,z)
        loss = global_loss + renconst_loss + reconst_loss
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if (i+1) % 10 == 0:
            print("Epoch[{}/{}], Step [{}/{}], Loss: {:.4f}" .format(epoch+1, num_epochs, i+1, len(trainloader),train_loss/i))

