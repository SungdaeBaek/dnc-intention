import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import ElectraModel, ElectraTokenizer
import torch.nn.functional as F
import os

os.environ["CUDA_VISIBLE_DEVICES"]= "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

default_directory = './save'

electra = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

tokenizer.add_special_tokens({"additional_special_tokens": ["<turn>", "<description>"]})
electra.resize_token_embeddings(len(tokenizer))


class MTGRU(nn.Module):

    def __init__(self, d_model = 128, device=None):
        super(MTGRU, self).__init__()
        torch.cuda.set_device(device)

        self.d_model = d_model
        self.tau1 = nn.Parameter(torch.tensor([1.0]),requires_grad=True)
        self.tau2 = nn.Parameter(torch.tensor([1.2]),requires_grad=True)
        self.grucell_1 = nn.GRUCell(768, self.d_model)
        self.grucell_2 = nn.GRUCell(768+self.d_model, self.d_model)
    
    def forward(self, encoder_outputs):

        batch_size = encoder_outputs.size(0)

        h1 = torch.zeros(batch_size, self.d_model).cuda()
        h2 = torch.zeros(batch_size, self.d_model).cuda()

        gru_output_1 = []
        gru_output_2 = []

        for i in range(encoder_outputs.size(1)):
            h1_ = self.grucell_1(encoder_outputs[:,i], h1)
            h1 = (1 - 1/self.tau1) * h1 + 1/self.tau1 * h1_
            h2_ = self.grucell_2(torch.cat((encoder_outputs[:,i], h1),1), h2)
            h2 = (1 - 1/self.tau2) * h2 + 1/self.tau2 * h2_
            gru_output_1.append(h1)
            gru_output_2.append(h2)
        
        gru_output_1 = torch.stack(gru_output_1).transpose(0,1) # (batch_size, seq_len, gru_hidden_size)
        gru_output_2 = torch.stack(gru_output_2).transpose(0,1) # (batch_size, seq_len, gru_hidden_size)
        gru_output = torch.cat((gru_output_1, gru_output_2), 2)
        gru_output = F.dropout(gru_output, p=0.5)

        return gru_output




class Mymodel(nn.Module):

    def __init__(self, electra, d_model = 128, device = None):
        super(Mymodel, self).__init__()
        torch.cuda.set_device(device)
        #self.bert = bert
        self.electra = electra
        self.mtgru = torch.jit.script(MTGRU(d_model, device = device))
        self.classifier = nn.Linear(d_model*2, 21)
    
       
    def forward(self, input_ids, token_num_list):
        batch_size = input_ids.size(0)
        input_ids_mask = input_ids != 0

        encoder_outputs_ = self.electra(input_ids, attention_mask=input_ids_mask.long()) # 1 for not MASKED
        encoder_outputs = encoder_outputs_[0]
        
        gru_output = self.mtgru(encoder_outputs)

        out_list = []
        for bn in range(batch_size):
            out_ = gru_output[bn][token_num_list[bn]-1]
            out_list.append(out_)

        gru_output_ = torch.stack(out_list)

        output = self.classifier(gru_output_)
        #loss = criterion(output, labels)

        return output


model = Mymodel(electra, device = 0)
model.to(device)


def test(TEXT):
    model.eval() 
    
    text = TEXT
       

    encoded_list = [tokenizer.encode(t, add_special_tokens=True) for t in text]
    padded_list =  [e + [0] * (512-len(e)) for e in encoded_list]
    token_num_list_ = []
    for enc_list in encoded_list:
        token_num_list_.append(len(enc_list))
    token_num_list = torch.tensor(token_num_list_)   

    sample = torch.tensor(padded_list)
    sample = sample.to(device)
    outputs = model.forward(sample, token_num_list)
    logits = outputs
   

    pred = torch.argmax(F.softmax(logits, dim=1), dim=1)
    pred = str(pred.item())
    print(pred)
             


def load_checkpoint(directory, filename='save0.tar.gz'):

    model_filename = os.path.join(directory, filename)
    if os.path.exists(model_filename):
        print("=> loading checkpoint")
        state = torch.load(model_filename)
        return state
    else:
        print("=> checkpoint does not exist.")
        return None



if __name__ == "__main__":
    checkpoint = load_checkpoint(default_directory)
    if not checkpoint:
        pass
    else:
        model.load_state_dict(checkpoint['state_dict']) 
        
    while True:
        TEXT = [input('입력 : ')]
        test(TEXT)
