"""@package docstring
Documentation for this module.
 
This package is server code.
"""

from grpc_wrapper.server import create_server, BaseModel
import time
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

#tokenizer.add_special_tokens({"additional_special_tokens": ["<turn>", "<description>"]})
#electra.resize_token_embeddings(len(tokenizer))


class MTGRU(nn.Module):
    """Documentation for a class.
	
    This class makes a custom MTGRU based on an original MTGRU.
    """

    def __init__(self, d_model = 128, device=None):
        """The constructor of a MTGRU class."""
        super(MTGRU, self).__init__()
        torch.cuda.set_device(device)

        self.d_model = d_model
        self.tau1 = nn.Parameter(torch.tensor([1.0]),requires_grad=True)
        self.tau2 = nn.Parameter(torch.tensor([1.2]),requires_grad=True)
        self.grucell_1 = nn.GRUCell(768, self.d_model)
        self.grucell_2 = nn.GRUCell(768+self.d_model, self.d_model)
    
    def forward(self, encoder_outputs):
        """This method handles a custom forward computation."""

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
    """Documentation for a class.
	
    This class represent my model with a customed MTGRU.
    """

    def __init__(self, electra, d_model = 128, device = None):
        """The constructor of a Mymodel class."""
        super(Mymodel, self).__init__()
        torch.cuda.set_device(device)
        #self.bert = bert
        self.electra = electra
        self.mtgru = torch.jit.script(MTGRU(d_model, device = device))
        self.classifier = nn.Linear(d_model*2, 21)
    
       
    def forward(self, input_ids, token_num_list):
        """This method handles a custom forward computation."""
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


class WoodongModel:
    """Documentation for a class.
	
    This class is example to utilize GRPC wrapper.
    """
    def __init__(self):
        """The constructor of a WoodongModel class."""

        print("init")

    def run(self, name):
        """This method handles request with given name."""
        return {
            "sentence": "Hi, "+name
        }


class YourModel(BaseModel):
    """Documentation for a class.
	
    This class uses Mymodel with GRPC wrapper.
    """
    def __init__(self):
        """The constructor of a YourModel class."""

        #self.model = WoodongModel()
        self.my_model = Mymodel(electra, device = 0)
        self.my_model.to(device)
        checkpoint = load_checkpoint(default_directory)
        if not checkpoint:
            pass
        else:
            self.my_model.load_state_dict(checkpoint['state_dict']) 
            
        # while True:
        #    TEXT = [input('입력 : ')]
        #    test(TEXT)

    def send(self, input):
        """This method handles request with given input."""
        print(input)
        # input & output can be dictionary or array
        #output = self.model.run(input["name"])
        output = self.test([input["sentence"]])
        output = label_change(output)
        return {"output": str(output)}
        
    def test(self, TEXT):
        """This method run Mymodel instance with TEXT as input."""
        self.my_model.eval() 

        text = TEXT


        encoded_list = [tokenizer.encode(t, add_special_tokens=True) for t in text]
        padded_list =  [e + [0] * (512-len(e)) for e in encoded_list]
        token_num_list_ = []
        for enc_list in encoded_list:
            token_num_list_.append(len(enc_list))
        token_num_list = torch.tensor(token_num_list_)   

        sample = torch.tensor(padded_list)
        sample = sample.to(device)
        outputs = self.my_model.forward(sample, token_num_list)
        logits = outputs


        pred = torch.argmax(F.softmax(logits, dim=1), dim=1)
        pred = str(pred.item())
        #print(pred)
        return pred
        

def load_checkpoint(directory, filename='save_20201120_intention.tar.gz'):
    """Documentation for a function.
 
    This function loads checkpoint with filename.
    """

    model_filename = os.path.join(directory, filename)
    if os.path.exists(model_filename):
        print("=> loading checkpoint")
        state = torch.load(model_filename)
        return state
    else:
        print("=> checkpoint does not exist.")
        return None

def label_change(input_label):
    """Documentation for a function.
 
    This function corrects model side class number and service side class number.
    """
    labels = [20, 2, 18, 11, 3, 1, 13, 17, 8, 16, 9, 19, 14, 7, 5, 6, 15, 10, 4, 12, 0]
    return labels[int(input_label)]


def run():
    """Documentation for a function.
 
    This function start server with GRPC wrapper.
    """
    model = YourModel()
    server = create_server(model, ip="[::]", port=50051, max_workers=5)
    server.start()
    try:
        while True:
            time.sleep(60 * 60 * 24)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ =="__main__":
    """Documentation for a function.
 
    Main function of this server.py.
    """

    keywords = ['where', 'video', 'mail', 'schedule', 'address', 'know', 'weather', 'reserv', 'flight', 'shop',
				'restaurant', 'wonder', 'door', 'news', 'movie', 'stock', 'summar', 'depress', 'sport', 'book']
    run()
