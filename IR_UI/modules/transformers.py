import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import numpy as np
import pandas as pd
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_s1 = pd.read_csv("tweets.csv")

def find_senti(score):
  scr = int(score)
  if scr < 0:
    return "Negative"
  elif scr == 0:
    return "Neutral"
   else:
    return "Positive"

pt_mdl = 'bert-base-cased'
tknzr = BertTokenizer.from_pretrained(pt_mdl)

class SentimentClassifier(nn.Module):

  def __init__(self, n_classes):
    super(SentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(pt_mdl)
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    
  def forward(self, input_ids, attention_mask):
     _, pooled_output = self.bert(
     input_ids=input_ids,
     attention_mask=attention_mask
    )
     output = self.drop(pooled_output)
     return self.out(output 


# Training Loop
loss_function=nn.MSELoss()
optimizer= torch.optim.Adagrad(seq_model.parameters(), lr=0.001)
trainvaltest_loaders = {'train': train_loader, 'val': test_loader}
loss_list=[]
val_loss=10000
for epoch in range(0, num_epochs):
    running_loss=0.0
    for phase in ['train','val']:
        start_time = timeit.default_timer()

        if phase == 'train':
            seq_model.train()
            optimizer.zero_grad()
        else:
            seq_model.eval()

        for i,data in enumerate(tqdm(trainvaltest_loaders[phase])):
            inputs,preds=data
            inputs,preds=inputs.float().cuda(),preds.float().cuda()
            preds=preds.reshape([preds.shape[0],1])
            inputs = Variable(inputs, requires_grad=True)
            preds= Variable(preds,requires_grad=False)
            if(phase == 'train'):  
                output,memory=seq_model(inputs)
                single_loss=loss_function(output,preds)
            else:
                with torch.no_grad():
                    output,val_memory=seq_model(inputs)
                    single_loss=loss_function(output,preds)
            if phase == 'train':
                seq_model.backward(single_loss,output,preds,inputs)
                optimizer.step()
            else:
                running_loss += single_loss.item()
    if(((running_loss)/len(trainvaltest_loaders['val']))<val_loss):
        val_loss=(running_loss/len(trainvaltest_loaders['val']))
        torch.save(seq_model.state_dict(),r'C:\Users\Sujith\Sentiment_Analysis\senti.pth')
        print('Saving Model at Epoch: ',epoch)
    else:
        pass
    print('Epoch :',epoch,' Loss : ',(running_loss/len(trainvaltest_loaders['val'])))
    loss_list.append(single_loss.item())

 