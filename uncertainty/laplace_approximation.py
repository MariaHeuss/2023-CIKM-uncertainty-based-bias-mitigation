from argparse import ArgumentParser
import numpy  as np
import torch
from backpack import backpack, extend
from backpack.extensions import BatchGrad

from transformers import BertModel, BertTokenizer

class PartialBert(torch.nn.Module):
    def __init__(self, bert: BertModel):
        super(PartialBert, self).__init__()
        self._bert = bert
        self._embedding_size = self._bert.config.hidden_size
        self.m = torch.nn.ReLU()

    def forward(self,encoded_qd) -> torch.Tensor:
        query_document_encoded = self._bert(**encoded_qd)
        out_features = query_document_encoded[0][:,0,:]
        return self.m(out_features)

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='bert-base-uncased')
    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--saved_model', type=str, )
    parser.add_argument('--saved_head', type=str, )

    return parser.parse_args()
if __name__ == '__main__':
    args = get_args()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    _bert = BertModel.from_pretrained('bert-base-uncased')
    llFF = torch.nn.Sequential(torch.nn.Linear(768, 2))

    #load learned model weights from desired checkpoint
    if args.saved_model:
        _bert_state_dict = torch.load(args.saved_model)
        _bert.load_state_dict(_bert_state_dict)
    PartialBert = PartialBert(_bert)

    #load head checkpoint to keep it separate from BERT. This is not necessary if you implement the approximation yourself, 
    #but it is necessary if you use the Backpack library.
    if args.saved_head:
        head_dict = torch.load(args.saved_head)
        llFF.load_state_dict(head_dict)
    

    #we need to get the expected FIM of a loss function so we approximate it with sampled data.
    q_doc = ['why is the sky blue?' + tokenizer.sep_token + 'due to the rayleigh scattering',
              'why is the sky blue?' + tokenizer.sep_token + 'soccer is a common outdoor activity played across the world',
              'what is a corporation?'+ tokenizer.sep_token + 'a corporation is a company or group of people authorized to act as a single entity and recognized as such in law.',
              'what is a corporation?'+ tokenizer.sep_token + 'if stock is issued, the corporation will usually be governed by its shareholder']
    target = torch.tensor([0,1,0,1])
    q_doc_encoded = tokenizer(q_doc, return_tensors='pt', max_length=512, padding=True, truncation=True)
        
    #extend the layers to support the backpack library
    llFF = extend(llFF)

    outputs = PartialBert(q_doc_encoded)
    scores = llFF(outputs)
    loss = torch.nn.CrossEntropyLoss()
    loss = extend(loss)
    loss_val = loss(scores, target)
    
    #use backpack to get the per sample gradients to get the diagonal of the FIM
    fisher_diagonal = []
    with backpack(BatchGrad()):
        loss_val.backward()
    for param in llFF.parameters():
        if param.requires_grad:
            # square per-sample gradients to compute diagonal of Fisher
            fisher_diagonal_param = param.grad_batch.pow(2)
            mean_fisher_diagonal_param = fisher_diagonal_param.mean(dim=0)
            #Measuring uncertainty w.r.t. it being relevant (class 0)
            fisher_diagonal.append(fisher_diagonal_param[0])

    # Sample from inverse FIM diagonal and compute empirical predictive distribution
    n_samples = args.num_samples
    empirical_predictions = []
    llFF.eval()
    mean_params = [param.data for param in llFF.parameters()]
    with torch.no_grad():
        for _ in range(n_samples):
            sampled_params = []
            for param, fisher_diag_param in zip(mean_params, fisher_diagonal):
                var = 1.0 / (fisher_diag_param + 1e-5)  # add a small constant for numerical stability
                sampled_param = torch.normal(mean=param.data, std=var.sqrt())
                sampled_params.append(sampled_param)
            
            # Set the sampled parameters and compute predictions
            for param, sampled_param in zip(llFF.parameters(), sampled_params):
                param.data = sampled_param
            
            sampled_output = llFF(outputs)
            #take logsoftmax
            sampled_output = torch.nn.LogSoftmax(dim=-1)(sampled_output)
            empirical_predictions.append(sampled_output[:,0])

            #get the mean and variance of the empirical predictions for each sample
    empirical_predictions = torch.stack(empirical_predictions)
    empirical_mean = empirical_predictions.mean(dim=0)
    empirical_std = empirical_predictions.std(dim=0)
    print ('done')