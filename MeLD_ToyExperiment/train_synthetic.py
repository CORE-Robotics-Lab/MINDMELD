
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.optimizer import Optimizer
from argparse import Namespace
import typing as t
from MINDMELD.MeLD_ToyExperiment.MeLD_model import MapModel
from torch.autograd import Variable
import os

import matplotlib.pyplot as plt
import numpy as np


def train(model: torch.nn.Module,
          dataloader: DataLoader,
          optimizer: Optimizer,
          training_args: Namespace):
    """
    Assuming a dataloader, this loads data in and trains a model. assumes model makes 3 predictions,
    sums all 3 losses evenly weighted.
    Args:
        model: model to train
        dataloader: dataloader for the dataset to train over
        opt: optimizer to use
        training_args: other training arguments. needs .epochs and .cuda
    Returns:
    """
    allLoss=0
    all_w_loss=0
    all_o_loss=0
    celoss = torch.nn.CrossEntropyLoss()
    mseloss = torch.nn.MSELoss()
    if training_args.cuda:
        celoss.to('cuda')
        mseloss.to('cuda')

    for index, sample in enumerate(dataloader):
        labels, w_ind,diff,gt,l_to_map = sample
        if training_args.cuda:
            labels = labels.to('cuda')
            diff = diff.cuda()
        #w_set=all_w[w_ind.long(),:]
        diff_hat,w_hat = model(labels,w_ind,batch_size=args.batch_size)
        diff_loss = mseloss(diff_hat, diff.unsqueeze(1))
        ws=model.w

        w_loss = mseloss(w_hat, ws[w_ind.long(),:])
        total_loss = diff_loss + .01*w_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        allLoss+=total_loss
        all_w_loss+=w_loss
        all_o_loss+=diff_loss

    return model, optimizer,allLoss,all_w_loss,all_o_loss


def evaluate(dataloader,eval_args: Namespace, model: t.Union[torch.nn.Module, t.List]):
    loss=0
    all_w_loss=0
    all_o_loss=0
    celoss = torch.nn.CrossEntropyLoss()
    mseloss = torch.nn.MSELoss()
    if eval_args.cuda:
        celoss.to('cuda')
        mseloss.to('cuda')
    i=0
    percent_better=0
    for index, sample in enumerate(dataloader):
        labels, w_ind,diff,gt,l_to_map = sample
        if eval_args.cuda:
            labels = labels.to('cuda')
            diff = diff.cuda()
            gt=gt.cuda()
            l_to_map=l_to_map.cuda()
        #w_set = all_w[w_ind.long(), :]
        #print("ALL W",all_w)
        ws=model.w
        diff_hat, w_hat = model(labels,w_ind,batch_size=1)
        diff_loss = mseloss(diff_hat, diff.unsqueeze(1))
        new_label=diff_hat+l_to_map
        new_error=abs(new_label-gt)
        percent_better+=(abs(diff.unsqueeze(1))-new_error)/abs(diff.unsqueeze(1))
        w_set=ws[w_ind.long(), :]
        w_loss = mseloss(w_hat, w_set)
        total_loss = diff_loss + w_loss
        loss+=total_loss
        all_w_loss+=w_loss
        all_o_loss+=diff_loss
        i+=1
    print(ws)
    print("all_loss",loss/i)
    print("o_loss",all_o_loss/i)
    print("all_w_loss",all_w_loss/i)

    print("PERCENT Better",percent_better/(i*1.0))

    return loss,all_w_loss,all_o_loss




def train_eval_loop(dataloader,
                    eval_dataloader,
                    model: t.Union[torch.nn.Module, t.List],
                    optimizer: t.Union[Optimizer, t.List],
                    training_args: Namespace) -> torch.nn.Module:
    """
    Assuming a dataloader (future?), this loads data in and trains a model. assumes model makes 3 predictions,
    sums all 3 losses evenly weighted.
    Args:
        dataloader: dataloader for training data
        eval_dataloader: dataloader for evaluation
        model: model for three tasks or list of 3 models for 3 tasks [type, target, number] order
        optimizer: the optimizer that is used for the full model or 3 separate optimizers in [type, target, number] order
        training_args: other training arguments. needs .epochs and .device
    Returns:
    """

    for e in range(training_args.epochs):
        model, optimizer, loss_val,w_loss,o_loss = train(model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            training_args=training_args)


        eval_loss,eval_w_loss,eval_o_loss = evaluate(dataloader=eval_dataloader, eval_args=training_args,model=model)

        if (e %20)==0:
            torch.save(model.state_dict(),os.path.join('Models','mapper'))

    return model



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-e', '--epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('-n', '--num_samples', type=int, default=1000, help='number of samples')
    parser.add_argument('-cuda', action='store_true', default=True, help='use cuda?')
    parser.add_argument('-batch','--batch_size', type=int,default=128, help='batch size')
    parser.add_argument('-train-path', '--train_path', type=str, default=None)
    parser.add_argument('-eval-path', '--eval_path', type=str, default=None)
    parser.add_argument('-save', '--save', help="Save things for future use?", action="store_true")

    args = parser.parse_args()
    if args.cuda and torch.cuda.is_available():
        args.device = 'cuda'
    else:
        args.device = 'cpu'



    map_model = MapModel( input_size=1, z_dim=10, w_dim=2,num_subj=40).to(args.device)
    map_opt=torch.optim.Adam(map_model.parameters(), lr=1e-3)


    train_test='test'


    num_subjects=15
    size=2
    if train_test=='train':
        train_loader = torch.load(os.path.join('Data', train_test, 'Data_Loaders', 'train_loader.pt'))
        eval_loader = torch.load(os.path.join('Data', train_test, 'Data_Loaders', 'test_loader.pt'))
        train_eval_loop(train_loader,
                        eval_loader,
                        model=map_model,
                        optimizer=map_opt,
                        training_args=args)
    else:
        eval_loader = torch.load(os.path.join('Data', train_test, 'Data_Loaders', 'test_loader.pt'))
        map_model.load_state_dict(torch.load(os.path.join('Models','mapper')))
        evaluate(dataloader=eval_loader, eval_args=args, model=map_model)

    if args.save:
        torch.save({'model_data': map.state_dict()}, 'Models/model.pt')
