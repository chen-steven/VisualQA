import torch
import torch.nn as nn
from dataset import Dataset
from vqa_model import VQAModel
def train():
    model = VQAModel(18248, 300)
    checkpoint = torch.load('checkpoint.tar')
    model.load_state_dict(checkpoint)
    model.train()
    optim = torch.optim.Adam(model.grad_params(), 0.001)
    dataset = Dataset('', 'train')
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)
    device = torch.device("cuda:0")
    best_acc = 63.33
    model.to(device)
    for f, bb, spat, obs, a, q, q_len, qid in loader:
        f, bb, spat, obs, a, q, q_len, qid = f.to(device), bb.to(device), spat.to(device), \
                                             obs.to(device), a.to(device), q.to(device), q_len.to(device), qid.to(device)
                                             
        optim.zero_grad()
        pred=model(q,q_len, f,obs)
        loss = compute_multi_loss(pred, a)
       
        loss_item =  round(loss.item(),3)
        loss.backward()
        nn.utils.clip_grad_norm_(model.grad_params(), 5)
        optim.step()

        acc = compute_multi_acc(pred, a)
        print(loss)
        if acc > best_acc:
            print(acc,loss)
            best_acc = acc
            print('saving...')
            torch.save(model.state_dict(), 'checkpoint.tar')
def compute_multi_loss(logits, labels):
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels).to(labels.device)
    loss *= labels.size(1)
    return loss
def test():
    model = VQAModel(18248, 300)
    checkpoint = torch.load('checkpoint.tar')
    dataset = Dataset('', 'val')
    model.eval()
    model.load_state_dict(checkpoint)
    device = torch.device('cuda:0')
    model.to(device)
    loader = torch.utils.data.DataLoader(dataset, batch_size=512)
    for f, bb, spat, obs, a, q, q_len, qid in loader:
        f, bb, spat, obs, a, q, q_len, qid = f.to(device), bb.to(device), spat.to(device), \
                                             obs.to(device), a.to(device), q.to(device), q_len.to(device), qid.to(device)
        with torch.no_grad():
            pred = model(q, q_len, f, obs)

        acc = round(compute_multi_acc(pred, a),3)
        print(acc)
def compute_multi_acc(logits, labels):
    logit_max = logits.max(dim = 1)[1] # argmax
    logit_max.unsqueeze_(1).to(labels.device)
    one_hots = torch.zeros(*labels.size()).to(labels.device)
    one_hots.scatter_(1, logit_max, 1)
    acc = (one_hots * labels).sum() / labels.size(0) * 100
    return acc.item()

test()

