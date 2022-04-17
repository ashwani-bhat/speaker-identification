import torch
import time
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def check_type_forward(self, in_types):
        assert len(in_types) == 3

        x0_type, x1_type, y_type = in_types
        assert x0_type.size() == x1_type.shape
        assert x1_type.size()[0] == y_type.shape[0]
        assert x1_type.size()[0] > 0
        assert x0_type.dim() == 2
        assert x1_type.dim() == 2
        assert y_type.dim() == 1

    def forward(self, x0, x1, y):
        self.check_type_forward((x0, x1, y))

        # euclidian distance
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss


def train_epoch(dataloader, model, optimizer, epoch, args):
    train_loss = []
    model.train()
    start = time.time()
    start_epoch = time.time()

    with tqdm(enumerate(dataloader), total=len(dataloader), desc="Training epoch "+str(epoch)) as tepoch:
        for batch_idx, (x0, x1, labels) in tepoch:
            labels = labels.float()
            if args.cuda:
                x0, x1, labels = x0.cuda(), x1.cuda(), labels.cuda()
            x0, x1, labels = Variable(x0), Variable(x1), Variable(labels)
            output1, output2 = model(x0, x1)
            loss = ContrastiveLoss()(output1, output2, labels)
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tepoch.set_postfix(loss=loss.item())
            accuracy = []

            for idx, logit in enumerate([output1, output2]):
                corrects = (torch.max(logit, 1)[1].data == labels.long().data).sum()
                accu = float(corrects) / float(labels.size()[0])
                accuracy.append(accu)

    torch.save(model.state_dict(), './saved_models/model-epoch-%s.pth' % epoch)
    end = time.time()
    took = end - start_epoch
    print('Train epoch: {} \tTook:{:.2f}'.format(epoch, took))
    return train_loss


def test_epoch(dataloader, model, args):
    model.eval()
    all = []
    all_labels = []

    for batch_idx, (x, labels) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Testing"):
        if args.cuda:
            x, labels = x.cuda(), labels.cuda()
        x, labels = Variable(x, volatile=True), Variable(labels)
        output = model.forward_once(x)
        all.extend(output.data.cpu().numpy().tolist())
        all_labels.extend(labels.data.cpu().numpy().tolist())

    numpy_pred = np.array(all)
    numpy_labels = np.array(all_labels)
    return numpy_pred, numpy_labels