def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', '-e', type=int, default=5,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images in each mini-batch')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--model', '-m', default='',
                        help='Give a model to test')
    parser.add_argument('--train-plot', action='store_true', default=False,
                        help='Plot train loss')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print("Args: %s" % args)

    # create pair dataset iterator
    train = dsets.MNIST(
        root='../data/',
        train=True,
        # transform=transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.1307,), (0.3081,))
        # ]),
        download=True
    )
    test = dsets.MNIST(
        root='../data/',
        train=False,
        # XXX ToTensor scale to 0-1
        transform=transforms.Compose([
            transforms.ToTensor(),
        #     transforms.Normalize((0.1307,), (0.3081,))
        ])
    )

    train_iter = create_iterator(
        train.train_data.numpy(),
        train.train_labels.numpy(),
        args.batchsize)

    # model
    model = SiameseNetwork()
    if args.cuda:
        model.cuda()

    learning_rate = 0.01
    momentum = 0.9
    # Loss and Optimizer
    criterion = ContrastiveLoss()
    # optimizer = torch.optim.Adam(
    #     [p for p in model.parameters() if p.requires_grad],
    #     lr=learning_rate
    # )

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                momentum=momentum)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        train_iter,
        batch_size=args.batchsize, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        test,
        batch_size=args.batchsize, shuffle=True, **kwargs)

    def train(epoch):
        train_loss = []
        model.train()
        start = time.time()
        start_epoch = time.time()
        for batch_idx, (x0, x1, labels) in enumerate(train_loader):
            labels = labels.float()
            if args.cuda:
                x0, x1, labels = x0.cuda(), x1.cuda(), labels.cuda()
            x0, x1, labels = Variable(x0), Variable(x1), Variable(labels)
            output1, output2 = model(x0, x1)
            loss = criterion(output1, output2, labels)
            train_loss.append(loss.data[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            accuracy = []

            for idx, logit in enumerate([output1, output2]):
                corrects = (torch.max(logit, 1)[1].data == labels.long().data).sum()
                accu = float(corrects) / float(labels.size()[0])
                accuracy.append(accu)

            if batch_idx % args.batchsize == 0:
                end = time.time()
                took = end - start
                for idx, accu in enumerate(accuracy):
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}\tTook: {:.2f}\tOut: {}\tAccu: {:.2f}'.format(
                        epoch, batch_idx * len(labels), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.data[0],
                        took, idx, accu * 100.))
                start = time.time()
        torch.save(model.state_dict(), './model-epoch-%s.pth' % epoch)
        end = time.time()
        took = end - start_epoch
        print('Train epoch: {} \tTook:{:.2f}'.format(epoch, took))
        return train_loss

    def test(model):
        model.eval()
        all = []
        all_labels = []

        for batch_idx, (x, labels) in enumerate(test_loader):
            if args.cuda:
                x, labels = x.cuda(), labels.cuda()
            x, labels = Variable(x, volatile=True), Variable(labels)
            output = model.forward_once(x)
            all.extend(output.data.cpu().numpy().tolist())
            all_labels.extend(labels.data.cpu().numpy().tolist())

        numpy_all = np.array(all)
        numpy_labels = np.array(all_labels)
        return numpy_all, numpy_labels

    def plot_mnist(numpy_all, numpy_labels):
        c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
                '#ff00ff', '#990000', '#999900', '#009900', '#009999']

        for i in range(10):
            f = numpy_all[np.where(numpy_labels == i)]
            plt.plot(f[:, 0], f[:, 1], '.', c=c[i])
        plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
        plt.savefig('result.png')

    if len(args.model) == 0:
        train_loss = []
        for epoch in range(1, args.epoch + 1):
            train_loss.extend(train(epoch))

        if args.train_plot:
            plt.gca().cla()
            plt.plot(train_loss, label="train loss")
            plt.legend()
            plt.draw()
            plt.savefig('train_loss.png')
            plt.gca().clear()

    else:
        saved_model = torch.load(args.model)
        model = SiameseNetwork()
        model.load_state_dict(saved_model)
        if args.cuda:
            model.cuda()

    numpy_all, numpy_labels = test(model)
    plot_mnist(numpy_all, numpy_labels)

if __name__ == '__main__':
    main()
    