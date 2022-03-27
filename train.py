import torchvision
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from model import vgg


if __name__ == '__main__':

    total_train_step = 0

    total_test_step = 0

    epoch = 30

    learn_rate = 0.05

    writer = SummaryWriter("logs_train")

    m = vgg(3, 10)
    if torch.cuda.is_available():
        m = m.cuda()


    loss_fn = nn.CrossEntropyLoss()

    loss_fn = loss_fn.cuda()

    optim = torch.optim.SGD(m.parameters(), lr=learn_rate)

    # optim = torch.optim.Adam(m.parameters(), lr=learn_rate)



    img_preprocess = transforms.Compose([transforms.Resize(size = (224, 224)), transforms.ToTensor(), transforms.Normalize(
       (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])



    train_data = torchvision.datasets.CIFAR10("./data", train=True,transform=img_preprocess,download=True)

    test_data = torchvision.datasets.CIFAR10("./data", train=False,transform=img_preprocess,download=True)

    test_data_size = len(test_data)

    print(len(train_data))
    print(len(test_data))

    train_dataloader = DataLoader(train_data, batch_size=32)

    test_dataloader = DataLoader(test_data, batch_size=32)

    for i in range(epoch):
        print("epoch {}:".format(i))
        m.train()
        for data in train_dataloader:
            imgs, targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs = m(imgs)
            loss = loss_fn(outputs, targets)

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_train_step = total_train_step + 1


            if (total_train_step % 50 == 0):
                print("trained {} batch(s), loss {}".format(total_train_step, loss.item()))
                writer.add_scalar("train_loss", loss.item(), total_train_step)

        m.eval()
        total_test_loss = 0
        total_accuracy = 0
        with(torch.no_grad()):
            for data in test_dataloader:
                imgs, targets = data
                imgs = imgs.cuda()
                targets = targets.cuda()
                outputs = m(imgs)
                loss = loss_fn(outputs, targets)
                total_test_loss = total_test_loss + loss
                accuracy = (outputs.argmax(1) == targets).sum()
                total_accuracy = total_accuracy + accuracy

        print("total test loss {}".format(total_test_loss))
        print("total accuracy {}".format(total_accuracy / test_data_size))
        writer.add_scalar("test_loss", total_test_loss.item(), total_test_step)
        writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
        total_test_step = total_test_step + 1

        torch.save(m.state_dict(), "save_models/VGG16_" + str(i) + ".pt")

    writer.close()
    #tensorboard --logdir=logs_train --port=6007

