import os
import math
import time

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from CNN import CNN
from CNN2 import CNN2
from load_dataset import chunksDataset, imgDataset
from load_dataset import read_chunks, read_image_w_label


test_dir = os.path.join("xingbake", "test.txt")
train_dir = os.path.join("xingbake", "train.txt")
sample_path = os.path.join("xingbake", "api")
record_file = os.path.join(os.getcwd(), "record.txt")
model_path = os.path.join(os.getcwd(), "models")
model_path2 = os.path.join(os.getcwd(), "models2")

iter_num = 100

dtype1 = torch.cuda.FloatTensor
dtype2 = torch.cuda.LongTensor


def trainer2(train_data, test_data, save_dir, iter):
    # Initialize
    model = CNN2(num_classes=3)
    # model.load_state_dict(torch.load(os.path.join(model_path, 'model7-3.pkl')))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    loss_func = torch.nn.CrossEntropyLoss()
    writer = SummaryWriter()
    train_loader = DataLoader(dataset=train_data, batch_size=4, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=1)
    model.cuda()

    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

    print("train begin...")
    total_loss = 0
    last_time = time.time()
    # main loop
    for epoch in range(iter):
        step = 0
        for imgs, labels in train_loader:
            step += 1
            imgs, labels = Variable(imgs.type(dtype1)), Variable(labels.type(dtype2))
            optimizer.zero_grad()
            output = model(imgs)
            loss = loss_func(output, labels)
            # if math.isnan(loss.data[0]):
            #     print("%d step loss is nan" % step)
            total_loss += loss.data[0]
            loss.backward()
            optimizer.step()
            writer.add_scalar('data/loss', loss.data[0], step)
            # report every 100 steps
            if step % 100 == 0:
                current_time = time.time()
                duration = current_time - last_time
                last_time = current_time
                f = open(record_file, "a+")
                print("epoch: %d, step:%d, loss:%.4f, time cost:%.2f" % (epoch, step, (total_loss / 100), duration), file=f)
                f.close()
                total_loss = 0
            # evaluate the accuracy
            if step % 2000 == 0:
                print("evaluating...")
                total = 0
                correct = 0
                for data in test_loader:
                    imgs2, labels2 = data
                    output2 = model(Variable(imgs2).type(dtype1))
                    output2 = output2.type(torch.LongTensor)
                    labels2 = labels2.type(torch.LongTensor)
                    _, predicted = torch.max(output2.data, 1)
                    total += labels2.size(0)
                    correct += (predicted == labels2).sum()
                    # print(total)
                acc = 100 * correct / total
                writer.add_scalar('data/accuracy', acc, step)
                f = open(record_file, "a+")
                print("Accuracy: %.4f %%" % acc, file=f)
                f.close()
            # save model every 2000 steps
            if step % 2000 == 0:
                model_name = "model" + str(epoch) + "-" +str(int(step / 2000)) + ".pkl"
                print(model_name)
                model_file = os.path.join(save_dir, model_name)
                torch.save(model.state_dict(), model_file)
        model_name = "model" + str(epoch) + ".pkl"
        model_file = os.path.join(save_dir, model_name)
        torch.save(model.state_dict(), model_file)
    model_file = os.path.join(save_dir, "last_model.pkl")
    torch.save(model.state_dict(), model_file)

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()

def trainer(train_data, test_data, save_dir, iter):
    # Initialize
    model = CNN(num_classes=26)
    model.load_state_dict(torch.load(os.path.join(model_path, 'model7-3.pkl')))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    loss_func = torch.nn.CrossEntropyLoss()
    writer = SummaryWriter()
    train_loader = DataLoader(dataset=train_data, batch_size=4, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=1)
    model.cuda()

    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

    print("train begin...")
    total_loss = 0
    last_time = time.time()
    # main loop
    for epoch in range(iter):
        step = 0
        for imgs, labels in train_loader:
            step += 1
            imgs, labels = Variable(imgs.type(dtype1)), Variable(labels.type(dtype2))
            optimizer.zero_grad()
            output = model(imgs)
            loss = loss_func(output, labels)
            # if math.isnan(loss.data[0]):
            #     print("%d step loss is nan" % step)
            total_loss += loss.data[0]
            loss.backward()
            optimizer.step()
            writer.add_scalar('data/loss', loss.data[0], step)
            # report every 100 steps
            if step % 100 == 0:
                current_time = time.time()
                duration = current_time - last_time
                last_time = current_time
                f = open(record_file, "a+")
                print("epoch: %d, step:%d, loss:%.4f, time cost:%.2f" % (epoch, step, (total_loss / 100), duration), file=f)
                f.close()
                total_loss = 0
            # evaluate the accuracy
            if step % 2000 == 0:
                print("evaluating...")
                total = 0
                correct = 0
                for data in test_loader:
                    imgs2, labels2 = data
                    output2 = model(Variable(imgs2).type(dtype1))
                    output2 = output2.type(torch.LongTensor)
                    labels2 = labels2.type(torch.LongTensor)
                    _, predicted = torch.max(output2.data, 1)
                    total += labels2.size(0)
                    correct += (predicted == labels2).sum()
                    # print(total)
                acc = 100 * correct / total
                writer.add_scalar('data/accuracy', acc, step)
                f = open(record_file, "a+")
                print("Accuracy: %.4f %%" % acc, file=f)
                f.close()
            # save model every 2000 steps
            if step % 2000 == 0:
                model_name = "model" + str(epoch) + "-" +str(int(step / 2000)) + ".pkl"
                print(model_name)
                model_file = os.path.join(save_dir, model_name)
                torch.save(model.state_dict(), model_file)
        model_name = "model" + str(epoch) + ".pkl"
        model_file = os.path.join(save_dir, model_name)
        torch.save(model.state_dict(), model_file)
    model_file = os.path.join(save_dir, "last_model.pkl")
    torch.save(model.state_dict(), model_file)

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()

if __name__ == "__main__":
    # train_files = find_samples(train_dir)
    # print("find files number: %d" % len(train_files))
    # test_files = list(find_images(test_dir))
    # train_ds = chunksDataset(train_files, train_dir)
    # test_ds = chunksDataset(test_files, test_dir, istrain=False)
    # trainer(train_ds, test_ds, model_path, iter_num)
    train_samples = list(read_image_w_label(train_dir))
    test_samples = list(read_image_w_label(test_dir))
    train_ds = imgDataset(train_samples, sample_path)
    test_ds = imgDataset(test_samples, sample_path)
    trainer2(train_ds, test_ds, model_path2, iter_num)
