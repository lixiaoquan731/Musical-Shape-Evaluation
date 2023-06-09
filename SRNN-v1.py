# import the necessary libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import config_SRNN as config
from utils import imshow, show_plot
import torchvision
from torch.autograd import Variable
from PIL import Image
import PIL.ImageOps
import os
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score, \
    precision_recall_fscore_support, f1_score, precision_score, recall_score
import logging
from torchsummary import summary
from thop import profile

# preprocessing and loading the dataset
class SiameseDataset:
    def __init__(self, training_csv=None, training_dir=None, transform=None):
        # used to prepare the labels and images path
        self.train_df = pd.read_excel(training_csv)
        self.train_df.columns = ["image1", "image2", "label"]
        self.train_dir = training_dir
        self.transform = transform

    def __getitem__(self, index):
        # getting the image path
        image1_path = os.path.join(self.train_dir, self.train_df.iat[index, 0])
        image2_path = os.path.join(self.train_dir, self.train_df.iat[index, 1])

        # Loading the image
        img0 = Image.open(image1_path)
        img1 = Image.open(image2_path)
        # img0 = img0.convert("RGB")
        # img1 = img1.convert("RGB")
        # img0 = self.transform(img0)
        # img1 = self.transform(img1)
        # label= torch.LongTensor(self.train_df.iat[index, 2])
        # return img0, img1, label
        # Apply image transformations
               
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        # img2 = Image.fromarray(out.astype('uint8')).convert('RGB')
        return (img0,img1,
                torch.from_numpy(
                    np.array(int(self.train_df.iat[index, 2]), dtype=np.float32)).view(1, -1)[0].type(torch.LongTensor))

    def __len__(self):
        return len(self.train_df)


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(torch.nn.Module):

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = InstanceNormalization(channels)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = InstanceNormalization(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))

        out = out + residual
        return out


class InstanceNormalization(torch.nn.Module):
    """InstanceNormalization
    Improves convergence of neural-style.
    ref: https://arxiv.org/pdf/1607.08022.pdf
    """

    def __init__(self, dim, eps=1e-9):
        super(InstanceNormalization, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor(dim))
        self.shift = nn.Parameter(torch.FloatTensor(dim))
        self.eps = eps
        self._reset_parameters()

    def _reset_parameters(self):
        self.scale.data.uniform_()
        self.shift.data.zero_()

    def forward(self, x):
        n = x.size(2) * x.size(3)
        t = x.view(x.size(0), x.size(1), n)
        mean = torch.mean(t, 2).unsqueeze(2).unsqueeze(3).expand_as(x)
        # Calculate the biased var. torch.var returns unbiased var
        var = torch.var(t, 2).unsqueeze(2).unsqueeze(3).expand_as(x) * ((n - 1) / float(n))
        scale_broadcast = self.scale.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        scale_broadcast = scale_broadcast.expand_as(x)
        shift_broadcast = self.shift.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        shift_broadcast = shift_broadcast.expand_as(x)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = out * scale_broadcast + shift_broadcast
        return out


# create a siamese network
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(

            nn.Conv2d(3, 64, kernel_size=7, stride=1),
            nn.BatchNorm2d(64),
            nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

        )
        self.cnn2 = nn.Sequential(

            nn.Conv2d(64, 128, kernel_size=5, stride=1),
            nn.BatchNorm2d(128),
            # nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

        )
        self.cnn3 = nn.Sequential(

            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            # nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

        )
        self.cnn4 = nn.Sequential(

            nn.Conv2d(128, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            # nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

        )

        self.res1 = ResidualBlock(64)
        # self.res2 = ResidualBlock(64)
        self.res3 = ResidualBlock(128)
        # self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        # self.res6 = ResidualBlock(256)
        self.res7 = ResidualBlock(64)
        # self.res8 = ResidualBlock(512)

        # Defining the fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(6400, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(2048, 256))

        # Non-linearities
        self.relu = nn.ReLU()
        self.dense1 = nn.Linear(512, 128)
        self.dense2 = nn.Linear(128, 28)
        self.dropout = nn.Dropout(p=0.4)

    def forward_once(self, x):
        # Forward pass
        output = self.cnn1(x)
        y = self.res1(output)
        y = self.cnn2(y)
        y = self.res3(y)
        y = self.cnn3(y)
        y = self.res5(y)
        y = self.cnn4(y)
        y = self.res7(y)

        output = y.view(y.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # forward pass of input 1
        output1 = self.forward_once(input1)
        # forward pass of input 2
        output2 = self.forward_once(input2)

        output = torch.cat((output1, output2), 1)

        output = self.dropout(self.dense1(output))

        output = self.dense2(output)

        return output


#train the model
def train(train_dataloader):
    loss=[] 
    train_loss=0
    for i, data in enumerate(train_dataloader,0):
      img0, img1 , label = data
      img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
      optimizer.zero_grad()
      output = net(img0,img1)
      loss = criterion(output,label.squeeze(1))
      loss.backward()
      optimizer.step()
      train_loss += loss.item()
    return train_loss/len(train_dataloader)


def eval(eval_dataloader):
    loss=[] 
    y_pred_test = []
    ytest=[]
    iteration_number = 0
    correct=0
    total=0
    val_loss=0
    for i, data in enumerate(eval_dataloader,0):
      img0, img1 , label = data
      img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
      output = net(img0,img1)
      predicted = np.argmax(output.detach().cpu().numpy(), axis=1)
      loss = criterion(output,label.squeeze(1))
      val_loss += loss.item()
      y_pred_test = np.concatenate((y_pred_test, predicted))
      ytest = np.concatenate((ytest, [label.item()]))
    return val_loss/len(eval_dataloader), y_pred_test, ytest


if __name__ == '__main__':
    
    # load the dataset
    training_dir = config.training_dir
    testing_dir = config.testing_dir
    training_csv = config.training_csv
    testing_csv = config.testing_csv
    
    # Load the the dataset from raw image folders
    siamese_dataset = SiameseDataset(
        training_csv,
        training_dir,
        transform=transforms.Compose(
            [transforms.Resize([224,224]), transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),]
        ),
    )


    # # Viewing the sample of images and to check whether its loading properly
    # vis_dataloader = DataLoader(siamese_dataset, shuffle=True, batch_size=8)
    # dataiter = iter(vis_dataloader)


    # example_batch = next(dataiter)
    # concatenated = torch.cat((example_batch[0], example_batch[1]), 0)
    # imshow(torchvision.utils.make_grid(concatenated))
    # print(example_batch[2].numpy())
    
    
    # Load the dataset as pytorch tensors using dataloader
    train_dataloader = DataLoader(siamese_dataset,
                            shuffle=True,
                            num_workers=2,
                            batch_size=config.batch_size) 

    #set the device to cuda
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Declare Siamese Network
    net = SiameseNetwork().cuda()
    # Decalre Loss Function
    criterion = nn.CrossEntropyLoss()
    # Declare Optimizer
    # optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.0005)
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.0005)
    # optimizer = torch.optim.SGD(net.parameters(), lr=1e-2, momentum=0.9, weight_decay=0.0001)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,90], gamma=0.1)
    
    # Load the test dataset
    test_dataset = SiameseDataset(
    training_csv=testing_csv,
    training_dir=testing_dir,
    transform=transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225]),]
    ),
    )
    
    eval_dataloader = DataLoader(test_dataset, num_workers=2, batch_size=1, shuffle=False)
    prec=np.zeros(config.epochs)
    recall=np.zeros(config.epochs)
    F1=np.zeros(config.epochs)
    Tloss=np.zeros(config.epochs)
    Eloss=np.zeros(config.epochs)
    counter=[]
    iteration_number = 0
    best_eval_loss = 9999
    for epoch in range(0,config.epochs):
        train_loss = train(train_dataloader)
        eval_loss,y_pred_test, ytest = eval(eval_dataloader)
        Tloss[epoch]=train_loss
        Eloss[epoch]=eval_loss
        prec[epoch]=precision_score(ytest, y_pred_test,average='macro')
        recall[epoch]=recall_score(ytest, y_pred_test,average='macro')
        F1[epoch]=f1_score(ytest, y_pred_test,average='macro')
        AA=accuracy_score(ytest, y_pred_test)
        print(f"Epoch={epoch}","-"*5,f"Training_loss={train_loss}","-"*5,f"Eval_loss={eval_loss}","-"*5, "precision= %.4f" % prec[epoch],
              "recall= %.4f" % recall[epoch],"Fmeasure= %.4f" % F1[epoch])
        # print(f"Training loss{train_loss}")
        # print("-"*10)
        # print(f"Eval loss{eval_loss}")
        # print("-"*10)
        # print("precision= %.4f" % prec[epoch])
        # print("recall= %.4f" % recall[epoch])
        # print("Fmeasure= %.4f" % F1[epoch])
        iteration_number += 10
        counter.append(iteration_number)
        if eval_loss<best_eval_loss:
            best_eval_loss = eval_loss
            print("-"*20)
            print(f"Best Eval loss{best_eval_loss}")
            torch.save(net, ".\checkpoint\model.pth")
            print("Model Saved Successfully") 

    # net = torch.load('D:\Project\Music-IQA\siamese_net-master\siamese-net\checkpoint\model-SRNN-30-224-28class-16patch.pth')
    # net.eval()
    # y_pred_test=[]
    # ytest=[]
    # for i, data in enumerate(eval_dataloader, 0):
    #     x0, x1, label = data
    #     # concat = torch.cat((x0, x1), 0)
    #     output = net(x0.cuda(), x1.cuda())
    #     predicted = np.argmax(output.detach().cpu().numpy(), axis=1)
    #     y_pred_test = np.concatenate((y_pred_test, predicted))
    #     ytest = np.concatenate((ytest, [label.item()]))
    # classification_report(ytest,y_pred_test,digits=4)
    # cm=confusion_matrix(ytest, y_pred_test)
    # prec=precision_score(ytest, y_pred_test,average='macro')
    # recall=recall_score(ytest, y_pred_test,average='macro')
    # F1=f1_score(ytest, y_pred_test,average='macro')
    # AA=accuracy_score(ytest, y_pred_test)