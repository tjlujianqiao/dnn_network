import torch as t
import torchvision as tv
import numpy as np
import time
import pandas as pd
from torchvision.datasets.mnist import MNIST
from PIL import Image
import matplotlib.pyplot as plt

# 超参数

EPOCH = 10
BATCH_SIZE = 1
DOWNLOAD_MNIST = True   
N_TEST_IMG = 10          
__all__ = {
    '无': 0,
    '轻微': 1,
    '中级': 2,
    '强烈': 3,
    '中等': 2,
}

error =[]

class Rockstone(MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(MNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train  # training set or test set

        if self.train:
            data_file = './data/training_data.xlsx'
        else:
            data_file = './data/testing_data.xlsx'
        
        self.data, self.targets = self.getrawdata(data_file)
    def getrawdata(self, data_file):
        df = pd.read_excel(data_file)
        return t.tensor(df.loc[:, ['sig1','sig2','sig3','sig4']].values.tolist(), dtype=t.float32), np.array(df.loc[:,'岩爆等级'].values.tolist())
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        
        img, target = self.data[index], self.targets[index]
        
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
      
        target = __all__[target]

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder,
                                            self.test_file)))

class DNN(t.nn.Module):
    
    def __init__(self):
        super(DNN, self).__init__()
        train_data = Rockstone( 
        root="./fashionmnist/",
        train=True,
        transform=tv.transforms.ToTensor(),
        download=DOWNLOAD_MNIST)
        

        test_data = Rockstone(
        root="./fashionmnist/",
        train=False,
        transform=tv.transforms.ToTensor(),
        download=DOWNLOAD_MNIST
        )

        #print(test_data)


        # Data Loader for easy mini-batch return in training, the image batch need to be modified
        self.train_loader = t.utils.data.DataLoader(
            dataset=train_data, 
            batch_size=BATCH_SIZE,
            shuffle=True)
        # actually, we don't need shuffle here.

        self.test_loader = t.utils.data.DataLoader(
            dataset=test_data, 
            batch_size=1000,
            shuffle=True) 
            

        self.dnn = t.nn.Sequential(
            t.nn.Linear(4*1,50),
            #here 4*1 need to be modified
            t.nn.Dropout(0.5),
            t.nn.ELU(),
            t.nn.Linear(50,25),
            t.nn.Dropout(0.5),
            t.nn.ELU(),
            t.nn.Linear(25,4),
        )

        self.lr = 0.001
        self.loss = t.nn.CrossEntropyLoss()
        self.opt = t.optim.Adam(self.parameters(), lr = self.lr)

    def forward(self,x):

        nn1 = x.view(-1,4*1)
        #also need to  
        #print(nn1.shape)

        out = self.dnn(nn1)
        #print(out.shape)
        return(out)

def train():
    use_gpu =   not t.cuda.is_available()
    model = DNN()
    if(use_gpu):
        model.cuda()
    print(model)
    loss = model.loss
    opt = model.opt
    dataloader = model.train_loader
    testloader = model.test_loader

    
    for e in range(EPOCH):
        step = 0
        ts = time.time()
        for (x, y) in (dataloader):

            model.train()# train model dropout used
            step += 1
            b_x = x   # batch x, shape (batch, 4)
            #print(b_x.shape)
            b_y = y
            if(use_gpu):
                b_x = b_x.cuda()
                b_y = b_y.cuda()
            out = model(b_x)

            losses = loss(out,b_y)
            opt.zero_grad()
            losses.backward()
            opt.step()
        
            if(step%5 == 0):
 
                if(use_gpu):
                    print(e,step,losses.data.cpu().numpy())
                else:
                    print(e,step,losses.data.numpy())

                model.eval() # train model dropout not use
                for (tx,ty) in testloader:
                    t_x = tx   # batch x, shape (batch, 4*1)
                    t_y = ty
                    if(use_gpu):
                        t_x = t_x.cuda()
                        t_y = t_y.cuda()
                    t_out = model(t_x)

                    if(use_gpu):
                        acc = (np.argmax(t_out.data.cpu().numpy(),axis=1) == t_y.data.cpu().numpy())
                    else:
                        acc = (np.argmax(t_out.data.numpy(),axis=1) == t_y.data.numpy())
                    error.append(np.sum(acc)/5)
                    print(time.time() - ts ,np.sum(acc)/6)
                    ts = time.time()
                    break
            


    t.save(model, './model.pkl')  # 保存整个网络
    t.save(model.state_dict(), './model_params.pkl')   # 只保存网络中的参数 (速度快, 占内存少)
    #加载参数的方式
    """net = DNN()
    net.load_state_dict(t.load('./model_params.pkl'))
    net.eval()"""
    #加载整个模型的方式
    net = t.load('./model.pkl')
    net.cpu()
    net.eval()
    for (tx,ty) in testloader:
        t_x = tx   # batch x, shape (batch, 4*1)
        t_y = ty

        t_out = net(t_x)
        #acc = (np.argmax(t_out.data.CPU().numpy(),axis=1) == t_y.data.CPU().numpy())
        acc = (np.argmax(t_out.data.numpy(),axis=1) == t_y.data.numpy())
        print('111')
        print(np.sum(acc)/6)
        print(error)
        fig,ax = plt.subplots(1,1,sharex = True,figsize=(6,5))
        ax.plot(range(1,len(error)+1),error)
        fig.savefig('1.png',format = 'png')
if __name__ == "__main__":
    train()
