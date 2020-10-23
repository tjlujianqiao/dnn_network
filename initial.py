import torch as t
import torchvision as tv
import numpy as np
import time


# 超参数
EPOCH = 10
BATCH_SIZE = 100
DOWNLOAD_MNIST = True   # 下过数据的话, 就可以设置成 False
N_TEST_IMG = 10          # 到时候显示 5张图片看效果, 如上图一



class DNN(t.nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
 
        train_data = tv.datasets.FashionMNIST(
        root="./fashionmnist/",
        train=True,
        transform=tv.transforms.ToTensor(),
        download=DOWNLOAD_MNIST
        # here I need to figure out the type of the train_data
        )

        test_data = tv.datasets.FashionMNIST(
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
            t.nn.Linear(28*28,512),
            #here 28*28 need to be modified
            t.nn.Dropout(0.5),
            t.nn.ELU(),
            t.nn.Linear(512,128),
            t.nn.Dropout(0.5),
            t.nn.ELU(),
            t.nn.Linear(128,10),
        )

        self.lr = 0.001
        self.loss = t.nn.CrossEntropyLoss()
        self.opt = t.optim.Adam(self.parameters(), lr = self.lr)

    def forward(self,x):

        nn1 = x.view(-1,28*28)
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
            b_x = x   # batch x, shape (batch, 28*28)
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
            if(step%100 == 0):
                if(use_gpu):
                    print(e,step,losses.data.cpu().numpy())
                else:
                    print(e,step,losses.data.numpy())
                
                model.eval() # train model dropout not use
                for (tx,ty) in testloader:
                    t_x = tx   # batch x, shape (batch, 28*28)
                    t_y = ty
                    if(use_gpu):
                        t_x = t_x.cuda()
                        t_y = t_y.cuda()
                    t_out = model(t_x)
                    if(use_gpu):
                        acc = (np.argmax(t_out.data.cpu().numpy(),axis=1) == t_y.data.cpu().numpy())
                    else:
                        acc = (np.argmax(t_out.data.numpy(),axis=1) == t_y.data.numpy())

                    print(time.time() - ts ,np.sum(acc)/1000)
                    ts = time.time()
                    break#只测试前1000个
            


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
        t_x = tx   # batch x, shape (batch, 28*28)
        t_y = ty

        t_out = net(t_x)
        #acc = (np.argmax(t_out.data.CPU().numpy(),axis=1) == t_y.data.CPU().numpy())
        acc = (np.argmax(t_out.data.numpy(),axis=1) == t_y.data.numpy())

        print(np.sum(acc)/1000)

if __name__ == "__main__":
    train()
