import matplotlib.pyplot as plt
import matplotlib
import numpy as np


import matplotlib.pyplot as plt
import matplotlib
import numpy as np

def display_scores(sc):
    
    
    ft=10
    ft_label=12
    
    bs=sc.size(0)
    nb_class=sc.size(1)
    
    f, ax = plt.subplots(1, bs)

    if bs ==2:
         f.set_size_inches(8, 8)
         f.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=None, hspace=0.5)
    else:
         f.set_size_inches(12, 12)
         f.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=None, hspace=0.25)
    
    max_score= sc.max().item()
    min_score=sc.min().item()

    label_pos= min_score-8
    xmin=-5.5
    xmax=5.5

    
    label=[]
    for i in range(nb_class):
        str_nb="{0:.0f}".format(i)
        mystr='class '+ str_nb
        label.append(mystr)
        
    y_pos = np.arange(nb_class)*1.2
               
        
        
    for count in range(bs):
        
        ax[count].set_title('data point '+ "{0:.0f}".format(count))
        
        scores=sc[count].numpy()

        width=0.9
        col='darkgreen'

    #    plt.rcdefaults()
        
        # line in the middle
        ax[count].plot([0,0], [y_pos[0]-1,y_pos[-1]+1], color='k',linewidth=4)


        # the plot
        barlist=ax[count].barh(y_pos, scores, width , align='center', color=col)

        for idx,bar in enumerate(barlist):
            if scores[idx]<0:
                bar.set_color('r')

        ax[count].set_xlim([xmin, xmax])
        ax[count].invert_yaxis()  

        # no y label
        ax[count].set_yticklabels([])
        ax[count].set_yticks([])

        # x label
        ax[count].set_xticklabels([])
        ax[count].set_xticks([])


        ax[count].spines['right'].set_visible(False)
        ax[count].spines['top'].set_visible(False)
        ax[count].spines['bottom'].set_visible(False)
        ax[count].spines['left'].set_visible(False)
        
        ax[count].set_aspect('equal')


        for i in range(len(scores)):

            str_nb="{0:.1f}".format(scores[i])
            if scores[i]>=0:
                ax[count].text( scores[i] + 0.05 , y_pos[i] ,str_nb ,
                     horizontalalignment='left', verticalalignment='center',
                     transform=ax[count].transData, color= col,fontsize=ft)
            else:
                ax[count].text( scores[i] - 0.05 , y_pos[i] ,str_nb ,
                     horizontalalignment='right', verticalalignment='center',
                     transform=ax[count].transData, color= 'r',fontsize=ft)
                
            if  count ==0: 
                ax[0].text( label_pos , y_pos[i] , label[i] ,
                         horizontalalignment='left', verticalalignment='center',
                         transform=ax[0].transData, color= 'black',fontsize=ft_label)

         
    plt.show()



import os.path
def check_mnist_dataset_exists():
    flag_train_data = os.path.isfile('../data/mnist/train_data.pt') 
    flag_train_label = os.path.isfile('../data/mnist/train_label.pt') 
    flag_test_data = os.path.isfile('../data/mnist/test_data.pt') 
    flag_test_label = os.path.isfile('../data/mnist/test_label.pt') 
    if flag_train_data==False or flag_train_label==False or flag_test_data==False or flag_test_label==False:
        print('MNIST dataset missing - downloading...')
        import torchvision
        import torchvision.transforms as transforms
        trainset = torchvision.datasets.MNIST(root='../data/mnist/temp', train=True,
                                                download=True, transform=transforms.ToTensor())
        testset = torchvision.datasets.MNIST(root='../data/mnist/temp', train=False,
                                               download=True, transform=transforms.ToTensor())
        train_data=torch.Tensor(60000,28,28)
        train_label=torch.LongTensor(60000)
        for idx , example in enumerate(trainset):
            train_data[idx]=example[0].squeeze()
            train_label[idx]=example[1]
        torch.save(train_data,'../data/mnist/train_data.pt')
        torch.save(train_label,'../data/mnist/train_label.pt')
        test_data=torch.Tensor(10000,28,28)
        test_label=torch.LongTensor(10000)
        for idx , example in enumerate(testset):
            test_data[idx]=example[0].squeeze()
            test_label[idx]=example[1]
        torch.save(test_data,'../data/mnist/test_data.pt')
        torch.save(test_label,'../data/mnist/test_label.pt')

def check_fashion_mnist_dataset_exists():
    flag_train_data = os.path.isfile('../data/fashion-mnist/train_data.pt') 
    flag_train_label = os.path.isfile('../data/fashion-mnist/train_label.pt') 
    flag_test_data = os.path.isfile('../data/fashion-mnist/test_data.pt') 
    flag_test_label = os.path.isfile('../data/fashion-mnist/test_label.pt') 
    if flag_train_data==False or flag_train_label==False or flag_test_data==False or flag_test_label==False:
        print('FASHION-MNIST dataset missing - downloading...')
        import torchvision
        import torchvision.transforms as transforms
        trainset = torchvision.datasets.FashionMNIST(root='../data/fashion-mnist/temp', train=True,
                                                download=True, transform=transforms.ToTensor())
        testset = torchvision.datasets.FashionMNIST(root='../data/fashion-mnist/temp', train=False,
                                               download=True, transform=transforms.ToTensor())
        train_data=torch.Tensor(60000,28,28)
        train_label=torch.LongTensor(60000)
        for idx , example in enumerate(trainset):
            train_data[idx]=example[0].squeeze()
            train_label[idx]=example[1]
        torch.save(train_data,'../data/fashion-mnist/train_data.pt')
        torch.save(train_label,'../data/fashion-mnist/train_label.pt')
        test_data=torch.Tensor(10000,28,28)
        test_label=torch.LongTensor(10000)
        for idx , example in enumerate(testset):
            test_data[idx]=example[0].squeeze()
            test_label[idx]=example[1]
        torch.save(test_data,'../data/fashion-mnist/test_data.pt')
        torch.save(test_label,'../data/fashion-mnist/test_label.pt')

def check_cifar_dataset_exists():
    flag_train_data = os.path.isfile('../data/cifar/train_data.pt') 
    flag_train_label = os.path.isfile('../data/cifar/train_label.pt') 
    flag_test_data = os.path.isfile('../data/cifar/test_data.pt') 
    flag_test_label = os.path.isfile('../data/cifar/test_label.pt') 
    if flag_train_data==False or flag_train_label==False or flag_test_data==False or flag_test_label==False:
        print('CIFAR dataset missing - downloading... (5min)')
        import torchvision
        import torchvision.transforms as transforms
        trainset = torchvision.datasets.CIFAR10(root='../data/cifar/temp', train=True,
                                        download=True, transform=transforms.ToTensor())
        testset = torchvision.datasets.CIFAR10(root='../data/cifar/temp', train=False,
                                       download=True, transform=transforms.ToTensor())  
        train_data=torch.Tensor(50000,3,32,32)
        train_label=torch.LongTensor(50000)
        for idx , example in enumerate(trainset):
            train_data[idx]=example[0]
            train_label[idx]=example[1]
        torch.save(train_data,'../data/cifar/train_data.pt')
        torch.save(train_label,'../data/cifar/train_label.pt') 
        test_data=torch.Tensor(10000,3,32,32)
        test_label=torch.LongTensor(10000)
        for idx , example in enumerate(testset):
            test_data[idx]=example[0]
            test_label[idx]=example[1]
        torch.save(test_data,'../data/cifar/test_data.pt')
        torch.save(test_label,'../data/cifar/test_label.pt')