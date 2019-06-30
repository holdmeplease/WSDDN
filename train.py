import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchsummary import summary
from torch.autograd import Variable
import torch.optim as optim
import argparse
from model_wsddn import WSDDN
from data_pre import myDataSet
import os
from tensorboardX import SummaryWriter
import ssw

Transform = transforms.Compose([
    transforms.Resize([480, 480]),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                         std  = [ 0.229, 0.224, 0.225 ]),
    ])

parser = argparse.ArgumentParser(description='wsddn Input:BatchSize initial LR EPOCH')
parser.add_argument('--test','-t', action = 'store_true',
 help='set test mode')
parser.add_argument('--model_path', type=str,default='./model_para',
 help='dir to save para')
parser.add_argument('--BATCH_SIZE', type=int,default=1,
 help='batch_size')
parser.add_argument('--LR', type=float,default=0.00001,
 help='Learning Rate')
parser.add_argument('--EPOCH', type=int,default=40,
 help='epoch')
parser.add_argument('--GPU', type=int,default=0,
 help='GPU')
args = parser.parse_args()
model_path=args.model_path
BATCH_SIZE=args.BATCH_SIZE
LR=args.LR
EPOCH=args.EPOCH
print('model_path:',model_path)
print('batch_size:',BATCH_SIZE)
print('initial LR:',LR)
print('epoch:',EPOCH)

torch.cuda.set_device(args.GPU)
net_wsddn = WSDDN('VGG11')
if os.path.exists(os.path.join(model_path, 'wsddn.pkl')):
    net_wsddn.load_state_dict(torch.load(os.path.join(model_path, 'wsddn.pkl')))
else:
    pretrained_dict = torch.load('vgg11_bn-6002323d.pth.1')
    modified_dict = net_wsddn.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in modified_dict}
    modified_dict.update(pretrained_dict)
    net_wsddn.load_state_dict(modified_dict)
net_wsddn.cuda()

criterion = nn.BCELoss(weight=None, size_average=True) 
optimizer1 = optim.SGD(net_wsddn.parameters(), lr = LR, momentum = 0.9)
optimizer2 = optim.SGD(net_wsddn.parameters(), lr = 0.1 * LR, momentum = 0.9)
writer = SummaryWriter('WSDDN')
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
trainData = myDataSet('JPEGImages/', 0, Transform)
testData = myDataSet('JPEGImages/' ,1, Transform)
#print('trainData', len(trainData))
#print('testData', len(testData))

trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=False,num_workers=1)
testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCH_SIZE, shuffle=False)
if not args.test:
    net_wsddn.train()
    for epoch in range(EPOCH):
        #scheduler.step(epoch)
        running_loss = 0.0
        print(epoch)
        for i, (images, kuang,labels) in enumerate(trainLoader):
            images = Variable(images).cuda()
            labels = Variable(labels).cuda()
            kuang =Variable(kuang).cuda()
            if epoch < 10:
                optimizer1.zero_grad()
            else:
                optimizer2.zero_grad()
            #ssw
            #print(kuang)
            '''
            if kuang.size(1)==0:
                print(kuang)
                continue
                '''
            #kuang=kuang.view([1,*kuang.shape])
            #print(kuang.shape)
            #forward + backward + optimizer
            outputs_1, output_2, output_3 = net_wsddn(images,kuang)
            outputs_1=torch.sigmoid(outputs_1)
            loss = criterion(outputs_1 , labels)
            loss.backward()
            if epoch < 10:
                optimizer1.step()
            else:
                optimizer2.step()

            running_loss += loss.item()

            if i % 500 == 499:
                print('[%d , %5d] loss: %.3f' % (epoch + 1 , i + 1 , running_loss / 500))
                running_loss = 0.0
        writer.add_scalar('Train/loss', loss.item(),epoch)
        torch.save(net_wsddn.state_dict(), os.path.join(model_path, 'wsddn.pkl'))
    print('Finished Training')
    writer.close()
    torch.save(net_wsddn.state_dict(), os.path.join(model_path, 'wsddn.pkl'))
else:
    ##UNFINISHED
    
    net_wsddn.eval()
    result_name = 'box_result.txt'
    f = open(result_name, 'w')
    for i, (images, kuang, labels) in enumerate(testLoader):
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        kuang = Variable(kuang).cuda()
        outputs_1, output_2, output_3 = net_wsddn(images,kuang)
        for j in range(outputs_1.size(1)):
            if outputs_1[0, j] > 0.05:
                for k in range(output_2.size(0)):
                    if output_2[0, k, j] > 0.1:
                        #print(kuang.shape)
                        new_line = [i, j, float('%.3f' % output_3[0, k, j].item()), 8 * kuang[0, k, 0].item(), 
                                    8 * kuang[0, k, 1].item(), 8 * kuang[0, k, 2].item(), 8 * kuang[0, k, 3].item()]
                        #new_line = str(i) + ' ' +  str(j) + ' ' + str(kuang[0, k, 0].item()) + ' ' + str(kuang[0, k, 1].item()) +
                        #           ' ' + str(kuang[0, k, 2].item()) + ' ' + str(kuang[0, k, 3].item()) + '\n'
                        for line_mem in new_line:
                            f.write(str(line_mem) + ' ')
                        f.write('\r\n')
        if (i % 500) == 0:
            print(i)
            #predicted = outputs_1.data>=0.5
            #vec_1 += (predicted.float() == labels).cpu().float().sum(0) #correct_num
            #vec_2 += labels.cpu().sum(0)#appear_num
            #equal to predicted=outputs.data>=0
            #total += labels.size(0)*labels.size(1)
            #correct += (predicted.float() == labels).sum()
        #print('Classification Accuracy of the model on the train images(mAcc): %.4f %%' % (100 * float(correct) / float(total)))
        #print('Localization Accuracy of the model on the train images(mAP): %.4f %%' % (100 * (vec_1*vec_2).sum()))
    f.close()
    data1 = open('box_result.txt', 'r')
    data2 = open('bonus_ground_truth.txt', 'r')
    #data3 = open('meiren/annotations.txt', 'r')
    f = open('for_map.txt', 'w')
    for line in data1:
        c = 0
        #print('c', c)
        line = line.rstrip()
        words = line.split()
        data3 = open('annotations.txt', 'r')
        for line1 in data3:
            #print(int(words[0]))
            if c == int(words[0]):
                #print('ok')
                line1 = line1.rstrip()
                words1 = line1.split()
                new_line = [words1[0], words[1], words[2], words[3], words[4], words[5], words[6]]
                for line_mem in new_line:
                    f.write(str(line_mem) + ' ')
                f.write('\r\n')
                data3.close()
                break
            c += 1
    data1.close()
    data2.close()
    #data3.close()
    f.close()
    '''
        for images, labels in testLoader:
            images = Variable(images).cuda()
            labels= Variable(labels).cuda()
            outputs_1,output_2 = net_wsddn(images,kuang)
            predicted = outputs_1.data>=0.5
            vec_1 += (predicted.float() == labels).cpu().float().sum(0) #correct_num
            vec_2 += labels.cpu().sum(0)#appear_num
            #equal to predicted=outputs.data>=0
            total += labels.size(0)*labels.size(1)
            correct += (predicted.float() == labels).sum()
        print('Classification Accuracy of the model on the test images(mAcc): %.4f %%' % (100 * float(correct) / float(total)))
        print('Localization Accuracy of the model on the test images(mAP): %.4f %%' % (100 * (vec_1*vec_2).sum()))
    '''
