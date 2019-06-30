from torch.utils import data
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
from math import floor

Transform = transforms.Compose([
    transforms.Resize([480, 480]),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                         std  = [ 0.229, 0.224, 0.225 ]),
    ])

class myDataSet(data.Dataset):
    def __init__(self, root, istest, transfrom):
        self.root = root
        self.data_txt = open('annotations.txt', 'r')
        self.ssw_txt = open('ssw.txt', 'r')
        self.ssw_test_txt = open('ssw_test.txt', 'r')
        self.istest = istest
        self.transform = transfrom
        self.imgs = []
        for line in self.data_txt:
            line = line.rstrip()
            words = line.split()
            if self.istest:
                if words[0][0:4] == '2007' or words[0][0:4] == '2008':
                    label_cur = [0 for i in range(20)]
                    for i in range(1, len(words)):
                        label_cur[int(words[i])] = 1
                        #label_cur.append(int(words[i]))
                    #self.imgs.append([words[0], label_cur])
                    for linee in self.ssw_test_txt:
                        linee = linee.rstrip()
                        wordss = linee.split()
                        if wordss[0] == words[0]:
                            ssw_block = torch.Tensor(floor((len(wordss) - 1) / 4), 4)
                            for i in range(floor((len(wordss) - 1) / 4)):
                                w=max(int(wordss[i * 4 + 3]),2)
                                h=max(int(wordss[i*4+4]),2)
                                ssw_block[i,0]=(30-w if (int(wordss[i*4+1])+w>=31) else int(wordss[i*4+1]))
                                ssw_block[i,2]=w
                                ssw_block[i,1]=(30-h if (int(wordss[i*4+2])+h>=31) else int(wordss[i*4+2]))
                                ssw_block[i,3]=h                        
                            break
                        else:
                            ssw_block = torch.tensor([[0,0,2,2]])   
                    self.imgs.append([words[0], ssw_block,label_cur])
                    
            else:
                if not (words[0][0:4] == '2007' or words[0][0:4] == '2008'):
                    label_cur = [0 for i in range(20)]
                    for i in range(1, len(words)):
                        label_cur[int(words[i])] = 1
                        #label_cur.append(int(words[i]))
                    for linee in self.ssw_txt:
                        linee = linee.rstrip()
                        wordss = linee.split()
                        if wordss[0] == words[0]:
                            ssw_block = torch.Tensor(floor((len(wordss) - 1) / 4), 4)
                            for i in range(floor((len(wordss) - 1) / 4)):
                                w=max(int(wordss[i * 4 + 3]),2)
                                h=max(int(wordss[i*4+4]),2)
                                ssw_block[i,0]=(30-w if (int(wordss[i*4+1])+w>=31) else int(wordss[i*4+1]))
                                ssw_block[i,2]=w
                                ssw_block[i,1]=(30-h if (int(wordss[i*4+2])+h>=31) else int(wordss[i*4+2]))
                                ssw_block[i,3]=h                        
                            break
                        else:
                            ssw_block = torch.tensor([[0,0,2,2]])   
                    self.imgs.append([words[0], ssw_block,label_cur])
                    
    def __getitem__(self, index):
        cur_img = Image.open(self.root + self.imgs[index][0] + '.jpg')
        data_once = self.transform(cur_img)
        label_once = self.imgs[index][2]
        ssw_block=self.imgs[index][1]
        '''
        for line in self.ssw_txt:
            line = line.rstrip()
            words = line.split()
            ssw_block = torch.tensor([0,0,2,2])
            if words[0] == self.imgs[index][0]:
                ssw_block = torch.Tensor(floor((len(words) - 1) / 4), 4)
                for i in range(floor((len(words) - 1) / 4)):
                    w=max(int(words[i * 4 + 3]),2)
                    h=max(int(words[i*4+4]),2)
                    ssw_block[i,0]=(30-w if (int(words[i*4+1])+w>=31) else int(words[i*4+1]))
                    ssw_block[i,2]=w
                    ssw_block[i,1]=(30-h if (int(words[i*4+2])+h>=31) else int(words[i*4+2]))
                    ssw_block[i,3]=h                    
                
                break
                '''
        return data_once, ssw_block, torch.Tensor(label_once)
    
    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    trainData = myDataSet('JPEGImages/', 0, Transform)
    testData = myDataSet('JPEGImages/' ,1, Transform)
    print('trainData', len(trainData))
    print('testData', len(testData))
    print(trainData[1][1])
