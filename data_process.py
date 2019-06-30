import ssw
import os
import cv2
import numpy
data_path="./JPEGImages"
f = open('./ssw.txt','w')
data_txt=open('annotations.txt', 'r')
c=0
string=[]
for line in data_txt:
    line = line.rstrip()
    words = line.split()
    if not (words[0][0:4] == '2007' or words[0][0:4] == '2008'):
        img=cv2.imread(os.path.join(data_path, str(words[0])+".jpg"))
        img=cv2.resize(img,(480,480))
        a=ssw.ssw(img)
        a=ssw.feature_mapping(a)
        a=list(numpy.array(a).flat)
        string=str(words[0])+" "+" ".join(str(i) for i in a)+'\n'
        f.write(string)
        print(c)
        c=c+1
f.close()
data_txt.close()
