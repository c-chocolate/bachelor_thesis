import random

f1 = open('blacklist.txt','r',encoding='utf-8', errors="", newline="")
#f2 = open('blacklist4000.txt','w',encoding='utf-8', errors="", newline="")

#lista=[]

blacklist = f1.readlines()

#for data in blacklist:
random.shuffle(blacklist)

with open('blacklist4000.txt', 'w') as f:
    f.writelines(blacklist)
