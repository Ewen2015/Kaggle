# -*- coding: utf-8 -*-
"""
Created on Tue May 01 12:07:47 2018

@author: Bender
"""

import random
import string
import time
import os

fn_path='C:/Users/Bender/Documents/data/alibaba'
fn_trn='train.txt'
n_sample=10000

os.chdir(fn_path)

t0=time.time()
x=open(fn_trn)
g=set()
a=x.readline()
s=map(string.strip,string.split(a))
i=s.index('user_id')

for a in x:
  s=map(string.strip,string.split(a))
  g.add(s[i])
x.close()
print(time.time()-t0) 


g=list(g)
random.shuffle(g)
g=set(random.sample(g,n_sample))

fin='train.txt'
fout='train-'+str(n_sample)+'.txt'

x,y=open(fin),open(fout,'w')
a=x.readline()
y.write(a)
s=map(string.strip,string.split(a))
i=s.index('user_id')

for a in x:
  s=map(string.strip,string.split(a))
  if s[i] in g: y.write(a)

x.close()
y.close()

