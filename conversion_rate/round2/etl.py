# -*- coding: utf-8 -*-
"""
Created on Tue May 01 16:33:19 2018

@author: Bender
"""

import numpy as np
import string
import datetime
import math
import time
import pandas as pd
import os
import gc

fn_path='/mnt/resource/tm'
os.chdir(fn_path)
gap=8*60*60

def add_time(fin,fout):
  key='context_timestamp'
  t0=time.time()
  x,y=open(fin),open(fout,'w')
  a=x.readline()
  s=map(string.strip,string.split(a))
  i=s.index(key)

  s+=['day','hour']
  y.write(string.join(s,'\t')+'\n')

  for a in x:
    s=map(string.strip,string.split(a))

    t=datetime.datetime.fromtimestamp(int(s[i])+gap)
    s+=map(str,[t.day,t.hour])

    y.write(string.join(s,'\t')+'\n')
  x.close()
  y.close()
f1='train.txt'
f2='train-time.txt'
#add_time(f1,f2)

f1='test.txt'
f2='test-time.txt'
#add_time(f1,f2)

fin='train-time.txt'
fout='train-day7.txt'
'''
x,y=open(fin),open(fout,'w')
a=x.readline()
s=string.split(a,'\t')
n=len(s)
i=s.index('day')
y.write(a)
for a in x:
  s=string.split(a,'\t')
  if int(s[i])==7: y.write(a)
x.close()
y.close()
'''
fout='train-day7.txt'
t0=time.time()
x=pd.read_table(fout)
print time.time()-t0
print x.shape
print len(x.user_id.unique())

fout='test-time.txt'
t0=time.time()
x=pd.read_table(fout)
print time.time()-t0
print x.shape
print len(x.user_id.unique())
