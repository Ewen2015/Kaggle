# -*- coding: utf-8 -*-
"""
Created on Fri May 04 20:29:26 2018

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

fn_path='C:/Users/Bender/Documents/data/alibaba'
os.chdir(fn_path)

fn_train='train-day7.txt'
target='is_trade'

t0=time.time()
y=pd.read_table(fn_train)
print(time.time()-t0)

for a in y: print(a)
print(y.shape)

tot_instnace=y.shape[0]

def map_catvar1(x,key):
  g,i={},0
  while i<x.shape[0]:
    v,t=x[key][i],x[target][i]
    if v in g:
      h=g[v]
      h[t]+=1
    else:
      h={}
      if t==1: h[0],h[1]=0.0,1.0
      else: h[0],h[1]=1.0,0.0
      g[v]=h
    i+=1
  for a in g:
    h=g[a]
    g[a]=h[1]/(h[0]+h[1])
  return g

t0=time.time()
g=map_catvar1(y,'shop_id')
print(time.time()-t0)
  