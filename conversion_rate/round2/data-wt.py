# -*- coding: utf-8 -*-
"""
Created on Wed May 09 16:35:36 2018

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
fin='train-day7.txt'
f1='train-day7.txt'
f2='test.txt'

x=pd.read_table(f1)
target='is_trade'
base_rate=sum(x[target])*1.0/len(x[target])

def get_map(x,key):
  g,i,n={},0,x.shape[0]
  while i<n:
    a,t=x[key][i],x[target][i]
    if a in g: g[a][t]+=1
    else:
      h={}
      h[t],h[1-t]=1.0,0.0
      g[a]=h
    i+=1
  for a in g:
    h=g[a]
    k=h[0]+h[1]
    g[a]=[k,h[1]/k]
  return g
key='shop_id'
g=get_map(x,key)

def add_wt(x,g,key):
  n,k=x.shape[0],x[key]
  t,i=[base_rate]*n,0
  while i<n:
    a=k[i]
    if a in g:t[i]=g[a][1]

    i+=1
  x['wt_'+key]=t
  return

def add_wt_all(f1,f2):
  x1=pd.read_table(f1)
  x2=pd.read_table(f1)
  vs=['item_id','item_city_id','item_price_level','item_sales_level',\
      'item_collected_level','item_pv_level','user_id','user_gender_id',\
      'user_age_level','user_occupation_id','user_star_level',\
      'context_page_id','shop_id','shop_review_num_level','shop_star_level',\
      'match_cat_ct','match_prop_ct',]
  for key in vs:
    g=get_map(x1,key)
    add_wt(x1,g,key)
    add_wt(x2,g,key)
  x1.to_csv(f1, index=None, sep='\t')
  x2.to_csv(f2, index=None, sep='\t')
  return

add_wt_all(f1,f2)