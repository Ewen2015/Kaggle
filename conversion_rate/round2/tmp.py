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

fn_path='/mnt/resource/tm'
os.chdir(fn_path)
os.chdir(fn_path)
fin='train-base.txt'
f1='train1.txt'
f2='tmp-round2.txt'

x=pd.read_table(fin)
target='is_trade'
base_rate=sum(x[target])*1.0/len(x[target])
del x
gc.collect()

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

def get_map_tuple(x,keys):
  g,i,n={},0,x.shape[0]
  b,rb=[0]*len(keys),range(len(keys))
  while i<n:
    for j in rb: b[j]=x[keys[j]][i]
    a=tuple(b)
    t=x[target][i]
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
'''
keys=['shop_id','item_sales_level']
t0=time.time()
g=get_map_tuple(x,keys)
print time.time()-t0
'''
def add_wt_tuple(x,g,keys):
  n,b=x.shape[0],[0]*len(keys)
  t,i=[base_rate]*n,0
  rb=range(len(keys))
  while i<n:
    for j in rb: b[j]=x[keys[j]][i]
    a=tuple(b)
    if a in g:t[i]=g[a][1]

    i+=1
  s=''
  for a in keys: s+=a+'_'
  x[s+'wt']=t
  return
'''
keys=['shop_id','item_sales_level']
t0=time.time()
add_wt_tuple(x,g,keys)
print time.time()-t0
'''

def get_map_list(x,key):
  g,i,n={},0,x.shape[0]
  while i<n:
    s,t=x[key][i],x[target][i]
    s=string.split(s,';')
    for a in x:
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

def add_wt_list(x,g,key):
  n,k=x.shape[0],x[key]
  t,i=[base_rate]*n,0
  while i<n:
    s=k[i]
    s=string.split(s,';')
    z=[base_rate]*len(s)
    for j in range(len(s)):
      if s[j] in g: z[j]=g[s[j]][1]
    t[i]=max(z)

    i+=1
  x['list_wt_'+key]=t
  
  return
#key='shop_id'
#g=get_map(x,key)

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
  x2=pd.read_table(f2)
  ds=['context_id','context_timestamp']
  x1,x2=x1.drop(ds,axis=1),x2.drop(ds,axis=1)

  v1=set(['item_id','item_city_id','item_price_level','item_sales_level',\
          'item_collected_level','item_pv_level','user_id','user_gender_id',\
          'user_age_level','user_occupation_id','user_star_level',\
          'context_page_id','shop_id','shop_review_num_level',\
          'shop_star_level','match_cat_ct','match_prop_ct',\
          'item_category_list','item_brand_id'\
          ])

  for key in v1:
    g=get_map(x1,key)
    add_wt(x1,g,key)
    add_wt(x2,g,key)

  key='item_property_list'
  g=get_map_list(x1,key)
  add_wt_list(x1,g,key)
  add_wt_list(x2,g,key)
  
  key='item_category_list'
  g=get_map_list(x1,key)
  add_wt_list(x1,g,key)
  add_wt_list(x2,g,key)

  ds=['predict_category_property','item_property_list']
  x1,x2=x1.drop(ds,axis=1),x2.drop(ds,axis=1)

  v2=set(['item_city_id','item_price_level','item_sales_level',\
          'item_collected_level','item_pv_level','user_gender_id',\
          'user_age_level','user_occupation_id','user_star_level',\
          'context_page_id','shop_id','shop_review_num_level',\
          'shop_star_level','match_cat_ct','match_prop_ct',\
          'item_category_list','item_brand_id'\
          ])
  v2=list(v2)

  for i in range(0,len(v2)-1):
    for j in range(i+1,len(v2)):
      keys=[v2[i],v2[j]]
      g=get_map_tuple(x1,keys)
      add_wt_tuple(x1,g,keys)
      add_wt_tuple(x2,g,keys)
  ds=['item_category_list']
  x1,x2=x1.drop(ds,axis=1),x2.drop(ds,axis=1)

  x1.to_csv(f1, index=None, sep='\t')
  x2.to_csv(f2, index=None, sep='\t')
  return

t0=time.time()
add_wt_all(f1,f2)
print time.time()-t0
