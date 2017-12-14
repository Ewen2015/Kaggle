import os 
import pandas as pd 
import numpy as np
import operator
import datetime

wd = '/Users/ewenwang/OneDrive/IBM/Project_ICBC/code_wu'
os.chdir(wd)
datafile = 'LoanStats_clean.csv'
data = pd.read_csv(datafile, header=0, encoding='latin-1', low_memory=False)

data.select_dtypes(include=['object']).iloc[:,0:20].head(4)

def makeTarget(dataframe):
    """ Setup target of the model."""
    target = data['loan_status1']
    data['loan_status'] = target
    data = data.drop(['loan_status1'], 1)
    return None


## target 
target = data['loan_status1']
data['loan_status'] = target
data = data.drop(['loan_status1'], 1)

## string
# map
data['term'] = data['term'].map({' 36 months': 0, ' 60 months': 1})

# str.strip().astype()
#data['int_rate'] = data['int_rate'].str.strip('%').astype(float)
#data['revol_util'] = data['revol_util'].str.strip('%').astype(float)
data['zip_code'] = data['zip_code'].str.strip('xx').astype(int)

# str.extract('(\d+)').astype(float)
data['emp_length'] = data['emp_length'].str.extract('(\d+)', expand=True).astype(float)

# titles
senior = ['Manager', 'Director', 'Senior', 'manager', 'Supervisor', 'Lead', 'Sr.', 'Officer', 'Sr',
         'supervisor', 'Administrator','Management', 'Executive', 'VP', 'Vice', 'President', 'Chief',
         'director', 'Admin', 'Administrative', 'Director,' 'MANAGER', 'lead', 'officer','Leader',
         'Manager,', 'Mgr', 'Head', 'associate', 'Associate', 'leader', 'Partner', 'Manger', 
         'SR']
middle = ['Coordinator', 'Operations', 'Consultant', 'operator', 'Operator', 'consultant', 
         'Representative', 'coordinator', 'Advisor',  'Counselor', 'Instructor', 'District', 'Architect', 
          'Planner', 'Technologist', 'Master', 'Therapist', 'therapist', 'Professor', 'Investigator', 
         'Coach']
junior = ['Specialist', 'Analyst', 'Assistant', 'Sales', 'Engineer', 'Technician', 'Support', 'specialist', 
          'Account', 'service', 'technician', 'Clerk', 'Nurse','assistant', 'Maintenance', 'driver', 'Driver', 
          'clerk', 'Client', 'Staff', 'Worker', 'HR', 'Teacher', 'Designer', 'nurse', 'worker', 'Accountant', 
          'Inspector', 'agent', 'teacher', 'Member', 'Trainer', 'Secretary', 'Auditor', 'Sergeant', 'Processor', 
          'customer', 'SPECIALIST', 'Banker', 'Student', 'ASSISTANT']

#replacing all titles with 3, 2, 1, 0
def replace_titles(title):
    x = str(title['emp_title'])
    x = x.split()
    if any(i in senior for i in x):
        return 3
    elif any(i in middle for i in x):
        return 2
    elif any(i in junior for i in x):
        return 1
    else:
        return 0

data['emp_title'] = pd.DataFrame(data['emp_title']).apply(replace_titles, axis = 1)


## categories
# map
data['grade'] = data['grade'].map({
    'A': 1,
    'B': 2,
    'C': 3,
    'D': 4,
    'E': 5,
    'F': 6,
    'G': 7})

data['sub_grade'] = data['sub_grade'].map({
    'A1': 10,
    'A2': 12,
    'A3': 14,
    'A4': 16,
    'A5': 18,
    'B1': 20,
    'B2': 22,
    'B3': 24,
    'B4': 26,
    'B5': 28,
    'C1': 30,
    'C2': 32,
    'C3': 34,
    'C4': 36,
    'C5': 38,
    'D1': 40,
    'D2': 42,
    'D3': 44,
    'D4': 46,
    'D5': 48,
    'E1': 50,
    'E2': 52,
    'E3': 54,
    'E4': 56,
    'E5': 58,
    'F1': 60,
    'F2': 62,
    'F3': 64,
    'F4': 66,
    'F5': 68,
    'G1': 70,
    'G2': 72,
    'G3': 74,
    'G4': 76,
    'G5': 78,
    })

data['pymnt_plan'] = data['pymnt_plan'].map({'n': 0, 'y': 1})
data['initial_list_status'] = data['initial_list_status'].map({'f': 0, 'w': 1})
data['verification_status_joint'] = data['verification_status_joint'].map({'nan': 0, 'Not Verified': 1})

# pd.get_dummies
cates = ['home_ownership', 'verification_status', 'purpose', 'addr_state', 'title', 'application_type']

features = pd.DataFrame()
for cate in cates:
    features = pd.get_dummies(data[cate])
    data = data.join(features)
    data = data.drop([cate], 1)

## datetime
datetime = ['issue_d', 'last_pymnt_d', 'last_credit_pull_d', 'earliest_cr_line', 'next_pymnt_d']

def date2delta(datetime):
    date = pd.to_datetime(datetime)
    delta = date - date.min()
    return delta.dt.days

data[datetime] = data[datetime].apply(date2delta, axis = 0)



## 

.isnull().all()
data.select_dtypes(include=['object']).iloc[:,0:20].head(4)

outfile = 'dataset_fe.csv'
data.to_csv(outfile, index=False)


