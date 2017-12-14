"""
Feature Engineering

Feature engineering is the process of using domain knowledge of the data to create features that make 
machine learning algorithms work. Feature engineering is fundamental to the application of machine learning, 
and is both difficult and expensive. The need for manual feature engineering can be obviated by automated 
feature learning.

Feature engineering is an informal topic, but it is considered essential in applied machine learning.

Coming up with features is difficult, time-consuming, requires expert knowledge. "Applied machine learning" 
is basically feature engineering.
												— Andrew Ng, Machine Learning and AI via Brain simulations
"""
import os
import pandas as pd
import numpy as np
import operator
import datetime

wd = '/Users/ewenwang/OneDrive/IBM/Project_ICBC/code_wu'
os.chdir(wd)

# titles
senior = ['Manager', 'Director', 'Senior', 'manager', 'Supervisor', 'Lead', 'Sr.', 'Officer', 'Sr',
          'supervisor', 'Administrator', 'Management', 'Executive', 'VP', 'Vice', 'President', 'Chief',
          'director', 'Admin', 'Administrative', 'Director,' 'MANAGER', 'lead', 'officer', 'Leader',
          'Manager,', 'Mgr', 'Head', 'associate', 'Associate', 'leader', 'Partner', 'Manger',
          'SR']
middle = ['Coordinator', 'Operations', 'Consultant', 'operator', 'Operator', 'consultant',
          'Representative', 'coordinator', 'Advisor',  'Counselor', 'Instructor', 'District', 'Architect',
          'Planner', 'Technologist', 'Master', 'Therapist', 'therapist', 'Professor', 'Investigator',
          'Coach']
junior = ['Specialist', 'Analyst', 'Assistant', 'Sales', 'Engineer', 'Technician', 'Support', 'specialist',
          'Account', 'service', 'technician', 'Clerk', 'Nurse', 'assistant', 'Maintenance', 'driver', 'Driver',
          'clerk', 'Client', 'Staff', 'Worker', 'HR', 'Teacher', 'Designer', 'nurse', 'worker', 'Accountant',
          'Inspector', 'agent', 'teacher', 'Member', 'Trainer', 'Secretary', 'Auditor', 'Sergeant', 'Processor',
          'customer', 'SPECIALIST', 'Banker', 'Student', 'ASSISTANT']

# home ownership
cates = ['home_ownership', 'verification_status',
         'purpose', 'addr_state', 'title', 'application_type']

# datetime
DateTime = ['issue_d', 'last_pymnt_d', 'last_credit_pull_d',
            'earliest_cr_line', 'next_pymnt_d']


def replace_titles(title):
    """ Replacing all titles with 3, 2, 1, 0."""
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


def date2delta(time):
    date = pd.to_datetime(time)
    delta = date - date.min()
    return delta.dt.days


def makeTarget(dataframe):
    """ Setup target of the model."""
    target = dataframe['loan_status1']
    dataframe['loan_status'] = target
    dataframe = dataframe.drop(['loan_status1'], 1)
    return None


def makeString(dataframe):
    """ Convert strings to numbers."""
    dataframe['term'] = dataframe['term'].map(
        {' 36 months': 0, ' 60 months': 1})
    dataframe['zip_code'] = dataframe['zip_code'].str.strip('xx').astype(int)
    dataframe['emp_length'] = dataframe[
        'emp_length'].str.extract('(\d+)', expand=True).astype(float)
    dataframe['emp_title'] = pd.DataFrame(
        dataframe['emp_title']).apply(replace_titles, axis=1)
    dataframe['grade'] = dataframe['grade'].map(
        {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7})
    dataframe['sub_grade'] = dataframe['sub_grade'].map({
        'A1': 10, 'A2': 12, 'A3': 14, 'A4': 16, 'A5': 18,
        'B1': 20, 'B2': 22, 'B3': 24, 'B4': 26, 'B5': 28,
        'C1': 30, 'C2': 32, 'C3': 34, 'C4': 36, 'C5': 38,
        'D1': 40, 'D2': 42, 'D3': 44, 'D4': 46, 'D5': 48,
        'E1': 50, 'E2': 52, 'E3': 54, 'E4': 56, 'E5': 58,
        'F1': 60, 'F2': 62, 'F3': 64, 'F4': 66, 'F5': 68,
        'G1': 70, 'G2': 72, 'G3': 74, 'G4': 76, 'G5': 78,
    })
    dataframe['pymnt_plan'] = dataframe['pymnt_plan'].map({'n': 0, 'y': 1})
    dataframe['initial_list_status'] = dataframe[
        'initial_list_status'].map({'f': 0, 'w': 1})
    dataframe['verification_status_joint'] = dataframe[
        'verification_status_joint'].map({'nan': 0, 'Not Verified': 1})
    return None


def makeDummy(dataframe):
    """ Convert categories to dummies. """
    features = pd.DataFrame()
    for cate in cates:
        features = pd.get_dummies(dataframe[cate])
        dataframe = dataframe.join(features)
        dataframe = dataframe.drop([cate], 1)
    dataframe[DateTime] = dataframe[DateTime].apply(date2delta, axis=0)
    return None


def featureEngineering(to_csv=False):
    """ Combine all functions and can be called from other scripts. """
    datafile, outfile = 'LoanStats_clean.csv', 'dataset_fe.csv'
    data = pd.read_csv(datafile, header=0, encoding='latin-1', low_memory=False)
    st = datetime.datetime.now()
    makeTarget(data)
    makeString(data)
    makeDummy(data)
    print('Time Cost(Feature Engineering): ', datetime.datetime.now() - st)
    if to_csv:
        data.to_csv(outfile, index=False)
    return data

if __name__ == "__main__":
    datafile, outfile = 'LoanStats_clean.csv', 'dataset_fe.csv'
    data = pd.read_csv(datafile, header=0,
                       encoding='latin-1', low_memory=False)
    st = datetime.datetime.now()
    makeTarget(data)
    makeString(data)
    makeDummy(data)
    data.to_csv(outfile, index=False)
    # Time Cost:  0:01:41.371627
    print('Time Cost(Feature Engineering): ', datetime.datetime.now() - st)
