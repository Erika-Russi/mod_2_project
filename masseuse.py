#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 10:20:43 2018

@author: griggles
"""
import pandas as pd
import os

class Masseuse:
    
    #POINT DIR @ PROJECT DIRECTORY
    def __init__(self, csv_dir='./csv/'):
        self.csv_dir = csv_dir
        self.wb_dir = csv_dir + 'wb/'
        self.init_dir = csv_dir + 'init/'
        self.extra_dir = csv_dir + 'extra/'
        self.dfs = {}
        
    # create an individual dataframe for a given csv and add it to 
    # list of dataframes - separating concerns for cleaning individual datasets
    # and eventually aggregating the data into a single table
    def build_df(self, name, direc):
        df=pd.read_csv('{}{}.csv'.format(direc, name))
        self.dfs[name] = df
        return df
        
    # WorldBankData contains individual country data and larger regions
    # here we separate these categories to avoid recounting
    def init_and_filter_countries(self):
        fertility = self.build_df('fertility', self.init_dir)
        country = self.build_df('country', self.init_dir)
        codes = country['alpha-3']        
        mf = fertility[fertility['Country Code'].isin(codes)]
        mf = mf.set_index(mf['Country Code'])
        mf = mf.drop(columns=['Indicator Name', 'Indicator Code', 'Country Name', 'Country Code'])
        mf.columns.name='Years'
        mf = mf.stack()
        mf = mf.to_frame()
        mf = mf.rename(columns={0: "fertility"})
        self.dfs['mf'] = mf
        
    def merge_hofstede_csv(self):
        mf = self.dfs['mf']
        df = self.build_df('hofstede_dimensions', self.extra_dir)
        df = df.set_index('Country Code')
        self.dfs['mf'] = mf.join(df, how='inner')        
        
    def merge_wb_csv_into_mf(self, name):
        mf = self.dfs['mf']
        df = self.build_df(name, self.wb_dir)
        df = df.set_index(df['Country Code'])
        df = df.drop(columns=['Indicator Name', 'Indicator Code', 'Country Name', 'Country Code'])
        df.columns.name='Years'
        df = df.stack()
        df = df.to_frame()
        df = df.rename(columns={0: name})
        self.dfs['mf'] = mf.merge(df, how = 'inner', on = ['Country Code', 'Years'])
        
    def write_mf_to_csv(self):
        self.dfs['mf'].to_csv(self.csv_dir + 'mf.csv')
        
    def build_data(self):
        self.init_and_filter_countries()
    
        file_names = os.listdir(self.wb_dir)
        file_names = [f[:-4] for f in file_names if f.endswith('.csv')]
        
        for name in file_names:
            self.merge_wb_csv_into_mf(name)
            
        self.merge_hofstede_csv()
            
        self.write_mf_to_csv()
        
        self.dfs['mf'] = self.dfs['mf'].convert_objects(convert_numeric=True)
        
        return self.dfs['mf']


#m = Masseuse()
#data = m.build_data()
#print(data)
        

        
            
        
    
        