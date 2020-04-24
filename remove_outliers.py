# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 17:25:37 2020

@author: Admin
"""


def remove_outliers(df,finalDf, thresh):
    num_outliers = 0
    indices_outlier = finalDf['principal component 2'].le(-thresh) 
    num_outliers += sum(indices_outlier)
    newdf = df.drop(index = indices_outlier[indices_outlier].index)
    newdf.reset_index(drop=True,inplace=True)
    indices_outlier = finalDf['principal component 2'].ge(thresh)
    num_outliers += sum(indices_outlier)
    newdf = newdf.drop(index = indices_outlier[indices_outlier].index)
    newdf.reset_index(drop=True,inplace=True)
    return newdf, num_outliers