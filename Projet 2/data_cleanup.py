import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import re

data = pd.read_csv("mini_products.csv",engine= 'python',sep= '\t', decimal= '.')
print(data)
# middle_data = pd.read_csv("moyen_products.csv", engine = "python", sep= '\t|   ', decimal= '.')

# print(data)

filtered_columns = ['product_name', 'brands', 'nutrition_grade_fr', 'carbon-footprint_100g', 'carbon-footprint-from-meat-or-fish_100g', 'nutrition-score-fr_100g']
carbon_nut_score_columns = ['carbon-footprint_100g',	'carbon-footprint-from-meat-or-fish_100g', 	'nutrition-score-fr_100g', 	'nutrition-score-uk_100g']


# Studied_data = ['nutrition-score-fr_100g', ]
# print(data[filtered_columns])
# print(middle_data[filtered_columns])
# print(data['carbon-footprint_100g'].sum())

ingredient_dataFrame = data[data.columns[68:175]].copy().drop(carbon_nut_score_columns, axis=1)
carbon_nut_score_dataFrame = data[data.columns[164:168]].copy()

# ingredient_dataFrame = middle_data[middle_data.columns[68:175]].copy().drop(carbon_nut_score_columns, axis=1)
# carbon_nut_score_dataFrame = middle_data[middle_data.columns[164:168]].copy()

# print(ingredient_dataFrame)
# print(carbon_nut_score_dataFrame)

all_interesting_data = {
  'Ingredient' : [],
  'NaNPercent': []
}

ten_interesting_data = {
  'Ingredient' : [],
  'NaNPercent': []
}
twentyfive_interesting_data = {
  'Ingredient' : [],
  'NaNPercent': []
}

carbon_data = {
  'data_type' : [],
  'NaNPercent' : []
}


# for key, value in ingredient_dataFrame.iteritems():
#     missingNan = ingredient_dataFrame[key].isnull().sum()*100/len(ingredient_dataFrame)
#     # print('number of missing values : ', missingNan)
#     all_interesting_data['Ingredient'].append(key)
#     all_interesting_data['NaNPercent'].append(missingNan)
#
#
#     if(missingNan < 90):
#       ten_interesting_data['Ingredient'].append(key)
#       ten_interesting_data['NaNPercent'].append(missingNan)
#
#
#     if(missingNan < 75):
#       twentyfive_interesting_data['Ingredient'].append(key)
#       twentyfive_interesting_data['NaNPercent'].append(missingNan)
#
#     # print()
#
# for key, value in carbon_nut_score_dataFrame.iteritems():
#     missingNan = carbon_nut_score_dataFrame[key].isnull().sum()*100/len(carbon_nut_score_dataFrame)
#     # print('number of missing values : ', missingNan)
#
#     carbon_data['data_type'].append(key)
#     carbon_data['NaNPercent'].append(missingNan)
#
#     # if(missingNan < 90):
#     #   ten_interesting_data['Ingredient'].append(key)
#     #   ten_interesting_data['NaNPercent'].append(missingNan)
#
#     # if(missingNan < 75):
#     #   twentyfive_interesting_data['Ingredient'].append(key)
#     #   twentyfive_interesting_data['NaNPercent'].append(missingNan)
#
#     # print()
#


df = pd.DataFrame(all_interesting_data).sort_values(by = 'NaNPercent')
df_firsts = pd.DataFrame(all_interesting_data).sort_values(by = 'NaNPercent').head(15)
df_ten = pd.DataFrame(ten_interesting_data).sort_values(by = 'NaNPercent')
df_twentyfive = pd.DataFrame(twentyfive_interesting_data).sort_values(by = 'NaNPercent')

df_carbon = pd.DataFrame(carbon_data).sort_values(by = 'NaNPercent')

print(df, '\n')
print(df_firsts, '\n')
print(df_ten, '\n')
print(df_twentyfive, '\n')
print(df_carbon, '\n')
