import pandas as pd
from pandas.plotting import scatter_matrix, andrews_curves
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy
from collections import OrderedDict


# data = pd.read_csv("mini_products.csv",engine= 'python',sep= '\t', decimal= '.')
# print(data)
data = pd.read_csv("moyen_products.csv", engine = "python", sep= '\t|   ', decimal= '.')

nutriscore_columns = ['product_name','pnns_groups_1','nutrition-score-fr_100g', 'energy_100g', 'carbohydrates_100g', 'fat_100g', 'proteins_100g', 'sodium_100g', 'salt_100g', 'sugars_100g', 'saturated-fat_100g', 'fiber_100g', 'trans-fat_100g', 'cholesterol_100g', 'iron_100g', 'vitamin-c_100g', 'calcium_100g', 'vitamin-a_100g']
pnns1_columns = ['nutrition-score-fr_100g', 'energy_100g', 'carbohydrates_100g', 'fat_100g', 'proteins_100g', 'sodium_100g', 'salt_100g', 'sugars_100g', 'saturated-fat_100g', 'fiber_100g', 'trans-fat_100g', 'cholesterol_100g', 'iron_100g', 'vitamin-c_100g', 'calcium_100g', 'vitamin-a_100g']
nutriscoreIngre_columns = ['product_name', 'energy_100g','carbohydrates_100g', 'fat_100g', 'proteins_100g', 'sodium_100g', 'salt_100g', 'sugars_100g', 'saturated-fat_100g', 'fiber_100g', 'trans-fat_100g', 'cholesterol_100g', 'iron_100g', 'vitamin-c_100g', 'calcium_100g', 'vitamin-a_100g']
ingredients_columns = [ 'energy_100g', 'carbohydrates_100g', 'fat_100g', 'proteins_100g', 'sodium_100g', 'salt_100g', 'sugars_100g', 'saturated-fat_100g', 'fiber_100g', 'trans-fat_100g', 'cholesterol_100g', 'iron_100g', 'vitamin-c_100g', 'calcium_100g', 'vitamin-a_100g']
ingredients_woEnergy_columns = [ 'carbohydrates_100g', 'fat_100g', 'proteins_100g', 'sodium_100g', 'salt_100g', 'sugars_100g', 'saturated-fat_100g', 'fiber_100g', 'trans-fat_100g', 'cholesterol_100g', 'iron_100g', 'vitamin-c_100g', 'calcium_100g', 'vitamin-a_100g']
ingredients_group1_columns = [ 'carbohydrates_100g', 'fat_100g', 'proteins_100g', 'sugars_100g', 'saturated-fat_100g', 'fiber_100g']
ingredients_group2_columns = [ 'sodium_100g', 'salt_100g', 'trans-fat_100g', 'cholesterol_100g', 'iron_100g', 'vitamin-c_100g', 'calcium_100g', 'vitamin-a_100g']


# nutriscore_dataFrame = data
nutriscore_dataFrame = data[data['nutrition-score-fr_100g'].notnull()]
# print("there :",data[data['product_name'].notnull()])
# print("product name data :",data['product_name'])
# print("pnns groups data :",data['pnns_groups_1'])
# nutriscore_dataFrame = data[data['nutrition-score-fr_100g'].notnull()]
# print("product name nutriscore :",nutriscore_dataFrame['product_name'])
# print("pnns group nutriscore :",nutriscore_dataFrame['pnns_groups_1'])

print(nutriscore_dataFrame[nutriscore_columns].count())
print(nutriscore_dataFrame[nutriscore_columns].dtypes)
print(nutriscore_dataFrame[nutriscore_columns].columns[:])

grouped_df_mean = nutriscore_dataFrame[nutriscore_columns].groupby('pnns_groups_1').mean()
grouped_df_median = nutriscore_dataFrame[nutriscore_columns].groupby('pnns_groups_1').median()
grouped_df_std = nutriscore_dataFrame[nutriscore_columns].groupby('pnns_groups_1').std()
grouped_df_skew = nutriscore_dataFrame[nutriscore_columns].groupby('pnns_groups_1').skew()

print(grouped_df_mean)
print(grouped_df_median)
print(grouped_df_std)
print(grouped_df_skew)

little_dico = {
  'mean' : [],
  'median' : [],
  'std' : [],
  'skew' : []
}

pnns_data = OrderedDict([
  ('nutrition-score-fr_100g' , little_dico.copy()),
  ('carbohydrates_100g' , little_dico.copy()),
  ('proteins_100g' , little_dico.copy()),
  ('sodium_100g' , little_dico.copy()),
  ('salt_100g' , little_dico.copy()),
  ('sugars_100g' , little_dico.copy()),
  ('saturated-fat_100g' , little_dico.copy()),
  ('fiber_100g' , little_dico.copy()),
  ('trans-fat_100g' , little_dico.copy()),
  ('cholesterol_100g' , little_dico.copy()),
  ('iron_100g' , little_dico.copy()),
  ('vitamin-c_100g' , little_dico.copy()),
  ('calcium_100g' , little_dico.copy()),
  ('vitamin-a_100g' , little_dico.copy()),
])
pnns_df = {}

print(pnns_data['nutrition-score-fr_100g']['mean'])
for index in grouped_df_mean.columns[:]:
  # print('current index : ', index)

  pnns_data[index]['mean'] = grouped_df_mean[index]
  pnns_data[index]['median'] = grouped_df_median[index]
  pnns_data[index]['std'] = grouped_df_std[index]
  pnns_data[index]['skew'] = grouped_df_skew[index]
  pnns_df[index] = pd.DataFrame.from_dict(pnns_data[index])
  # print(pnns_data[index])
  # print(pnns_data['nutrition-score-fr_100g'])
  # print("\n")
# print(pnns_df)
  # pieces = [mean, median, std, skew]
  # print("concat for : ", index)
  # print(pd.concat(pieces, axis=1,sort=True).groupby('pnns_groups_1'))
  # grouped_df = pd.concat(pieces, axis=1,sort=False).groupby('pnns_groups_1')
  # grouped_df.get_group(index)
  # for key, item in grouped_df:
  #     print(grouped_df.get_group(key), "\n\n")
# nutriscore_dataFrame = nutriscore_dataFrame[nutriscore_dataFrame['energy_100g'].notnull()]
# nutriscore_dataFrame = nutriscore_dataFrame[nutriscore_dataFrame['fat_100g'].notnull()]

# print(data['nutrition-score-fr_100g'].notnull())
# print(data['nutrition-score-fr_100g'].isnull().sum())
# print(data['nutrition-score-fr_100g'])
# print(nutriscore_dataFrame[nutriscore_columns])

# pd.to_numeric(nutriscore_dataFrame['energy_100g'], errors='coerce')
# pd.to_numeric(nutriscore_dataFrame['energy_100g'], errors='ignore')
#
# pd.to_numeric(nutriscore_dataFrame['fat_100g'], errors='coerce')
# pd.to_numeric(nutriscore_dataFrame['fat_100g'], errors='ignore')
# data = data.to_numeric(df['fat_100g'], errors='coerce')

cols = nutriscore_dataFrame.columns.drop('energy_100g')
nutriscore_dataFrame[cols] = nutriscore_dataFrame[cols].apply(pd.to_numeric, errors='coerce')
# nutriscore_dataFrame[cols] = nutriscore_dataFrame[cols].apply(pd.to_numeric, errors='ignore')
# nutriscore_dataFrame.loc['energy_100g'] = nutriscore_dataFrame['energy_100g'].apply(pd.to_numeric, errors='coerce')

cols = nutriscore_dataFrame.columns.drop('fat_100g')
nutriscore_dataFrame[cols] = nutriscore_dataFrame[cols].apply(pd.to_numeric, errors='coerce')
# nutriscore_dataFrame[cols] = nutriscore_dataFrame[cols].apply(pd.to_numeric, errors='ignore')
# nutriscore_dataFrame.loc['fat_100g'] = nutriscore_dataFrame['fat_100g'].apply(pd.to_numeric, errors='coerce')


print(data[nutriscore_columns].dtypes)

nutriscore_dataFrame[ingredients_columns].to_csv('/Users/martinvielvoye/Desktop/OpenClassroom/Projet 2/cleaned+analysis_data/ingredients_data.csv')

all_interesting_data = {
  'ingredient' : [],
  'NaNPercent': [],
}

all_stat_data = {
  'ingredient' : [],
  'mean': [],
  'median': [],
  'std': [],
  'skew': [],
  'pearsonr': [],
}

ten_interesting_data = {
  'ingredient' : [],
  'NaNPercent': []
}
twentyfive_interesting_data = {
  'ingredient' : [],
  'NaNPercent': []
}
# pnns_groups = nutriscore_dataFrame.groupby(nutriscore_dataFrame["pnns_groups_1"].ne(nutriscore_dataFrame["pnns_groups_1"].shift()).cumsum()).apply(list).reset_index()
# print(nutriscore_dataFrame[pnns1_columns])
# pnns_groups = nutriscore_dataFrame[nutriscore_columns].groupby(["pnns_groups_1"])
# pnns_groups = pd.DataFrame(pnns_groups)
# # print({k: list(v) for k,v in nutriscore_dataFrame.groupby("pnns_groups_1")[nutriscoreIngre_columns]})
# # print('\n inthemiddlelul \n')
# print(nutriscore_dataFrame[nutriscore_columns].groupby("pnns_groups_1"))
# print(pnns_groups)

for key, value in nutriscore_dataFrame[pnns1_columns].iteritems():
    # print('key is : ', key)
    missingNan = nutriscore_dataFrame[key].isnull().sum()*100/len(nutriscore_dataFrame[nutriscore_columns])
    if (key == 'product_name') :
      missingNan = nutriscore_dataFrame[key].isna().sum()*100/len(nutriscore_dataFrame[nutriscore_columns])
      print('been here with ', key, ' : ', nutriscore_dataFrame[key].count())
    if(key == 'pnns_groups_1'):
      missingNan = nutriscore_dataFrame[key].isna().sum()*100/len(nutriscore_dataFrame[nutriscore_columns])
      print('been here', key, ' : ', nutriscore_dataFrame[key].count())
    # print()
    # print('number of missing values : ', missingNan)
    all_interesting_data['ingredient'].append(key)
    all_interesting_data['NaNPercent'].append(missingNan)

    if(missingNan < 90):
      ten_interesting_data['ingredient'].append(key)
      ten_interesting_data['NaNPercent'].append(missingNan)


    if(missingNan < 75):
      twentyfive_interesting_data['ingredient'].append(key)
      twentyfive_interesting_data['NaNPercent'].append(missingNan)

for key, value in nutriscore_dataFrame[pnns1_columns].iteritems():

    all_stat_data['ingredient'].append(key)
    all_stat_data['mean'].append(nutriscore_dataFrame[key].mean())
    all_stat_data['median'].append(nutriscore_dataFrame[key].median())
    all_stat_data['std'].append(nutriscore_dataFrame[key].std())
    all_stat_data['skew'].append(nutriscore_dataFrame[key].skew())


    guiding_board = pd.isna(nutriscore_dataFrame[key])
    iter_array = []
    X_array = []
    iter = 0
    for bol in guiding_board:
      iter_array.append(bol)

    for values in value :
      if(iter_array[iter]) :
        X_array.append(0)
      else : X_array.append(values)
      iter += 1
    # value = value.apply(pd.to_numeric, errors='coerce')
    all_stat_data['pearsonr'].append(st.pearsonr(X_array, nutriscore_dataFrame['nutrition-score-fr_100g'])[0])


    # print()
# print(all_stat_data)
df = pd.DataFrame(all_interesting_data).sort_values(by = 'NaNPercent')
df_stat = pd.DataFrame(all_stat_data)

# print(pnns_df['vitamin-a_100g']['mean'])
# print(pnns_df['vitamin-a_100g']['mean'].Beverages)
# print(pnns_df['vitamin-a_100g']['mean']['Beverages'])
# # pnns_df['vitamin-a_100g']['mean']['Beverages'] = 20
# # pnns_df['vitamin-a_100g']['mean']['general'] = 20
#
# print(pnns_df['vitamin-a_100g']['mean']['Beverages'])
# # print(pnns_df['vitamin-a_100g']['mean']['general'])
# print(pnns_df['vitamin-a_100g']['mean'])


for index, row in df_stat.iterrows():
  if((row['ingredient'] != 'energy_100g') & (row['ingredient'] != 'fat_100g')):
    general = {
      'general_avg_scores' : row['mean':'skew']
    }
    pnns_df[row['ingredient']] = pd.concat([pnns_df[row['ingredient']].sort_values(by = 'mean'),pd.DataFrame.from_dict(general, orient='index')])

    # pnns_df[row['ingredient']]['mean']['general']= mea
    # pnns_df[row['ingredient']]['median']['general'] = med
    # pnns_df[row['ingredient']]['std']['general'] = st
    # pnns_df[row['ingredient']]['skew']['general'] = skw
# df_firsts = pd.DataFrame(all_interesting_data).sort_values(by = 'NaNPercent').head(15)
# df_ten = pd.DataFrame(ten_interesting_data).sort_values(by = 'NaNPercent')
# df_twentyfive = pd.DataFrame(twentyfive_interesting_data).sort_values(by = 'NaNPercent')

# print(df, '\n')
print(df_stat, '\n')
print("pnns try : ")
# print(pnns_df['vitamin-a_100g']['mean']['Beverages'])
# print(pnns_df['vitamin-a_100g']['mean']['general'])
# print(pnns_df['vitamin-a_100g']['mean'])
for elements in pnns_df:
  print(elements)
  print(pnns_df[elements])
  print('\n')
df_stat.to_csv('/Users/martinvielvoye/Desktop/OpenClassroom/Projet 2/cleaned+analysis_data/uni+bivariate_analysis.csv')

# group1 = nutriscore_dataFrame[ingredients_group1_columns].boxplot()
# plt.ylabel('grammes',fontsize=20)
# plt.title("Boxplots for nutriments in food",fontsize=20)
# plt.show()
# group2 = nutriscore_dataFrame[ingredients_group2_columns].boxplot()
# plt.ylabel('grammes',fontsize=20)
# plt.title("Boxplots for nutriments in food",fontsize=20)
# plt.show()
# plt.show()

# pnns_df['nutrition-score-fr_100g'].boxplot()
# pnns_df['sugars_100g'].boxplot()
# pnns_df['nutrition-score-fr_100g'].histogramme()
# scatter_matrix(pnns_df['carbohydrates_100g'], alpha = 0.3)
# plt.show()

# pnns_df['nutrition-score-fr_100g'].plot(kind = 'bar')
# plt.title("bar chart for nutrition-score",fontsize=20)
# plt.show()
#
# pnns_df['carbohydrates_100g'].plot(kind = 'bar')
# plt.title("bar chart for carbohydrates",fontsize=20)
# plt.show()

# pnns_df['sodium_100g'].plot(kind = 'bar')
# plt.title("bar chart for sodium",fontsize=20)
# plt.show()
#
# pnns_df['salt_100g'].plot(kind = 'bar')
# plt.title("bar chart for salt",fontsize=20)
# plt.show()

# pnns_df['sugars_100g'].plot(kind = 'bar')
# plt.title("bar chart for sugars",fontsize=20)
# plt.show()

grouped_df_mean['Beverages':'Fruits and vegetables'].plot(kind = 'barh', width=1.3).grid()
plt.title("Means of pnns groups (1)")
plt.show()
#
grouped_df_mean['Milk and dairy products':'unknown'].plot(kind = 'barh', width=1.3).grid()
plt.title("Means of pnns groups (2)")
plt.show()

grouped_df_std['Beverages':'Fruits and vegetables'].plot(kind = 'barh', width=1.3).grid()
plt.title("Std of pnns groups (1)")
plt.show()
#
grouped_df_std['Milk and dairy products':'unknown'].plot(kind = 'barh', width=1.3).grid()
plt.title("Std of pnns groups (2)")
plt.show()
