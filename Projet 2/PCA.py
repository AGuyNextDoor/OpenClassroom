import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA


df = pd.read_csv('cleaned+analysis_data/ingredients_data.csv', engine = "python", sep= ',', decimal= '.')
df = df.drop([df.columns[0]], axis = 'columns')

features = df.columns[:]


# df['label'].replace(0, 'Benign',inplace=True)
# df['label'].replace(1, 'Malignant',inplace=True)

x = df.loc[:, features].values
for values in x[0]:
  if(isinstance(values, str)) :
    print('caught this one : ', values)
print("try 1")

print(x.shape, '\n', x)
print(x[0])

for arrays in x:
  i = 0
  for values in arrays:
    arrays[i] = np.nan_to_num(values)
    i += 1

print("try 2")

print(x.shape, '\n', x)

print(x[0])
min_max_scaler = MinMaxScaler()
x = StandardScaler().fit_transform(x) # normalizing the features

df_normalized = pd.DataFrame(x, columns = features )
print(df_normalized)
print(np.mean(x),np.std(x))

pca_nutriments = PCA(n_components=2)
principalComponents_nutriments = pca_nutriments.fit_transform(x)

df_pca = pd.DataFrame(data = principalComponents_nutriments
             , columns = ['principal component 1', 'principal component 2'])

print(df_pca)

print('Explained variation per principal component: {}'.format(pca_nutriments.explained_variance_ratio_))

# plt.figure()
# plt.figure(figsize=(10,10))
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=14)
# plt.xlabel('Principal Component - 1',fontsize=20)
# plt.ylabel('Principal Component - 2',fontsize=20)
# plt.title("Principal Component Analysis of Breast Cancer Dataset",fontsize=20)
# targets = ['energy_100g', ' saturated-fat_100g']
# colors = ['r', 'g']
# for target, color in zip(targets,colors):
#     indicesToKeep = features == target
#     print(indicesToKeep)
#     plt.scatter(df_pca.loc[indicesToKeep, 'principal component 1']
#                , df_pca.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)
#
# plt.legend(targets,prop={'size': 15})
#
# plt.show()
