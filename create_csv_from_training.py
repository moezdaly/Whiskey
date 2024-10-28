import os
import pandas as pd


current_path = os.getcwd()
s1_path = os.path.join(current_path, 'trainingsdatenS1')
s2_path = os.path.join(current_path, 'trainingsdatenS2')
print(s1_path)
print(s2_path)

labels = ['OK','NOK']
s1_data = []
s2_data = []
# create csv data for S1
for label in labels:
    path = os.path.join(s1_path, label)
    class_num = labels.index(label)
    for element in os.listdir(path):
        s1_data.append([element,class_num])
s1_data_df = pd.DataFrame(data=s1_data, columns=['image name', 'good or bad'])
s1_data_df.to_csv('s1 data.csv', sep=',')

# create csv data for s2
for label in labels:
    path = os.path.join(s2_path, label)
    class_num = labels.index(label)
    for element in os.listdir(path):
        s1_data.append([element,class_num])
s2_data_df = pd.DataFrame(data=s2_data, columns=['image name', 'good or bad'])
s2_data_df.to_csv('s2 data.csv', sep=',')