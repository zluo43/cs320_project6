# project: p6
# submitter: zluo43
# partner: none
# hours: 10







from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time, random
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.compose import make_column_transformer
import sqlite3
from zipfile import ZipFile
import io
import os
import re
from matplotlib.animation import FuncAnimation
from IPython.core.display import HTML



    
connection = sqlite3.connect("data/images.db")
df = pd.read_sql("""
SELECT file_name, district_name, lon, lat, water_ratio, forest_ratio, agriculture_ratio, developed_ratio
FROM sample INNER JOIN districts ON sample.district_id = districts.district_id LIMIT 400
""", connection) 
train, test = train_test_split(df, random_state=0)
df
test


#q1: What are the last 5 rows of the test dataset?
q1=test.tail()
q1




#q2: What is the relationship between developed_ratio and _____? [PLOT]
train.plot.scatter(x = 'developed_ratio', y = 'agriculture_ratio')



#q3: What are the developed_ratio predictions of a Linear model on the first 5 test points?
lr=LinearRegression()
df_prediction=test.copy()
feature= ['lat','lon','water_ratio','forest_ratio','agriculture_ratio'] #list in bracekt so it remians as a dataframe
d_ratio='developed_ratio'
lr.fit(train[feature],train[d_ratio])
df_prediction['predicted']=lr.predict(test[feature])
df_prediction.head()



#q4: How does the model score when evaluated against the test dataset?
lr.score(test[feature],test['developed_ratio'])


#q5: How do the predictions compare to the actual values? [PLOT]
df_prediction.plot.scatter(x = 'developed_ratio', y = 'predicted')


#q6: What are the coefficients for each feature in your model? [PLOT]
c=lr.coef_
print (c)
pd.Series(c,index = feature).plot.bar()


#q7:Can we beat our simple model's score of 0.756 with a more complicated model?
complex_model = Pipeline([('both',make_column_transformer((OneHotEncoder(),['district_name']),(PolynomialFeatures(degree = 2,include_bias=False),
                                                                                                ['water_ratio','forest_ratio','agriculture_ratio']),
                                    remainder = "passthrough")),
    ('lr', LinearRegression())])
complex_model.fit(train[['district_name','lat','lon','water_ratio','forest_ratio','agriculture_ratio']],
                   train['developed_ratio'])
complex_model.score(test[['district_name','lat','lon','water_ratio','forest_ratio','agriculture_ratio']],
                     test['developed_ratio'])
    
#Yes by doing this we can make it better 


#Q8: what are the mean (average) scores for simple and complex models, respectively?
simple_score = cross_val_score(
    lr,
    train[['lat','lon','water_ratio','forest_ratio','agriculture_ratio']],
    train['developed_ratio'],
    cv = 8)

complex_score = cross_val_score(
    complex_model,
    train[['district_name','lat','lon','water_ratio','forest_ratio','agriculture_ratio']],
    train['developed_ratio'],
    cv = 8)
print (simple_score.mean(),complex_score.mean())

(simple_score.mean(),complex_score.mean())


#q9: what is the standard deviation of scores for each model?
(np.std(simple_score),np.std(complex_score))



connection = sqlite3.connect("data/images.db")
df2 = pd.read_sql("""
SELECT file_name, district_name, lon, lat, water_ratio, forest_ratio, agriculture_ratio, developed_ratio
FROM sample
INNER JOIN districts ON sample.district_id = districts.district_id
""", connection)
df2


def encode_ratio_value(array):
    #"water" as code 11, "forest" as codes 41-43, "agriculture" as codes 81-82, and "developed" as codes 21-24
    water_ratio=[11]
    forest_ratio=[41,42,43]
    agriculture_ratio=[81,82]
    developed_ratio=[21,22,23,24]
    arr=array.reshape(-1,1) #vertical column dimension
    #using isin to get a boolean matrix; .sum() gives a number for all True result, and thus the ratio
    return (np.isin(arr,water_ratio).sum()/len(arr),
            np.isin(arr,forest_ratio).sum()/len(arr),
            np.isin(arr,agriculture_ratio).sum()/len(arr),
            np.isin(arr,developed_ratio).sum()/len(arr))     
#missing value starts at row 400

for i in range(400,2000):
    file_name=df2.loc[i,:][0]
    with ZipFile('data/images.zip') as zf:
        with zf.open(file_name) as f:
            buf = io.BytesIO(f.read())
            map_array = np.load(buf)
            df2.iloc[i,4:] = encode_ratio_value(map_array)        #all the features are missing, startingfrom col 4

#Q10: How many cells in all of area1234.npy contain code 52?
with ZipFile('data/images.zip') as zf:
    with zf.open('area1234.npy') as f:
        buf = io.BytesIO(f.read())
        map_array = np.load(buf)
np.isin(map_array.reshape(-1,1),52).sum()

#Q11: What are the last 5 rows of the new test dataset?
train2, test2 = train_test_split(df2, random_state=0)
test2.tail()

#Q12: what are the mean (average) scores for simple and complex models, respectively, on the larger dataset?


simple_score2 = cross_val_score(
    lr,
    train2[['lat','lon','water_ratio','forest_ratio','agriculture_ratio']],
    train2['developed_ratio'],
    cv = 8)

complex_score2 = cross_val_score(
    complex_model,
    train2[['district_name','lat','lon','water_ratio','forest_ratio','agriculture_ratio']],
    train2['developed_ratio'],
    cv = 8)
#print (simple_score.mean(),complex_score.mean())

(simple_score2.mean(),complex_score2.mean())



#map annotation 


with ZipFile("data/madison.zip") as zf:
    with zf.open("year-2001.npy") as f:
        buf = io.BytesIO(f.read())
        madison_01 = np.load(buf)
    with zf.open("year-2016.npy") as f:
        buf = io.BytesIO(f.read())
        madison_16 = np.load(buf)
        
        
#Q13: What is the shape of the 2001 Madison matrix?
madison_01.shape


#Q14: What portion of the points in Madison changed from 2001 to 2016?

madison_16.shape

diff_ratio=((madison_16 !=madison_01).sum())/(1200*1200)
diff_ratio

#Q15: What years appear in madison.zip
l = []
with ZipFile("data/madison.zip") as zf:
    file_list = zf.namelist()
    for name in file_list:
        l.append(name)
l=str(l)
numbers = re.findall('[0-9]+', l)
numbers


#Q16: How has Madison evolved over the years? [VIDEO]

from matplotlib.colors import ListedColormap

def get_usage_colormap():
    use_cmap = np.zeros(shape=(256,4))
    use_cmap[:,-1] = 1
    uses = np.array([
        [0, 0.00000000000, 0.00000000000, 0.00000000000],
        [11, 0.27843137255, 0.41960784314, 0.62745098039],
        [12, 0.81960784314, 0.86666666667, 0.97647058824],
        [21, 0.86666666667, 0.78823529412, 0.78823529412],
        [22, 0.84705882353, 0.57647058824, 0.50980392157],
        [23, 0.92941176471, 0.00000000000, 0.00000000000],
        [24, 0.66666666667, 0.00000000000, 0.00000000000],
        [31, 0.69803921569, 0.67843137255, 0.63921568628],
        [41, 0.40784313726, 0.66666666667, 0.38823529412],
        [42, 0.10980392157, 0.38823529412, 0.18823529412],
        [43, 0.70980392157, 0.78823529412, 0.55686274510],
        [51, 0.64705882353, 0.54901960784, 0.18823529412],
        [52, 0.80000000000, 0.72941176471, 0.48627450980],
        [71, 0.88627450980, 0.88627450980, 0.75686274510],
        [72, 0.78823529412, 0.78823529412, 0.46666666667],
        [73, 0.60000000000, 0.75686274510, 0.27843137255],
        [74, 0.46666666667, 0.67843137255, 0.57647058824],
        [81, 0.85882352941, 0.84705882353, 0.23921568628],
        [82, 0.66666666667, 0.43921568628, 0.15686274510],
        [90, 0.72941176471, 0.84705882353, 0.91764705882],
        [95, 0.43921568628, 0.63921568628, 0.72941176471],
    ])
    for row in uses:
        use_cmap[int(row[0]),:-1] = row[1:]
    return ListedColormap(use_cmap)


def draw(frame):
    axes.cla()
    with ZipFile("data/madison.zip") as zf:
        l = zf.namelist()
        file_name = l[frame]
        with zf.open(file_name) as f:
            buf = io.BytesIO(f.read())
            madison = np.load(buf)
    axes.imshow(madison, vmin=0, vmax=255,cmap = get_usage_colormap())
    #axes.set_title(tmp[:-4])

###
fig, axes = plt.subplots()
fa = FuncAnimation(fig, draw, frames = 7,interval=1000)
html = fa.to_html5_video()
plt.close()
HTML(html)