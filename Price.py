import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("Bengaluru_House_Data.csv")
a = data.head()
print(a)
b = data.shape
print(b)
c = data.info()
print(c)
for column in data.columns:
    print(data[column].value_counts())
    print("*" * 20)

d = data.isna().sum()
print(d)
data.drop(columns=['area_type' , 'availability' , 'society' , 'balcony'] , inplace = True)
e = data.describe()
print(e)
f = data.info()
print(f)
g = data['location'].value_counts()
print(g)
data['location'] = data['location'].fillna('Sarjapur Road')
h = data['size'].value_counts()
print(h)
data['size'] = data['size'].fillna('2 BHK')
data['bath'] = data['bath'].fillna(data['bath'].median())
i = data.info()
print(i)


data['bhk'] = data['size'].str.split().str.get(0).astype(int)
data[data.bhk > 20]
j = data['bhk']
print(j)

k = data['total_sqft'].unique()
print(k)

def convertRange(x):

    temp = x.split('_')
    if len(temp) == 2:
        return (float(temp[0]) + float(temp[1]))/2
    try:
        return float(x)
    except:
        return None

data['total_sqft'] =  data['total_sqft'].apply(convertRange) 
l = data['total_sqft']
print(l)
print(data.head())

# PRICE PER SQUARE FEET

print("PICE PER SQUARE FEET")
data['price_per_sqft'] = data['price'] * 100000 / data['total_sqft']
m = data['price_per_sqft']
print(m)
print(data.describe())

print(data['location'].value_counts())

# Removing the spaces from front and at the end
data['location'] = data['location'].apply(lambda x : x.strip())
location_count = data['location'].value_counts()

# CONVERTING LOCATIONS LESS THAN 1 TO OTHERS
print(" CONVERTING LOCATIONS LESS THAN 1 TO OTHERS")
location_count_less_10 = location_count[location_count <= 10]
print(location_count_less_10) 

data['location'] = data['location'].apply(lambda x : 'other' if x in location_count_less_10 else x)
print(data['location'].value_counts())


#OUTLIER DETECTION AND REMOVAL

print("OUTLIER DETECTION")
print(data.describe())
n = (data['total_sqft'] / data['bhk']).describe()
print(n)

print("Removing the bhk having data less than 300")
data = data[((data['total_sqft'] / data['bhk']) >=300)]
print(data.describe())
print(data.shape)

#Removing OUTLIER FROM PRICE PER SQUARE FEET

def remove_outlier_sqft(df):
    df_output = pd.DataFrame()
    for key , subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)        #Mean
        st = np.std(subdf.price_per_sqft)        # Standard Deviation

        gen_df = subdf[(subdf.price_per_sqft > (m-st)) & (subdf.price_per_sqft <= (m+st))]
        df_output = pd.concat([df_output , gen_df] , ignore_index=True)
    return df_output
data = remove_outlier_sqft(data)
print(data.describe())

#REMOVING OUTLIER FROM BHK

def bhk_outlier_remove(df):
    exclude_indices = np.array([])
    for location , location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk , bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean' : np.mean(bhk_df.price_per_sqft),
                'std' : np.std(bhk_df.price_per_sqft),
                'count' : bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices , bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices , axis = 'index')

data = bhk_outlier_remove(data)
print(data.shape)

print(data)    

# NOW REMOVING THE SIZE AND PRICE PER SQUARE FEET(As we have used it for Outliers)

data.drop(columns = ['size' , 'price_per_sqft'] , inplace = True)

# CLEANED DATA
print("Cleaned Data")
print(data.head())

data.to_csv("Cleaned_data.csv")

X = data.drop(columns=['price'])
Y = data['price']

'''
# GRAPH
data.hist()-
plt.show()
'''
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.2 , random_state=0)
print(X_train.shape)
print(X_test.shape)


# APPLYING LINEAR REGRESSION

column_trans = make_column_transformer(
    (OneHotEncoder(sparse=False), ['location']),
    remainder='passthrough'
)

scaler = StandardScaler()
lr = LinearRegression()

pipe = make_pipeline(column_trans, scaler, lr)
pipe.fit(X_train, Y_train)

Y_pred_lr = pipe.predict(X_test)
r2 = r2_score(Y_test, Y_pred_lr)

print("R-squared score:", r2)

# Applying Lasso

lasso = Lasso()
pipe = make_pipeline(column_trans , scaler , lasso)
pipe.fit(X_train , Y_train)
Y_pred_lasso = pipe.predict(X_test)
r3 = r2_score(Y_test , Y_pred_lasso)
print(r3)

# Applying Ridge
ridge = Ridge()
pipe = make_pipeline(column_trans , scaler , ridge)
pipe.fit(X_train , Y_train)
Y_pred_ridge = pipe.predict(X_test)
r4 = r2_score(Y_test , Y_pred_ridge)
print(r4)


print("Linear Regression :" , r2)
print("Linear Regression :" , r3)
print("Linear Regression :" , r4)
import pickle
pickle.dump(pipe, open('RidgeModel.pkl' , 'wb'))