import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('note_books/Algerian_forest_fires_cleaned_dataset.csv')
df['Classes'] = df['Classes'].apply(lambda x: 0 if 'not fire' in str(x) else 1)
df.drop(['day','month','year'], axis=1, inplace=True)

X = df.drop('FWI', axis=1)
y = df['FWI']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def remove_colinearity(data, threshold):
    co_set = set()
    corr_matrix = data.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j]) > threshold:
                col_name = corr_matrix.columns[i]
                co_set.add(col_name)
    data.drop(list(co_set), axis=1, inplace=True)

remove_colinearity(X_train, 0.85)

print('Features after collinearity removal:')
print(list(X_train.columns))
print(f'\nTotal: {len(X_train.columns)} features')
