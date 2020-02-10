
# coding: utf-8

# # script for customer classification on first order

# In[7]:


import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from joblib import load


# input : .csv (separator = comma) file with one row per order line and following data :
# Customer ID, InvoiceNO, InvoiceDate, StockCode, Quantity, UnitPrice

# returns a csv file (separator = comma) with CustomerID and his predicted group
# if a customer has multiple orders in the sequence, a prediction is calculated for each of the customer's invoice 
# and his predicted group is the most frequent value among these predictions

# the input customer timeserie is supposed to be clean.



# does nothing, but is here to collect numerical columns
class nothing(BaseEstimator, TransformerMixin):
       
    def fit(self, X, y=None):       
        
        return self
    
    def transform(self, X):          
    
        return X
    
    
class Aggregator(BaseEstimator, TransformerMixin):
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = pd.DataFrame(X)

        X = X.rename(columns = {0 :'InvoiceNo', 1:'Quantity', 2:'UnitPrice',3:'CustomerID' })
        X['amount'] = X.UnitPrice*X.Quantity
        X['InvoiceNo'] =  X['InvoiceNo'].astype('float64')
        X['InvoiceNo'] =  X['InvoiceNo'].astype('int')
        X['Quantity'] = X['Quantity'].astype('float64')
        X['UnitPrice'] = X['UnitPrice'].astype('float64')
        aggregations = dict()
        for col in range(4, X.shape[1]-1) : # 'amount' is the last column
            aggregations[col] = 'first'

        aggregations.update({ 'CustomerID' : 'first','amount' : 'sum','Quantity' : 'mean', 'UnitPrice' : 'mean'})

        # aggregating all basket lines
        result = X.groupby('InvoiceNo').agg(aggregations)


        # add number of lines in the basket
        result['lines_nb'] = X.groupby('InvoiceNo').size()

        return result
    
    
# read input file
X = pd.read_csv('new_invoices.csv')

# remove article returns (should not be prensent in a first invoice)
X = X[X['Quantity']>0]

# pre-process invoice rows
preproccessor = load('preproc.joblib')
X_preprocessed = preproccessor.transform(X)

# predict customer group
model = load('xgb_model.joblib') 
y_pred = model.predict(X_preprocessed)

# write results
results = X_preprocessed[['CustomerID']]
results['group'] = y_pred
# [0] because mode() could return multiple values 
# in the case where multiple values have the same nb of occurences
results = results.groupby('CustomerID')['group'].apply(lambda x : x.mode()[0] )
results.to_csv('customers_group.csv')




# In[8]:


get_ipython().system('jupyter nbconvert --to python prediction_3groups_xgboost.ipynb')

