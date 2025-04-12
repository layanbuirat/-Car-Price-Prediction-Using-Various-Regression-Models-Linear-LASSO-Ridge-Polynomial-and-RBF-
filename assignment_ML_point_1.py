import pandas as pd
import requests
import re
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import folium
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split


file_path = r'C:\Users\Admin\Desktop\m.csv'
df = pd.read_csv(file_path)

def clean_and_convert_prices():
    file_path = r'C:\Users\Admin\Desktop\m.csv'
    df = pd.read_csv(file_path)

   # Extract currency and numeric price using regular expressions
    def extract_currency_and_price(price_str):
        match = re.match(r"([A-Z]+)\s?([\d,]+\.?\d*)", str(price_str)) #to seprate notation from numric price
        if match:
            currency = match.group(1)
            try:
                price = float(match.group(2).replace(',', ''))
            except ValueError:
                price = None
            return currency, price
        else:
            return None, None

    # Apply extraction to the price column
    df[['Currency', 'Numeric_Price']] = df['price'].apply(
        lambda x: pd.Series(extract_currency_and_price(x))
    )

    #  Fetch exchange rates using an API 
    try:
        response = requests.get('https://api.exchangerate-api.com/v4/latest/USD')
        exchange_rates = response.json()['rates']
    except:
        
        print("API request failed. Using manual exchange rates.")
      

    # Define a function to convert prices to USD
    def convert_to_usd(row):
        currency = row['Currency']
        price = row['Numeric_Price']
        if currency in exchange_rates and price is not None:
            return price * 1/exchange_rates[currency]  #Converting prices into dollars
        else:
            return None

    #  Apply the conversion
    df['Price_in_USD'] = df.apply(convert_to_usd, axis=1)

    # Keep the original price if conversion is not possible 
    df['price'] = df['Price_in_USD'].combine_first(df['price']) # kept this without dropping row to ensure it convert correctly to USD

    # Drop temporary columns used for conversion
    df.drop(columns=['Currency', 'Numeric_Price', 'Price_in_USD'], inplace=True)
    

    

    # Ssave back to file the new prices in dollars 
    df.to_csv(file_path, index=False)
     
   


#data cleaning 

def handle_missing_data():
    data = pd.read_csv(file_path)
    
    # Convert all columns that should be numeric to actual numeric types (non-numeric will become NaN)
    numeric_cols = data.select_dtypes(include='number').columns
    if not numeric_cols.empty:
        data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')
    else:
        print("No numeric columns found.")
    
    total_cells = data.size
    missing_cells = data.isnull().sum().sum()
    total_samples = len(data)
    samples_with_missing = data.isnull().any(axis=1).sum()
    percent_samples_with_missing = (samples_with_missing / total_samples) * 100
    
    print(f"Total cells: {total_cells}")
    print(f"Missing cells: {missing_cells}")
    print(f"Total samples: {total_samples}")
    print(f"Samples with missing data: {samples_with_missing} ({percent_samples_with_missing:.2f}%)")
    
    # Impute missing values for numeric columns if they exist
    if not numeric_cols.empty:
        num_imputer = SimpleImputer(strategy='mean')
        data_imputed_num = pd.DataFrame(num_imputer.fit_transform(data[numeric_cols]))
        data_imputed_num.columns = numeric_cols
        data[numeric_cols] = data_imputed_num
    else:
        print("Skipping numeric imputation due to no numeric columns.")
    
    # Impute missing values for categorical columns
    cat_cols = data.select_dtypes(include='object').columns
    if not cat_cols.empty:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        data_imputed_cat = pd.DataFrame(cat_imputer.fit_transform(data[cat_cols]))
        data_imputed_cat.columns = cat_cols
        data[cat_cols] = data_imputed_cat
    else:
        print("Skipping categorical imputation due to no categorical columns.")
    
    try:
        data.to_csv(file_path, index=False)
        print("Missing values handled by imputation.")
        document_missing_values(data)
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")



def document_missing_values(data):
    missing_count = data.isnull().sum()
    missing_percentage = (missing_count / len(data)) * 100
    missing_data = pd.DataFrame({'Missing Count': missing_count, 'Missing Percentage': missing_percentage})
    missing_data = missing_data[missing_data['Missing Count'] > 0]
    print("Missing Values Documentation:")
    print(missing_data)


def one_hot_encoding(feature):

    data = pd.read_csv(file_path)

    if feature in data.columns:
        unique_values = data[feature].unique()
        print(f"Unique values in the '{feature}' column:")
        print(unique_values)
        
        original_column = data[[feature]].copy()
        
        # Perform OneHotEncoding
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform="pandas")
        ohe_transformed = ohe.fit_transform(data[[feature]])
        
        
        ohe_transformed.columns = [f"{feature}_{col}" for col in ohe_transformed.columns]
        
        # Drop the original column and concatenate the transformed data
        data = data.drop(columns=[feature])
        data = pd.concat([data, original_column, ohe_transformed], axis=1)
        
        # Save the modified dataset back to the same file
        data.to_csv(file_path, index=False)
        print(f"One-hot encoding applied to the '{feature}' column, and it is retained. Here's some of the modified data:")
        print(data.head())
    else:
        print(f"The '{feature}' column is not found in the dataset.")




###########################################################################################################################



# Split into 60% training and 40% (temporary) data
train_data, temp_data = train_test_split(df, test_size=0.4, random_state=42)

# Now split the temporary data into 50% validation and 50% test
validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

def __init__():
    # Run the function to handle missing data
    data = pd.read_csv(file_path)
    handle_missing_data()
    
    document_missing_values(data)
    # Print the sizes of each dataset
    print(f"Training data size: {len(train_data)}")
    print(f"Validation data size: {len(validation_data)}")
    print(f"Test data size: {len(test_data)}")
    one_hot_encoding("brand")
    one_hot_encoding("country")

clean_and_convert_prices()
__init__()