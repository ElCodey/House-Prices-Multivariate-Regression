import pandas as pd
import numpy as np

def import_data():
    df = pd.read_csv("London.csv")
    return df

def drop(df):
    #Index then drop Non-London City/County
    index = df[df["City/County"]!= "London"].index
    df.drop(index, inplace=True)
    
    unneeded_cols = ["City/County", "No. of Bathrooms", "No. of Receptions", "Property Name", "Location", "Postal Code"]
    df.drop(columns=unneeded_cols, inplace=True)
    return df

def drop_na(df):
    df.dropna(inplace=True)
    df.head()
    return df

def clean_col_names(df):
    new_col_names = {"Price": "price", "House Type": "house_type", "Area in sq ft": "area",
                     "No. of Bedrooms": "rooms"}
    new_df = df.rename(columns = new_col_names)
    return new_df

def clean_values(df):
    #Change data types then split categorical column into dummy varaibles
    df["price"] = df["price"].astype(float)
    df["area"] = df["area"].astype(float)
    df["rooms"] = df["rooms"].astype(float)
    
    df = pd.get_dummies(df)
    #There is only under 10 values for these types, so dropping cols
    drop_cols = ["house_type_Duplex", "house_type_Mews", "house_type_Studio"]
    df.drop(columns= drop_cols, inplace = True)
    #Renaming cols into lowercase and no spaces
    new_col_names = {"house_type_Flat / Apartment": "flat", "house_type_House": "house", "house_type_New development": "new_dev", "house_type_Penthouse": "penthouse"}
    df.rename(columns = new_col_names, inplace = True)
    return df

def full_clean():
    #Using all functions to return clean data so I can use in jupyter
    df = import_data()
    df = drop(df)
    df = drop_na(df)
    df = clean_col_names(df)
    df = clean_values(df)
    #Saving data to import in notebook
    df.to_csv("cleaned_house_data.csv")
    
    return df
