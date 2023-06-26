import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import KNNImputer

df = pd.read_csv("german_credit.csv")
print(df.columns)

# Missing Value Trating
def treat_missing_values(df):
    # Creating an instance of the OrdinalEncoder
    encoder = OrdinalEncoder()

    # Selecting categorical columns to be encoded
    cat_cols = ['Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Purpose', 'Risk']

    # Copying the dataframe to avoid changing the original one
    df_encoded = df.copy()

    # Encoding the categorical columns
    df_encoded[cat_cols] = encoder.fit_transform(df[cat_cols])

    # Creating an instance of the KNNImputer
    imputer = KNNImputer(n_neighbors=5)

    # Applying the imputer
    df_encoded = pd.DataFrame(imputer.fit_transform(df_encoded), columns = df.columns)

    # Decoding the categorical columns back to their original form
    df_encoded[cat_cols] = encoder.inverse_transform(df_encoded[cat_cols])
    return df

df = treat_missing_values(df)

print(df.info())