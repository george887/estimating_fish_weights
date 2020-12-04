import pandas as pd
import math
pi = math.pi
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler

################## Prepare data ###################
def prep_fish(df):
    '''
    Function takes in a DF. Columns are lower cased, renamed to interpret easier, converts cm to inches,
    converts, grams to lbs, creates average length and estimates area variables
    '''
    # Lower case columns
    df.columns = df.columns.str.lower()
    # Lower case species names
    df.species=df.species.str.lower()
    # Renamed length columns
    df = df.rename(columns={'length1': 'vertical_length', 'length2': 'diagonal_length', 'length3': 'cross_length'})
    # Weight converted to lbs
    df['weight'] = round(df.weight * 0.00220462,2)
    # Average lengths combined
    df['avg_lengths'] = round((df.vertical_length + df.diagonal_length + df.cross_length)/3,2)
    # Coverts lengths to inches
    df['vertical_length'] = round(df.vertical_length*0.393701,2)
    df['diagonal_length'] = round(df.diagonal_length*0.393701,2)
    df['cross_length'] = round(df.cross_length*0.393701,2)
    df['avg_lengths'] = round(df.avg_lengths*0.393701,2)
    df['height'] = round(df.height*0.393701,2)
    df['width'] = round(df.width*0.393701,2)
    # Estimating area of fish using an elipse as shape of fish
    df['est_area'] = round(pi*(df.avg_lengths /2) * (df.height /2),2)
    return df

################## Train, validate, test split #######################
def fish_split(df):
    '''This function splits the data into train, validate and test data sets
    '''
    train_validate, test = train_test_split(df, test_size=.25, random_state=123, stratify= df.species)

    train, validate = train_test_split(train_validate, test_size=.3, random_state=123, stratify=train_validate.species)
    
    print(f'The shape of the train data set is {train.shape}')
    print(f'The shape of the validate data set is {validate.shape}')
    print(f'The shape of the test data set is {test.shape}')

    return train, validate, test

################## Scaled data #######################
def add_scaled_columns(train, validate, test, scaler, columns_to_scale):
    ''' 
    This function takes in the train, validate, test, scaler and columns to scale and returns scaled
    columns
    '''
    new_column_names = [c + '_scaled' for c in columns_to_scale]
    scaler.fit(train[columns_to_scale])
    
    X_train_scaled = pd.DataFrame(scaler.transform(train[columns_to_scale]), 
                            columns=new_column_names, 
                            index=train.index)
    
    X_validate_scaled = pd.DataFrame(scaler.transform(validate[columns_to_scale]), 
                            columns=new_column_names, 
                            index=validate.index)
    
    X_test_scaled = pd.DataFrame(scaler.transform(test[columns_to_scale]), 
                            columns=new_column_names, 
                            index=test.index)
    
    return X_train_scaled, X_validate_scaled, X_test_scaled 
