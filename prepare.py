import math
pi = math.pi

def prep_fish(df):
    '''
    Function takes in a DF. Columns are lower cased, renamed to interpret easier, and creates average length 
    and estimated area variables
    '''
    # Lower case columns
    df.columns = df.columns.str.lower()
    # Lower case species names
    df.species=df.species.str.lower()
    # Renamed length columns
    df = df.rename(columns={'length1': 'vertical_length', 'length2': 'diagonal_length', 'length3': 'cross_length'})
    # Average lengths combined
    df['avg_lengths'] = round((df.vertical_length + df.diagonal_length + df.cross_length)/3,2)
    # Estimating area of fish using an elipse as shape of fish
    df['est_area'] = round(pi*(df.avg_lengths /2) * (df.height /2),2)
    return df
    