import numpy as np
import pandas as pd

def transform_features(df):
    # with age, having varying numbers is possible, so we can leave that field.
    # we use log transformations for fare - compresses the very large values
    # we use binning for SibSp and ParCh - as they are discrete counts: 0, 1-2 and 3+

    print("\n---Transforming the outlier values---\n")
    df['Fare_log'] = np.log1p(df['Fare']) # log(1+fare) -> takes ln value

    df['SibSp_bin'] = pd.cut(
        df['SibSp'],
        bins=[-1, 0, 2, df['SibSp'].max()],
        labels=['0', '1-2', '3+']
    )

    df['Parch_bin'] = pd.cut(
        df['Parch'],
        bins=[-1, 0, 2, df['Parch'].max()],
        labels=['0', '1-2', '3+']
    )

    print(df[['Fare', 'Fare_log', 'SibSp', 'SibSp_bin', 'Parch', 'Parch_bin']].head(10))
    return df
