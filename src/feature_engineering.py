import pandas as pd
import os

def mean_encode(df, cols, target_col='target'):
    for col in cols:
        gr = df.groupby(col)[target_col].mean()
        gr.name = gr.name + '_mean_enc'
        df = df.merge(gr.reset_index(), how='left', right_on=col, left_on=col)
        df.drop(col, axis=1, inplace=True)
    return df

def fe(df):
    #     col_nulls_sum = df.isnull().sum()
    #     null_cols = col_nulls_sum[col_nulls_sum>0].index.values
    #     for col in null_cols:
    #         df[f'{col}_null'] = df[col].isnull()
    #     df = df.fillna(0)

    m_columns = [x[:-3] for x in df.columns if x[-2:] == 'm1']
    #     m_columns_m2 = [x+'_m2' for x in m_columns]
    #     m_columns_m3 = [x+'_m3' for x in m_columns]
    #     df.drop(m_columns_m2, axis=1, inplace=True)
    #     df.drop(m_columns_m3, axis=1, inplace=True)

    #     for col in m_columns:
    #         df[f'{col}_m1_div_m2'] = df[col+'m1'] / df[col+'m2']
    # #         df[f'{col}_m1_sub_m2'] = df[col+'m1'] - df[col+'m2']

    #     df['data_type_1_m1_div_m2'] = df['data_type_1_m1'] / df['data_type_1_m2']
    # #     df['data_type_1_m1_sub_m2'] = df['data_type_1_m1'] - df['data_type_1_m2']
    #     df['data_type_2_m1_div_m2'] = df['data_type_2_m1'] / df['data_type_2_m2']
    # #     df['data_type_2_m1_sub_m2'] = df['data_type_2_m1'] - df['data_type_2_m2']
    #     df['data_type_3_m1_div_m2'] = df['data_type_3_m1'] / df['data_type_3_m2']
    # #     df['data_type_3_m1_sub_m2'] = df['data_type_3_m1'] - df['data_type_3_m2']

    # #     train['data_type_1_m1_null'] = train['data_type_1_m1'].isnull()
    # #     train['data_type_2_m1_null'] = train['data_type_2_m1'].isnull()
    # #     train['data_type_3_m1_null'] = train['data_type_3_m1'].isnull()

    count_url_cols = [x for x in df.columns if x.startswith('count_url_category')]
    vol_app_cols = [x for x in df.columns if x.startswith('vol_app')]
    count_app_cols = [x for x in df.columns if x.startswith('count_app')]
    count_sms_cols = [x for x in df.columns if x.startswith('count_sms_source')]
    df['data_type_m1_sum'] = df['data_type_1_m1'].fillna(0) + \
                             df['data_type_2_m1'].fillna(0) + df['data_type_3_m1'].fillna(0)
    #     df['data_type_sum_m2'] = df['data_type_1_m2'].fillna(0) + \
    #         df['data_type_2_m2'].fillna(0) + df['data_type_3_m2'].fillna(0)
    #     df['data_type_sum_div'] = df['data_type_sum_m1'].divide(df['data_type_sum_m2'])


    df['count_url_sum'] = df[count_url_cols].fillna(0).sum(axis=1)
    df['vol_app_sum'] = df[vol_app_cols].fillna(0).sum(axis=1)
    df['count_app_sum'] = df[count_app_cols].fillna(0).sum(axis=1)
    df['count_sms_sum'] = df[count_sms_cols].fillna(0).sum(axis=1)
    # df['content_count_m1_div_m2'] = df['content_count_m1'] / df['content_count_m2']
    # for col in count_url_cols:
    #     df[col] = df[col].divide(df['count_url_sum'])
    # for col in vol_app_cols:
    #     df[col] = df[col].divide(df['vol_app_sum'])
    # for col in count_app_cols:
    #     df[col] = df[col].divide(df['count_app_sum'])
    # for col in count_sms_cols:
    #     df[col] = df[col].divide(df['count_sms_sum'])
        #     for col in vol_app_cols:
        #         df[col] = df[col].divide(df['data_type_m1_sum'] )

    return df

def main():
    dtypes = {'target': 'bool', 'block_flag': 'bool'}
    train = pd.read_csv(os.path.join('..', 'input', 'train_music.csv'), dtype=dtypes)
    train = mean_encode(train, ['device_type', 'manufacturer_category', 'os_category'])
    # train = fe(train)
    train.to_hdf('train.hdf', 'train')

if __name__ == '__main__':
    main()