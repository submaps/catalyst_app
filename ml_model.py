import pickle
import pandas as pd
import random


# data paths
activ_pkl = 'data/pkl/activ.pkl'
activ_center_pkl = 'data/pkl/activ_center_scale.pkl'
activity_csv = 'data/activity_bears.csv'


atac1_pick = 'data/pkl/atactic_mod1.pkl'
atac1_center_pkl = 'data/pkl/atactic_center_scale.pkl'

atac2_mod_pkl = 'data/pkl/atactic_mod2.pkl'
atack2_center_pkl = 'data/pkl/atactic_center_scale.pkl'


atack3_mod_pkl = 'data/pkl/atactic_mod3.pkl'
atac_csv = "data/atactic_dsbears.csv"
atac_center_pkl = 'data/pkl/atactic_center_scale.pkl'


def get_act_true_act_pred(num=100):
    # get predicted
    if num is None:
        num = random.randint(1, 100)

    with open(activ_pkl, 'rb') as input:
        act_model = pickle.load(input)
        act_features = ['f24**2', 'f7_3H_mean**4', 'f4_6H_max**3', 'f24_1H_mean**3', 'f14**4', 'f24_6H_mean**3',
                        'f14_3H_mean**2', 'f24_3H_max**2', 'f14_1H_max**4', 'f25_3H_min**3', 'f7_3H_median**2',
                        'f25_3H_max**2', 'f12_6H_max**4', 'f7_1H_min**2', 'f25_1H_mean**3', 'f14_1H_min**4',
                        'f4_6H_min**3', 'f12_1H_mean**3', 'f12_3H_min**4', 'f14_1H_median**4', 'f12_3H_max**3',
                        'f25_3H_max**3', 'f4**4', 'f4_3H_mean**4', 'f4_1H_max**3', 'f14_3H_max**3', 'f6_3H_mean**4',
                        'f14_3H_max**2', 'f25_6H_median**4', 'f14_6H_mean**3', 'f25_6H_min**4', 'f12_1H_min**3',
                        'f25_1H_max**4', 'f14_6H_max**2', 'f7_6H_mean**3', 'f24_6H_mean**4', 'f4_3H_min**4',
                        'f14_6H_max**3', 'f7_3H_max**4', 'f7_1H_median**3', 'f4_6H_median**3', 'f6_1H_median**4',
                        'f12_1H_max**4', 'f25_6H_median**2', 'f4_6H_mean**4', 'f12_3H_median**2', 'f4_1H_min**3',
                        'f12_6H_min**2', 'f25_3H_median**4', 'f24_6H_max**2', 'f7_6H_max**2', 'f12_6H_mean**4',
                        'f6_1H_mean**4', 'f12_6H_median**3', 'f6**4', 'f7_1H_max**4', 'f24_1H_median**2',
                        'f12_1H_median**4', 'f25_6H_max', 'f4_3H_median**3', 'f24_1H_max**3', 'f24_1H_mean**2',
                        'f7_1H_mean**2', 'f12_3H_median**3', 'f12_1H_median**3', 'f24_3H_max**3', 'f24**3',
                        'f14_6H_mean**2', 'f24_3H_min**4', 'f12_3H_max**2', 'f24_6H_min', 'f25_1H_min**3',
                        'f25_1H_median**3', 'f12_6H_mean', 'f25_6H_mean**2', 'f12_3H_mean**3', 'f25_3H_mean**4',
                        'f12**3', 'f25_1H_median**2', 'f24_1H_max**4', 'f12**4', 'f12_3H_mean**4', 'f25_6H_min**2',
                        'f24_1H_min**2', 'f25_1H_max**3', 'f7_1H_median**4', 'f25**3', 'f7**4', 'f4_3H_max**3',
                        'f4_1H_mean**4', 'f4_3H_max**2', 'f4_1H_mean**3']
        act_std = 5.614206411116623
        act_mean = 33.83998261719136

        with open(activ_center_pkl, 'rb') as input:
            act_center = pickle.load(input)
            act_scale = pickle.load(input)

    activ_df = pd.read_csv(activity_csv)
    activ_trn = (activ_df.iloc[:, 2:].values - act_center) / act_scale
    exclude_list = list(set(list(activ_df.columns[2:])) - set(act_features))
    excl_list = list(map(lambda x: False if x in exclude_list else True, activ_df.columns[2:]))
    act_preds = act_model.predict([activ_trn[num, excl_list]]) * act_std + act_mean
    act_true = activ_df['activity'][num]
    act_pred = act_preds[0]
    return act_true, act_pred


def get_atac1_true_atac1_pred(num=1):
    # get predicted
    if num is None:
        num = random.randint(1, 100)

    with open(atac1_pick, 'rb') as input:
        atac_model1 = pickle.load(input)
        atac_features_1 = ['f4_6H_max', 'f25_6H_median**3', 'f12_6H_mean**3', 'f4_1H_min**3', 'f34_3H_min**3',
                           'f12_6H_mean**4', 'f4**2', 'f12**2', 'f34_6H_median**3', 'f34_3H_max**4', 'f25_1H_median**4',
                           'f25_1H_mean**4', 'f38_1H_max', 'f34_3H_mean**3', 'f34_1H_mean**3', 'f8_6H_min',
                           'f25_6H_mean**3', 'f34_6H_mean**3', 'f4_6H_mean**3', 'f34_1H_min**2', 'f34_1H_max**3',
                           'f25_3H_median**2', 'f34_3H_median**4', 'f25_3H_min**2', 'f12_6H_max**4', 'f34_1H_median**3',
                           'f34_1H_median**2', 'f25_1H_min**2', 'f12_3H_median**2', 'f12_6H_mean**2',
                           'f25_6H_median**2', 'f25_3H_mean**4', 'f25_6H_max**4', 'f25_1H_min**3', 'f25_6H_max**2',
                           'f25_1H_median', 'f34_3H_median**3', 'f12_3H_median', 'f4_6H_min**3', 'f25_6H_mean**2',
                           'f34_1H_min**3', 'f4_6H_median**3', 'f34_6H_mean**2', 'f34_1H_mean**2', 'f12_6H_min**3',
                           'f25_1H_max', 'f4_3H_median', 'f25_3H_min**3', 'f4_6H_max**3', 'f34_3H_mean**4',
                           'f34_3H_min**2', 'f12_6H_max**3', 'f34_1H_max**2', 'f12_3H_min**3', 'f25_3H_median**4',
                           'f41_1H_mean', 'f25_1H_mean', 'f25_6H_max**3', 'f12_3H_median**3', 'f38_3H_mean',
                           'f4_6H_median**2', 'f34_6H_median**4', 'f4_6H_max**4', 'f12_6H_min', 'f12', 'f12_3H_min**2',
                           'f25_6H_min**3', 'f34_3H_min**4', 'f12_1H_min**2', 'f4_6H_max**2', 'f25', 'f25_3H_median**3',
                           'f12_1H_mean', 'f12_1H_mean**2', 'f4_3H_max**4', 'f34_6H_median**2', 'f15_1H_min',
                           'f12_1H_min**3', 'f25**4', 'f4_1H_max**4', 'f34_3H_mean**2', 'f34_1H_min', 'f34_1H_mean**4',
                           'f34_1H_median', 'f4_3H_mean', 'f34_6H_mean**4', 'f4_3H_mean**2', 'f12_1H_max',
                           'f34_1H_median**4', 'f8', 'f12_6H_min**4', 'f34_3H_max**3', 'f4_3H_mean**3',
                           'f34_3H_median**2', 'f34_3H_min', 'f25_1H_max**4', 'f34_1H_mean', 'f25_3H_mean**2',
                           'f12_6H_min**2', 'f25_6H_min**2', 'f34_1H_max**4', 'f4**3', 'f34_6H_mean', 'f34_1H_min**4',
                           'f4_3H_max', 'f12_3H_median**4', 'f12_6H_max**2', 'f4_1H_mean', 'f12_1H_median',
                           'f12_1H_min', 'f4_3H_median**2', 'f34', 'f12_1H_mean**3', 'f4_3H_mean**4', 'f4_6H_median',
                           'f34_1H_max', 'f12_6H_mean', 'f25_6H_min', 'f12_3H_max**4', 'f12_3H_min**4', 'f4_3H_max**3',
                           'f25_1H_min**4', 'f12**3', 'f12_1H_median**4', 'f34_6H_median', 'f4', 'f34_3H_mean',
                           'f25_6H_median', 'f12_6H_median**2', 'f34_6H_max**4', 'f25_6H_max', 'f25_1H_mean**3',
                           'f25_1H_mean**2', 'f4_3H_max**2', 'f25_1H_median**3', 'f12_6H_median**3', 'f4**4',
                           'f4_1H_median', 'f25_1H_median**2', 'f12_3H_mean**4', 'f12_6H_max', 'f4_1H_max**3',
                           'f12_1H_max**2', 'f34_3H_median', 'f4_6H_min**4', 'f25_6H_mean**4', 'f4_3H_min**3',
                           'f4_1H_median**4', 'f12_3H_max**3', 'f4_1H_mean**2', 'f25_6H_median**4', 'f34**2',
                           'f4_3H_min**2', 'f41', 'f34_3H_max**2', 'f25_3H_mean**3', 'f4_1H_median**2',
                           'f4_6H_median**4', 'f4_3H_median**3', 'f4_1H_median**3', 'f8_3H_mean', 'f12_6H_median',
                           'f12_3H_min', 'f25_1H_min', 'f4_3H_min**4', 'f4_6H_min**2', 'f12_1H_median**2',
                           'f12_1H_median**3', 'f12_3H_max**2', 'f4_1H_mean**3', 'f12_6H_median**4', 'f41_3H_mean',
                           'f12_1H_mean**4', 'f4_1H_mean**4', 'f34_6H_max**3', 'f4_6H_mean**2', 'f4_3H_min',
                           'f4_6H_mean**4', 'f15_3H_median', 'f12_3H_mean**3', 'f41_3H_median', 'f12_3H_max',
                           'f4_3H_median**4', 'f34**3', 'f12_3H_mean', 'f8_1H_min', 'f25_3H_max', 'f25**3',
                           'f4_1H_min**4', 'f15', 'f15_1H_mean', 'f25_1H_max**2', 'f25_1H_max**3', 'f12_1H_max**3',
                           'f12_3H_mean**2', 'f25**2', 'f34_3H_max', 'f8_3H_median', 'f39_1H_median', 'f39_1H_mean',
                           'f4_1H_max**2', 'f15_3H_mean', 'f34_6H_max**2', 'f4_6H_min', 'f4_1H_min**2', 'f25_3H_min',
                           'f25_6H_mean', 'f34**4', 'f25_3H_mean', 'f12_1H_min**4', 'f12**4', 'f25_3H_min**4',
                           'f34_6H_max', 'f4_6H_mean', 'f12_1H_max**4', 'f41_1H_median', 'f34_6H_min', 'f8_1H_median',
                           'f25_6H_min**4', 'f4_1H_max', 'f34_6H_min**2', 'f25_3H_median', 'f8_6H_mean', 'f8_6H_median',
                           'f34_6H_min**3', 'f11_6H_median', 'f8_1H_mean', 'f25_3H_max**2', 'f39_6H_median',
                           'f15_1H_median', 'f3_6H_max', 'f11_3H_max', 'f25_3H_max**4', 'f15_6H_median', 'f8_1H_max',
                           'f4_1H_min', 'f25_3H_max**3', 'f15_6H_mean', 'f41_6H_mean', 'f8_3H_min', 'f41_6H_median',
                           'f15_1H_max', 'f15_3H_min', 'f15_6H_min']
        atac_std1 = 1.509217419829508
        atac_mean1 = 5.887898089171976

    with open(atac1_center_pkl, 'rb') as input:
        atac_center = pickle.load(input)
        # print (atac_center)
        atac_scale = pickle.load(input)
        # print (atac_scale)

    atactic_df = pd.read_csv(atac_csv)
    atactic_trn = (atactic_df.iloc[:, 4:].values - atac_center) / atac_scale

    exclude_list = list(set(list(atactic_df.columns[4:])) - set(atac_features_1))
    excl_list = list(map(lambda x: False if x in exclude_list else True, atactic_df.columns[4:]))
    atac_preds1 = atac_model1.predict([atactic_trn[num, excl_list]]) * atac_std1 + atac_mean1
    atac1_pred = atac_preds1[0]
    atac1_true = atactic_df['atactic_1'][num]
    return atac1_true, atac1_pred


def get_atac2_true_atac2_pred(num=1):
    if num is None:
        num = random.randint(1, 100)

    with open(atac2_mod_pkl, 'rb') as input:
        atac_model2 = pickle.load(input)
        atac_features_2 = ['f25_1H_median**3', 'f6_3H_mean**4', 'f6', 'f25_1H_max**3', 'f8_3H_median', 'f25_6H_mean**3',
                           'f34_1H_max**3', 'f25_1H_mean**3', 'f4_1H_median', 'f4_6H_max**3', 'f6_3H_max',
                           'f4_1H_min**3', 'f6_1H_mean**4', 'f4_3H_median', 'f15', 'f34_6H_mean**2', 'f3_1H_mean',
                           'f25_3H_min', 'f34_3H_min', 'f34_6H_median', 'f34_1H_mean', 'f34_3H_mean**2', 'f4_1H_min**4',
                           'f34_3H_median**2', 'f6_3H_median**4', 'f11_6H_median', 'f34_3H_median', 'f34_3H_mean**3',
                           'f34_6H_median**2', 'f6_6H_mean**4', 'f6_1H_min**4', 'f8_6H_min', 'f34_3H_min**2',
                           'f34_6H_mean**3', 'f4_3H_mean', 'f6_6H_mean**3', 'f4_1H_mean**3', 'f12_3H_max',
                           'f4_1H_median**2', 'f25_1H_min**3', 'f6_6H_median', 'f34_1H_max**4', 'f6_6H_max**4',
                           'f4_6H_min**3', 'f34', 'f25_3H_min**2', 'f11_6H_mean', 'f4_1H_mean**4', 'f6_3H_median**3',
                           'f4_3H_median**2', 'f12_1H_mean**2', 'f4_3H_min', 'f25**3', 'f4_1H_max**2', 'f34_1H_max**2',
                           'f25_6H_median**3', 'f4_1H_max**3', 'f25_3H_median**3', 'f3_1H_max', 'f6_1H_mean**3',
                           'f37_6H_min', 'f11_6H_max', 'f12_6H_median**2', 'f6_1H_max', 'f6_3H_mean**3', 'f6_3H_max**2',
                           'f6_1H_median**4', 'f34_6H_mean', 'f34_1H_min', 'f12_3H_min', 'f34_1H_median',
                           'f34_1H_mean**2', 'f12_6H_min', 'f34**2', 'f34_3H_median**3', 'f34_3H_mean', 'f6**2',
                           'f34_6H_median**3', 'f12_3H_median**3', 'f34_3H_mean**4', 'f34_6H_min', 'f34_6H_mean**4',
                           'f12_1H_mean', 'f4_1H_mean**2', 'f12_3H_median**2', 'f34_3H_min**3', 'f4_6H_max**2',
                           'f18_1H_max', 'f4_1H_min**2', 'f12_6H_max', 'f4_6H_mean**4', 'f15_3H_median', 'f37_6H_max',
                           'f6_6H_max**3', 'f4_1H_max', 'f4_1H_mean', 'f4_3H_mean**2', 'f12_6H_median**3',
                           'f4_1H_median**3', 'f25_6H_min**2', 'f4_6H_max**4', 'f25_6H_min', 'f34_1H_max',
                           'f12_3H_median**4', 'f25_3H_min**3', 'f6_6H_mean**2', 'f34**3', 'f4_6H_median**2',
                           'f3_1H_median', 'f25_3H_mean**3', 'f4_1H_max**4', 'f25_3H_mean**4', 'f6_6H_median**2',
                           'f6_1H_min**3', 'f25_6H_max**4', 'f12_3H_median', 'f25_1H_min**2', 'f4_6H_max',
                           'f34_1H_min**2', 'f37_3H_min', 'f6_3H_max**3', 'f34_1H_median**2', 'f34_1H_mean**3',
                           'f4_3H_median**3', 'f6_3H_median**2', 'f11_3H_mean', 'f34_6H_min**2', 'f37_1H_min',
                           'f15_3H_min', 'f34_6H_median**4', 'f34_3H_median**4', 'f39_6H_mean', 'f34_3H_min**4',
                           'f25_1H_mean**2', 'f34**4', 'f25_1H_min', 'f55_3H_min', 'f12_1H_mean**3', 'f31_3H_min',
                           'f25_1H_max**2', 'f6_3H_mean**2', 'f6_1H_max**2', 'f6_1H_mean**2', 'f25_1H_median**2',
                           'f6_1H_median**3', 'f6_6H_max**2', 'f6**3', 'f15_1H_median', 'f25_3H_median**4',
                           'f12_3H_max**2', 'f25_1H_mean', 'f34_6H_min**3', 'f6_3H_max**4', 'f34_1H_min**3',
                           'f34_1H_mean**4', 'f34_1H_median**3', 'f54_3H_min', 'f11_3H_max', 'f6_6H_median**3',
                           'f25_1H_median', 'f25_1H_max', 'f4_3H_max', 'f6_6H_mean', 'f25**4', 'f12_6H_median',
                           'f25_6H_median**4', 'f4_1H_median**4', 'f12_1H_mean**4', 'f12_6H_max**2', 'f25_6H_mean**2',
                           'f25_3H_median**2', 'f4_6H_mean**3', 'f6_1H_min**2', 'f12_3H_min**2', 'f25_3H_median',
                           'f25_1H_max**4', 'f4_1H_min', 'f4_3H_max**2', 'f12_1H_median', 'f6_3H_median', 'f6_6H_max',
                           'f25**2', 'f55_6H_min', 'f6_1H_max**3', 'f3_3H_median', 'f4_3H_median**4', 'f34_6H_min**4',
                           'f8_3H_mean', 'f34_1H_median**4', 'f6_3H_mean', 'f37_6H_median', 'f34_1H_min**4',
                           'f4_6H_median**3', 'f3_6H_median', 'f12', 'f12_6H_min**2', 'f4_3H_min**2', 'f6**4',
                           'f6_1H_mean', 'f6_1H_median**2', 'f25_1H_median**4', 'f25_3H_mean', 'f25_1H_mean**4',
                           'f25_3H_mean**2', 'f6_6H_median**4', 'f25_6H_mean**4', 'f4_3H_mean**3', 'f31', 'f15_3H_mean',
                           'f25_6H_median**2', 'f12**4', 'f37_3H_max', 'f12_6H_mean**4', 'f12_3H_max**3', 'f37_1H_max',
                           'f37_1H_mean', 'f25', 'f12_6H_max**3', 'f6_1H_max**4', 'f4_3H_max**3', 'f31_6H_min',
                           'f25_1H_min**4', 'f12**2', 'f8_1H_median', 'f11_3H_median', 'f6_1H_min', 'f25_6H_max**3',
                           'f25_6H_min**3', 'f4**4', 'f12**3', 'f11_1H_mean', 'f25_6H_mean', 'f12_6H_median**4',
                           'f4_6H_min**2', 'f41_3H_median', 'f6_1H_median', 'f34_3H_max**4', 'f15_6H_mean',
                           'f12_6H_max**4', 'f12_6H_mean**3', 'f12_3H_max**4', 'f31_3H_median', 'f41_1H_median',
                           'f25_6H_median', 'f12_6H_mean', 'f8_1H_min', 'f11_1H_median', 'f31_3H_max', 'f25_3H_max**4',
                           'f12_6H_mean**2', 'f25_3H_min**4', 'f34_3H_max**3', 'f4_3H_max**4', 'f4**3', 'f53_6H_mean',
                           'f12_3H_mean', 'f3_6H_max', 'f4_3H_mean**4', 'f4_6H_min**4', 'f8_6H_mean', 'f41_6H_median',
                           'f12_1H_median**2', 'f12_3H_min**3', 'f25_6H_max**2', 'f22_3H_min', 'f37_1H_median', 'f4',
                           'f4_6H_median', 'f37_6H_mean', 'f4**2', 'f31_6H_median', 'f34_3H_max**2', 'f39',
                           'f15_6H_median', 'f3_3H_max', 'f25_6H_max', 'f11_1H_min', 'f12_3H_mean**2', 'f34_6H_max**4',
                           'f31_6H_mean', 'f3', 'f15_1H_mean', 'f3_3H_mean', 'f4_6H_mean**2', 'f12_6H_min**3',
                           'f3_1H_min', 'f12_3H_mean**3', 'f12_3H_mean**4', 'f12_1H_max**4', 'f34_3H_max',
                           'f39_3H_mean', 'f34_6H_max**3', 'f12_1H_max**3', 'f31_3H_mean', 'f41_1H_mean',
                           'f4_3H_min**3', 'f54_6H_min', 'f12_1H_min', 'f12_1H_median**3', 'f34_6H_max**2',
                           'f15_1H_max', 'f12_1H_max**2', 'f11_1H_max', 'f25_3H_max**3', 'f4_6H_median**4',
                           'f41_3H_mean', 'f37_3H_mean', 'f8_6H_median', 'f25_6H_min**4', 'f41', 'f34_6H_max',
                           'f15_1H_min', 'f12_1H_min**2', 'f39_1H_mean', 'f53_3H_mean', 'f12_1H_median**4',
                           'f41_1H_min', 'f12_1H_max', 'f8_1H_mean', 'f4_6H_min', 'f12_3H_min**4', 'f25_3H_max**2',
                           'f25_3H_max', 'f37_3H_median', 'f11', 'f39_1H_median', 'f18_1H_min', 'f12_1H_min**3', 'f8',
                           'f4_6H_mean', 'f3_6H_mean', 'f11_3H_min', 'f11_6H_min', 'f12_6H_min**4', 'f39_3H_median',
                           'f37', 'f4_3H_min**4', 'f12_1H_min**4', 'f8_3H_max', 'f41_6H_mean', 'f8_3H_min', 'f8_1H_max',
                           'f38_1H_max', 'f38_3H_mean', 'f38_3H_median', 'f39_6H_median', 'f15_3H_max', 'f31_6H_max',
                           'f38_6H_mean', 'f15_6H_max', 'f15_6H_min']
        atac_std2 = 1.6355447204969313
        atac_mean2 = 5.8713375796178315

    with open(atack2_center_pkl, 'rb') as input:
        atac_center = pickle.load(input)
        atac_scale = pickle.load(input)

    atactic_df = pd.read_csv(atac_csv)
    atactic_trn = (atactic_df.iloc[:, 4:].values - atac_center) / atac_scale

    exclude_list = list(set(list(atactic_df.columns[4:])) - set(atac_features_2))
    excl_list = list(map(lambda x: False if x in exclude_list else True, atactic_df.columns[4:]))
    atac_preds2 = atac_model2.predict([atactic_trn[num, excl_list]]) * atac_std2 + atac_mean2
    atac2_pred = atac_preds2[0]
    atac2_true = atactic_df['atactic_2'][num]
    return atac2_true, atac2_pred


def get_atac3_true_atac3_pred(num=8):
    if num is None:
        num = random.randint(1, 100)

    with open(atack3_mod_pkl, 'rb') as input:
        atac_model3 = pickle.load(input)
        atac_features_3 = ['f34_6H_mean**3', 'f49_6H_min', 'f34_6H_median**3', 'f8_3H_median', 'f24_3H_std**4',
                           'f7_1H_median**2', 'f34_1H_min**4', 'f36_1H_std', 'f6**4', 'f16_3H_std', 'f42_6H_mean',
                           'f24_1H_std**3', 'f25_3H_median**4', 'f33_3H_std', 'f24_1H_std**4', 'f25_1H_max**4', 'f24**2',
                           'f37_3H_std', 'f11_3H_max', 'f24_6H_min**2', 'f43_1H_std**4', 'f33_3H_median', 'f7_3H_mean**4',
                           'f14_6H_mean**4', 'f6_1H_min**2', 'f24_6H_max**2', 'f42_3H_max', 'f6_3H_mean**2', 'f49_1H_min',
                           'f25_3H_max**2', 'f7_6H_median**3', 'f4_6H_std**3', 'f11_6H_min', 'f4_1H_mean**3',
                           'f24_6H_median', 'f7_6H_mean**3', 'f24_1H_min**2', 'f30_1H_mean', 'f48_6H_max', 'f24_3H_min**2',
                           'f6_6H_median**2', 'f25_1H_mean**3', 'f34_1H_min**3', 'f25_1H_min**2', 'f54_3H_std',
                           'f24_3H_max', 'f14_1H_min', 'f6_1H_max**2', 'f47_3H_max', 'f25_3H_mean**4', 'f32_6H_std',
                           'f4_1H_median**3', 'f25**2', 'f6_3H_max**2', 'f6_3H_median**2', 'f7_3H_min', 'f13_6H_min',
                           'f25_3H_min**4', 'f11_1H_max', 'f18_1H_median', 'f6_1H_mean**2', 'f13_3H_min', 'f24_1H_std**2',
                           'f25_1H_median**3', 'f34_6H_mean**2', 'f34_6H_median**2', 'f34_6H_mean**4', 'f6_1H_median**2',
                           'f34_6H_median**4', 'f7_1H_min**2', 'f37_6H_max', 'f11_6H_median', 'f14_6H_median',
                           'f6_6H_max**4', 'f4_6H_mean**3', 'f4**3', 'f9_6H_mean', 'f24_1H_mean**2', 'f38_1H_mean',
                           'f4_1H_max', 'f6_1H_median**3', 'f24_1H_median**2', 'f11', 'f24_3H_std**3', 'f38_3H_max',
                           'f24_6H_mean**2', 'f53_6H_max', 'f24_1H_max**2', 'f6_1H_mean**3', 'f4_3H_mean**4', 'f6_3H_max',
                           'f34_1H_min**2', 'f49_1H_mean', 'f23_3H_median', 'f24_3H_median', 'f24_3H_mean', 'f22_6H_mean',
                           'f6_3H_median', 'f21_1H_mean', 'f6_6H_median**3', 'f30_1H_median', 'f6_1H_max**3', 'f53_3H_max',
                           'f35_1H_std', 'f20_3H_median', 'f21_1H_median', 'f34_3H_mean**4', 'f7_1H_max**3',
                           'f7_6H_median**4', 'f41_3H_max', 'f23_3H_mean', 'f7', 'f53_1H_std', 'f6_3H_mean',
                           'f24_3H_mean**2', 'f24_3H_median**2', 'f7_6H_mean**2', 'f6_1H_min**3', 'f25_3H_mean**3',
                           'f24_6H_mean', 'f24_1H_max', 'f32_1H_std', 'f10_3H_mean', 'f43_1H_std**2', 'f14_3H_median**3',
                           'f15_1H_median', 'f24_1H_median', 'f12_3H_min', 'f14_1H_max**2', 'f24_1H_mean', 'f34_6H_mean',
                           'f8_6H_min', 'f34_6H_median', 'f44_6H_max', 'f6_6H_mean**4', 'f6**3', 'f22_3H_median',
                           'f34_1H_min', 'f25_3H_min**3', 'f7_1H_mean**2', 'f7_3H_median**4', 'f14**2', 'f14_1H_median**3',
                           'f14_3H_mean**4', 'f34_1H_max**4', 'f44', 'f25_3H_max', 'f42_3H_mean', 'f24_3H_max**2',
                           'f24_3H_min', 'f24_6H_max**3', 'f24_1H_min', 'f34_3H_mean**3', 'f24_6H_median**2', 'f25**3',
                           'f9_6H_median', 'f34_3H_median**4', 'f6_3H_mean**3', 'f48_6H_mean', 'f14_1H_mean**2',
                           'f24_6H_min', 'f6_1H_min', 'f14_1H_mean', 'f14_3H_mean**3', 'f7_6H_median**2', 'f14_1H_max',
                           'f11_1H_mean', 'f24', 'f6_6H_median', 'f42_1H_median', 'f42_1H_mean', 'f12_1H_mean**4', 'f14',
                           'f25_3H_median**3', 'f6_3H_max**3', 'f7_6H_mean**4', 'f4_6H_median**2', 'f8_1H_min',
                           'f7_1H_mean**3', 'f25_1H_median**4', 'f6_1H_max', 'f6_3H_median**3', 'f14_3H_median**2',
                           'f34_1H_max**3', 'f21_3H_median', 'f55_3H_min', 'f25_1H_mean**4', 'f6_1H_mean', 'f46_3H_max',
                           'f25_3H_max**3', 'f25_1H_max**3', 'f53_1H_max', 'f6_6H_max**3', 'f30_6H_max', 'f53_3H_std',
                           'f6_1H_median', 'f24**3', 'f33_3H_mean', 'f34_1H_mean**4', 'f6_1H_median**4', 'f24_6H_min**3',
                           'f25_6H_median**4', 'f7_1H_max**4', 'f34_3H_mean**2', 'f6_1H_mean**4', 'f4_6H_min**2',
                           'f34_3H_median**3', 'f6_6H_median**4', 'f24_6H_max', 'f50_1H_median', 'f30', 'f42_1H_min',
                           'f24_1H_min**3', 'f25_1H_min**3', 'f24_3H_min**3', 'f54_6H_std', 'f6_1H_max**4', 'f33_6H_min',
                           'f33_6H_std', 'f11_6H_mean', 'f37_1H_min', 'f12_3H_std**4', 'f6_3H_min**2', 'f14_1H_median**2',
                           'f33_1H_max', 'f7_3H_mean**3', 'f10', 'f31_1H_median', 'f12_1H_min', 'f34_1H_max**2',
                           'f6_3H_min', 'f7_1H_min', 'f6_1H_min**4', 'f37_3H_mean', 'f42_6H_median', 'f0_6H_max',
                           'f34_3H_min', 'f6_3H_min**3', 'f34_1H_median**4', 'f8', 'f34_1H_mean**3', 'f6_6H_mean**3',
                           'f25_3H_mean**2', 'f25', 'f14_6H_mean**3', 'f12**3', 'f24_1H_mean**3', 'f7_6H_mean',
                           'f24_1H_median**3', 'f24_6H_mean**3', 'f7**2', 'f20_6H_median', 'f42', 'f24_1H_max**3', 'f6**2',
                           'f34_3H_mean', 'f23_1H_median', 'f14_6H_median**2', 'f4_6H_max', 'f9_1H_min', 'f31_3H_max',
                           'f34_3H_median**2', 'f39_6H_median', 'f39_6H_min', 'f23_1H_min', 'f12_6H_min**2', 'f6_6H_min',
                           'f34_1H_max', 'f25_3H_min**2', 'f49_3H_mean', 'f10_1H_mean', 'f6_3H_mean**4', 'f24_3H_mean**3',
                           'f24_3H_median**3', 'f21_3H_mean', 'f0_3H_mean', 'f34', 'f34_1H_mean**2', 'f34_1H_median**3',
                           'f34_3H_min**2', 'f5_3H_mean', 'f11_1H_median', 'f33_1H_median', 'f42_3H_min', 'f6_3H_min**4',
                           'f33_1H_mean', 'f6_3H_max**4', 'f41_6H_max', 'f4_1H_median**4', 'f44_1H_min', 'f7_1H_median',
                           'f25_1H_mean**2', 'f42_3H_median', 'f10_6H_median', 'f7_6H_median', 'f10_1H_median',
                           'f6_3H_median**4', 'f14_1H_min**2', 'f47_6H_max', 'f7_1H_median**3', 'f34_6H_max**4',
                           'f25_1H_min', 'f18_3H_std', 'f39_1H_max', 'f25_3H_median**2', 'f24_6H_max**4', 'f25_3H_min',
                           'f34_3H_median', 'f10_3H_median', 'f25_3H_max**4', 'f53_3H_min', 'f4_1H_mean**4', 'f44_6H_mean',
                           'f24_3H_max**3', 'f25_3H_mean', 'f53_6H_std', 'f13_3H_max', 'f34**2', 'f25_1H_median**2',
                           'f34_3H_max**4', 'f49_1H_median', 'f34_1H_mean', 'f21_6H_mean', 'f34_1H_median**2', 'f25**4',
                           'f33', 'f37_3H_min', 'f23', 'f6_6H_max**2', 'f30_1H_max', 'f7_3H_median**3', 'f24_6H_median**3',
                           'f4_6H_median**3', 'f34_3H_min**3', 'f4_1H_min', 'f9_6H_max', 'f22_3H_mean', 'f39_1H_min',
                           'f33_1H_min', 'f13', 'f37_1H_max', 'f31_3H_mean', 'f39_1H_mean', 'f42_6H_min', 'f7_3H_min**2',
                           'f24_3H_std**2', 'f34_6H_min', 'f50_6H_mean', 'f49_3H_median', 'f38_1H_median', 'f7_1H_min**3',
                           'f31_3H_min', 'f34_6H_max**3', 'f37_1H_median', 'f12**4', 'f23_1H_mean', 'f33_6H_median',
                           'f4_1H_std', 'f52_1H_std**4', 'f24**4', 'f2_1H_median', 'f34**3', 'f6_6H_mean**2', 'f41_1H_min',
                           'f24_6H_min**4', 'f37_1H_mean', 'f25_6H_median**3', 'f34_1H_median', 'f34_3H_min**4',
                           'f33_3H_min', 'f13_6H_max', 'f24_3H_min**4', 'f24_1H_min**4', 'f22_6H_median', 'f52_3H_std**2',
                           'f34_3H_max**3', 'f4_1H_median**2', 'f46_6H_max', 'f4_1H_mean**2', 'f12_3H_mean**4',
                           'f47_3H_std', 'f6', 'f29_3H_median', 'f20_3H_mean', 'f14_6H_min**4', 'f7_1H_max**2',
                           'f46_3H_std', 'f13_3H_mean', 'f25_3H_median', 'f18_6H_std', 'f34_6H_min**2', 'f37_3H_max',
                           'f11_1H_min', 'f4**4', 'f5_3H_median', 'f10_1H_max', 'f23_6H_mean', 'f34**4', 'f49_6H_mean',
                           'f14_1H_max**3', 'f24_1H_mean**4', 'f34_6H_max**2', 'f2_1H_mean', 'f24_1H_median**4',
                           'f14_3H_median**4', 'f24_6H_mean**4', 'f49_3H_min', 'f24_1H_max**4', 'f4_1H_max**2',
                           'f7_1H_mean', 'f7_3H_mean**2', 'f12_1H_mean**3', 'f14_6H_max**4', 'f33_6H_mean', 'f7_6H_max**4',
                           'f15_1H_max', 'f25_1H_max**2', 'f25_1H_min**4', 'f34_3H_max**2', 'f24_3H_mean**4', 'f14**3',
                           'f24_3H_median**4', 'f14_3H_mean**2', 'f25_6H_mean**4', 'f14_3H_median', 'f25_6H_max**4',
                           'f7_3H_max**4', 'f9', 'f44_1H_median', 'f44_1H_mean', 'f13_1H_median', 'f49_6H_max',
                           'f34_6H_min**3', 'f12_3H_max**4', 'f54_6H_max', 'f34_6H_max', 'f6_6H_max', 'f52_1H_std**3',
                           'f7_1H_mean**4', 'f12_3H_max**3', 'f8_6H_mean', 'f14_6H_median**3', 'f13_1H_mean',
                           'f4_3H_mean**3', 'f7_1H_std**2', 'f13_3H_median', 'f47_3H_mean', 'f12_6H_min**3',
                           'f24_3H_max**4', 'f10_6H_mean', 'f12_1H_min**2', 'f14_1H_mean**3', 'f6_6H_mean',
                           'f14_1H_median**4', 'f10_6H_min', 'f14_6H_mean**2', 'f41_3H_mean', 'f24_6H_median**4',
                           'f4_6H_mean**4', 'f48_3H_min', 'f34_3H_max', 'f7_6H_max**3', 'f23_6H_median', 'f16_1H_std',
                           'f21', 'f12_3H_max**2', 'f14_1H_std**2', 'f34_6H_min**4', 'f30_3H_max', 'f7_3H_median**2',
                           'f5_6H_median', 'f19_6H_mean', 'f41_1H_max', 'f50_1H_min', 'f4**2', 'f5_6H_mean', 'f42_1H_max',
                           'f14_1H_median', 'f25_6H_median**2', 'f31', 'f46_3H_mean', 'f55_6H_mean', 'f4_1H_min**2',
                           'f7_3H_max**3', 'f4_6H_min', 'f12_3H_max', 'f54_6H_min', 'f25_1H_mean', 'f49_3H_std',
                           'f24_6H_std**4', 'f7_6H_max**2', 'f44_3H_max', 'f13_1H_min', 'f14_1H_std', 'f2_1H_max',
                           'f43_1H_std**3', 'f21_6H_median', 'f11_3H_mean', 'f54_6H_mean', 'f48_6H_median', 'f14_6H_max**3',
                           'f7_1H_median**4', 'f2', 'f7**3', 'f20_3H_std', 'f4_6H_mean**2', 'f25_1H_median', 'f13_6H_mean',
                           'f55_1H_min', 'f11_3H_min', 'f20_6H_mean', 'f44_1H_max', 'f44_6H_median', 'f7_3H_mean',
                           'f54_3H_max', 'f9_6H_min', 'f15_6H_min', 'f49', 'f5_6H_min', 'f14_3H_max', 'f7_3H_min**3',
                           'f12_6H_median', 'f1_1H_std', 'f7_1H_min**4', 'f25_6H_mean**3', 'f50_3H_median', 'f14_1H_min**3',
                           'f7_6H_max', 'f4_6H_median', 'f11_3H_median', 'f12_3H_std**3', 'f4_6H_max**2', 'f7_3H_max**2',
                           'f6_6H_min**2', 'f13_6H_median', 'f25_1H_max', 'f7_1H_max', 'f9_1H_mean', 'f39_3H_median',
                           'f14_6H_median**4', 'f48_3H_median', 'f12_3H_min**2', 'f8_3H_mean', 'f13_1H_max',
                           'f25_6H_min**4', 'f15_3H_median', 'f14_6H_min**3', 'f14_3H_mean', 'f37_3H_median', 'f19_6H_max',
                           'f14_1H_max**4', 'f7_3H_median', 'f4_1H_median', 'f37_6H_min', 'f48_3H_mean', 'f14_6H_mean',
                           'f37', 'f4_1H_mean', 'f5_3H_min', 'f25_6H_median', 'f49_1H_std', 'f10_1H_min', 'f25_6H_max**3',
                           'f14_6H_max**2', 'f4_1H_min**3', 'f41_3H_min', 'f52_1H_min', 'f52_1H_min**2', 'f52_1H_min**3',
                           'f49_1H_max', 'f12_1H_median**4', 'f52_1H_min**4', 'f4_6H_std', 'f15_3H_mean', 'f12_6H_max**4',
                           'f8_6H_median', 'f14**4', 'f4_1H_max**3', 'f11_6H_max', 'f22_3H_min', 'f38_3H_std', 'f3_1H_max',
                           'f7_3H_max', 'f12_6H_mean**4', 'f14_3H_max**2', 'f47_3H_median', 'f44_3H_median', 'f51_3H_min',
                           'f12_3H_mean**3', 'f12_6H_median**2', 'f2_1H_min', 'f55_3H_median', 'f49_6H_median',
                           'f44_3H_mean', 'f31_3H_median', 'f4_6H_median**4', 'f54_3H_min', 'f55_6H_min', 'f14_1H_mean**4',
                           'f20_1H_mean', 'f5_1H_max', 'f33_3H_max', 'f3_1H_mean', 'f12**2', 'f4_3H_mean**2',
                           'f4_6H_std**2', 'f46_3H_median', 'f25_6H_mean**2', 'f29_3H_mean', 'f3', 'f0_3H_max',
                           'f14_6H_std', 'f31_6H_min', 'f24_1H_std', 'f14_6H_max', 'f3_1H_median', 'f18_3H_max',
                           'f23_3H_std', 'f51_1H_min', 'f4_6H_min**3', 'f46_1H_max', 'f4_1H_min**4', 'f7_3H_min**4',
                           'f47_1H_max', 'f49_3H_max', 'f39_1H_median', 'f22_1H_std', 'f31_6H_median', 'f25_6H_min**3',
                           'f21_3H_max', 'f3_6H_max', 'f7**4', 'f4_6H_std**4', 'f55_6H_median', 'f5_1H_mean', 'f42_6H_max',
                           'f54_1H_std', 'f10_3H_max', 'f29_1H_median', 'f12_6H_median**3', 'f21_1H_max', 'f6_6H_min**3',
                           'f14_3H_max**3', 'f52_1H_std**2', 'f18_1H_std', 'f24_6H_std', 'f8_1H_median', 'f4_3H_std',
                           'f48_6H_min', 'f0_3H_std', 'f48_1H_min', 'f3_6H_min', 'f3_3H_min', 'f12_6H_max**3',
                           'f14_6H_min**2', 'f55_1H_median', 'f4', 'f40_6H_max', 'f12_1H_mean**2', 'f15_6H_median',
                           'f41_1H_median', 'f14_1H_min**4', 'f9_1H_median', 'f52_6H_std**3', 'f53', 'f18_1H_mean',
                           'f55_1H_mean', 'f31_1H_mean', 'f42_3H_std', 'f7_6H_min', 'f53_3H_median', 'f3_1H_min',
                           'f50_1H_max', 'f15_6H_mean', 'f22_1H_min', 'f25_6H_mean', 'f44_3H_min', 'f5_1H_median',
                           'f48_3H_max', 'f25_6H_max**2', 'f0_3H_median', 'f12_6H_mean**3', 'f21_1H_min', 'f29_1H_max',
                           'f9_3H_median', 'f12_6H_min', 'f22_1H_mean', 'f25_6H_min', 'f25_6H_min**2', 'f4_6H_mean',
                           'f2_3H_max', 'f52_6H_std**2', 'f12_6H_median**4', 'f30_1H_min', 'f4_1H_max**4', 'f33_6H_max',
                           'f12_6H_min**4', 'f4_3H_std**2', 'f12_3H_std**2', 'f4_6H_max**3', 'f37_6H_median', 'f48_1H_mean',
                           'f15_3H_max', 'f6_6H_min**4', 'f12_1H_min**3', 'f41_3H_median', 'f20_1H_median', 'f14_3H_max**4',
                           'f12_3H_mean**2', 'f29_1H_mean', 'f24_6H_std**3', 'f48', 'f30_3H_mean', 'f37_6H_mean',
                           'f18_6H_max', 'f12_6H_max**2', 'f51_1H_std', 'f48_1H_median', 'f29_3H_max', 'f4_3H_median**4',
                           'f52_1H_std', 'f9_3H_min', 'f4_3H_mean', 'f20_1H_min', 'f26_6H_min', 'f14_3H_min**4',
                           'f0_1H_mean', 'f5', 'f18_3H_mean', 'f31_6H_mean', 'f46_1H_median', 'f4_3H_min', 'f52_3H_std',
                           'f14_6H_min', 'f54_1H_max', 'f44_6H_min', 'f48_1H_max', 'f7_6H_min**2', 'f39_3H_min',
                           'f53_1H_mean', 'f23_3H_min', 'f55_3H_mean', 'f20_1H_std', 'f50_1H_std', 'f10_3H_min',
                           'f12_6H_std', 'f9_3H_mean', 'f38_6H_max', 'f16_6H_std', 'f12_6H_max', 'f15_1H_mean',
                           'f0_1H_median', 'f12_6H_mean**2', 'f38_3H_min', 'f21_6H_max', 'f12_3H_median**4', 'f41_6H_mean',
                           'f32', 'f12_3H_min**3', 'f51_3H_std', 'f4_3H_median**3', 'f39_6H_mean', 'f38_1H_std',
                           'f7_6H_std', 'f4_3H_max', 'f54_1H_mean', 'f25_6H_max', 'f12', 'f12_1H_median**3', 'f12_1H_max',
                           'f30_3H_median', 'f32_3H_max', 'f9_1H_max', 'f38_6H_mean', 'f37_6H_std', 'f22_3H_std',
                           'f12_3H_mean', 'f32_6H_mean', 'f8_3H_min', 'f32_1H_min', 'f29_3H_min', 'f32_6H_min',
                           'f32_6H_median', 'f3_3H_mean', 'f30_6H_mean', 'f32_1H_max', 'f0_1H_min', 'f41_1H_mean',
                           'f7_6H_min**3', 'f19_3H_max', 'f4_6H_min**4', 'f32_1H_median', 'f53_3H_mean', 'f16_3H_max',
                           'f32_1H_mean', 'f4_3H_max**2', 'f14_3H_min**3', 'f32_3H_median', 'f10_6H_max', 'f12_1H_mean',
                           'f4_3H_median**2', 'f32_3H_mean', 'f12_6H_mean', 'f5_3H_max', 'f4_6H_max**4', 'f23_6H_min',
                           'f5_1H_min', 'f20_6H_min', 'f41_6H_min', 'f23_1H_std', 'f32_6H_max', 'f12_3H_median**3',
                           'f2_3H_median', 'f4_3H_max**3', 'f5_1H_std', 'f12_6H_std**2', 'f37_1H_std', 'f12_1H_min**4',
                           'f7_3H_std', 'f7_6H_min**4', 'f50_3H_std', 'f29_1H_min', 'f32_3H_min', 'f4_3H_median',
                           'f38_3H_median', 'f30_6H_median', 'f38_6H_median', 'f3_3H_max', 'f8_6H_max', 'f53_6H_mean',
                           'f53_6H_median', 'f24_3H_std', 'f4_3H_min**2', 'f22_1H_median', 'f14_3H_min**2', 'f2_3H_mean',
                           'f44_3H_std', 'f55', 'f46_6H_std', 'f4_3H_max**4', 'f12_1H_max**2', 'f47_6H_std', 'f16_1H_max',
                           'f12_3H_median**2', 'f15', 'f5_6H_max', 'f7_1H_std', 'f32_3H_std', 'f22_6H_min', 'f20_3H_max',
                           'f3_6H_mean', 'f41_6H_median', 'f19_6H_median', 'f12_3H_std', 'f36_3H_std', 'f12_3H_min**4',
                           'f15_1H_min', 'f24_6H_std**2', 'f3_3H_median', 'f12_3H_median', 'f9_3H_max', 'f12_1H_std',
                           'f18_3H_min', 'f53_6H_min', 'f0_1H_max', 'f12_1H_median**2', 'f3_6H_median', 'f14_3H_min',
                           'f39_3H_mean', 'f46_1H_std', 'f4_3H_std**3', 'f8_3H_max', 'f44_1H_std', 'f30_3H_std', 'f20',
                           'f47_1H_std', 'f50_3H_max', 'f39', 'f12_1H_max**3', 'f41', 'f20_1H_max', 'f35_6H_median',
                           'f52_6H_std**4', 'f20_6H_max', 'f8_1H_mean', 'f49_6H_std', 'f4_3H_min**3', 'f22', 'f29',
                           'f22_6H_std', 'f38_1H_min', 'f38_3H_mean', 'f50_6H_min', 'f54_3H_mean', 'f12_1H_max**4',
                           'f8_6H_std', 'f4_3H_std**4', 'f12_1H_median', 'f15_6H_max', 'f50_3H_min', 'f18_1H_max',
                           'f4_3H_min**4', 'f0', 'f31_6H_max', 'f17', 'f14_3H_std', 'f53_1H_min', 'f48_1H_std',
                           'f18_1H_min', 'f50', 'f38_1H_max', 'f15_6H_std', 'f31_1H_max', 'f8_1H_max', 'f31_1H_min',
                           'f15_3H_min', 'f31_3H_std', 'f16_6H_max', 'f50_6H_median', 'f18_6H_median', 'f30_6H_std',
                           'f0_6H_std', 'f29_6H_std']
        atac_std3 = 1.4152851339756212
        atac_mean3 = 6.261146496815288


    with open(atac_center_pkl, 'rb') as input:
        atac_center = pickle.load(input)
        atac_scale = pickle.load(input)

    atactic_df = pd.read_csv(atac_csv)
    atactic_trn = (atactic_df.iloc[:, 4:].values - atac_center) / atac_scale

    exclude_list = list(set(list(atactic_df.columns[4:])) - set(atac_features_3))
    excl_list = list(map(lambda x: False if x in exclude_list else True, atactic_df.columns[4:]))
    atac_preds3 = atac_model3.predict([atactic_trn[num, excl_list]]) * atac_std3 + atac_mean3
    atac_3_pred = atac_preds3[0]
    atac_3_true = atactic_df['atactic_3'][num]
    return atac_3_true, atac_3_pred


def main():
    print('activity', get_act_true_act_pred())
    print('atac1', get_atac1_true_atac1_pred())
    print('atac2', get_atac2_true_atac2_pred())
    print('atac3', get_atac3_true_atac3_pred())


if __name__ == '__main__':
    main()
