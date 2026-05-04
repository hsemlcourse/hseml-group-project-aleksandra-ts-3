import pandas as pd
from scipy import stats


def chi2_independence(df, feature_col, target_col='Attrition'):
    sample = df[[feature_col, target_col]].dropna()
    target_values = sample[target_col].astype(int)
    feature_values = sample[feature_col]
    contingency = pd.crosstab(feature_values, target_values)
    chi2_stat, p_value, _, _ = stats.chi2_contingency(contingency)
    return float(chi2_stat), float(p_value), 'chi2_independence'


def ttest_income_by_attrition(df, value_col='MonthlyIncome', target_col='Attrition'):
    group_stay = df.loc[df[target_col] == 0, value_col].dropna()
    group_leave = df.loc[df[target_col] == 1, value_col].dropna()
    t_stat, p_value = stats.ttest_ind(group_stay, group_leave, equal_var=False)
    return float(t_stat), float(p_value), 'welch_ttest'
