from typing import Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
# noinspection PyUnresolvedReferences
from sklearn.experimental import enable_iterative_imputer
# noinspection PyProtectedMember
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler

effective_zero = -20
final_regression_pipeline_name = 'final_regression_pipeline.joblib'


def cap_y(y: Union[pd.Series, np.array]) -> np.array:
    y_capped = y.copy()
    y_capped[y_capped < effective_zero] = effective_zero
    return np.array(y_capped)


# noinspection PyPep8Naming,PyUnusedLocal
class SpeciesAttributesTransformer(BaseEstimator, TransformerMixin):
    """A custom transformer acting on a pandas.DataFrame, which makes desired transformations and
    maintains the DataFrame data type."""

    column_constructors = (
        ('lhs_hform', lambda df: df['reactant_1_hform'] + df['reactant_2_hform']),
        ('rhs_hform', lambda df: df['product_1_hform'] + df['product_2_hform']),
        ('delta_hform', lambda df: df['rhs_hform'] - df['lhs_hform']),
        ('lhs_hform_neutral', lambda df: df['reactant_1_hform_neutral'] + df['reactant_2_hform_neutral']),
        ('rhs_hform_neutral', lambda df: df['product_1_hform_neutral'] + df['product_2_hform_neutral']),
        ('delta_hform_neutral', lambda df: df['rhs_hform_neutral'] - df['lhs_hform_neutral']),
        ('lhs_polarizability_factor',
         lambda df: pd.concat(
             (df['reactant_1_polarizability'] * df['reactant_2_charge'],
              df['reactant_2_polarizability'] * df['reactant_1_charge']),
             axis='columns'
         ).abs().sum(axis='columns')),
        ('rhs_polarizability_factor',
         lambda df: pd.concat(
             (df['product_1_polarizability'] * df['product_2_charge'],
              df['product_2_polarizability'] * df['product_1_charge']),
             axis='columns'
         ).abs().sum(axis='columns')),
        ('lhs_dipole_moment_factor',
         lambda df: pd.concat(
             (df['reactant_1_dipole_moment'] * df['reactant_2_charge'],
              df['reactant_2_dipole_moment'] * df['reactant_1_charge']),
             axis='columns'
         ).abs().sum(axis='columns')),
        ('rhs_dipole_moment_factor',
         lambda df: pd.concat(
             (df['product_1_dipole_moment'] * df['product_2_charge'],
              df['product_2_dipole_moment'] * df['product_1_charge']),
             axis='columns'
         ).abs().sum(axis='columns')),
        ('lhs_blocks_s', lambda df: df[['reactant_1_blocks_s', 'reactant_2_blocks_s']].sum(axis='columns')),
        ('lhs_blocks_p', lambda df: df[['reactant_1_blocks_p', 'reactant_2_blocks_p']].sum(axis='columns')),
        ('lhs_groups_IA', lambda df: df[['reactant_1_groups_IA', 'reactant_2_groups_IA']].sum(axis='columns')),
        ('lhs_groups_IVA', lambda df: df[['reactant_1_groups_IVA', 'reactant_2_groups_IVA']].sum(axis='columns')),
        ('lhs_groups_VA', lambda df: df[['reactant_1_groups_VA', 'reactant_2_groups_VA']].sum(axis='columns')),
        ('lhs_groups_VIA', lambda df: df[['reactant_1_groups_VIA', 'reactant_2_groups_VIA']].sum(axis='columns')),
        ('lhs_groups_VIIA', lambda df: df[['reactant_1_groups_VIIA', 'reactant_2_groups_VIIA']].sum(axis='columns')),
    )

    drop_columns = [f'{rp}_{num}_{attr}'
                    for rp in ['reactant', 'product']
                    for num in [1, 2]
                    for attr in ['name', 'hform', 'hform_neutral', 'polarizability', 'dipole_moment', 'charge',
                                 'blocks_s', 'blocks_p',
                                 'groups_IA', 'groups_IVA', 'groups_VA', 'groups_VIA', 'groups_VIIA']]

    def __init__(self):
        super().__init__()

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        for new_col, constructor in self.column_constructors:
            X[new_col] = constructor(X)
        X.drop(columns=self.drop_columns, inplace=True)
        return X


# noinspection PyPep8Naming,PyUnusedLocal
class SpeciesAttributesImputer(BaseEstimator, TransformerMixin):
    """A custom transformer which imputes nan values in numeric columns of a pandas.DataFrame,
    belonging to species attributes (as in the same value should be present for the same species).
    """

    def __init__(self, base_regressor=None, random_state: int = 42, ):
        super().__init__()
        self.base_regressor = base_regressor
        if self.base_regressor is None:
            self.imputer = IterativeImputer(random_state=random_state)
        #             self.imputer = IterativeImputer(
        #                 RandomForestRegressor(max_depth=2, max_samples=0.2, random_state=random_state),
        #                 random_state=random_state
        #             )
        elif self.base_regressor == 'avg':
            self.imputer = SimpleImputer()
        else:
            self.imputer = IterativeImputer(self.base_regressor, random_state=random_state)
        self.random_state = random_state
        self.sp_attrs = ['mass', 'charge', 'hform', 'hform_neutral', 'polarizability', 'dipole_moment',
                         'blocks_s', 'blocks_p',
                         'groups_IA', 'groups_IVA', 'groups_VA', 'groups_VIA', 'groups_VIIA']
        self.non_neg_attrs = ['polarizability']
        self.cols_to_drop_after_transform = [
            f'{rp}_{num}_{attr}'
            for rp in ['reactant', 'product']
            for num in [1, 2]
            for attr in ['mass']
        ]

    @staticmethod
    def species_attributes(X: pd.DataFrame) -> list[str]:
        prefix = 'reactant_1_'
        return [col[len(prefix):] for col in X.columns if col.startswith(prefix)]

    @staticmethod
    def species_numerical_attributes(X: pd.DataFrame) -> list[str]:
        return [attr for attr in SpeciesAttributesImputer.species_attributes(X) if attr != 'name']

    def check_consistency(self, X: pd.DataFrame) -> None:
        assert self.species_numerical_attributes(X) == self.sp_attrs, \
            f'Not containing (strictly only) species attributes: {self.sp_attrs}'

    @staticmethod
    def distinct_species_df(X: pd.DataFrame) -> pd.DataFrame:
        df_parts = []
        prefixes = [f'{rp}_{num}' for rp in ['reactant', 'product'] for num in [1, 2]]
        attrs = SpeciesAttributesImputer.species_attributes(X)
        for pref in prefixes:
            cols = [f'{pref}_{attr}' for attr in attrs]
            df_part = X.loc[:, cols]
            df_part.columns = attrs
            df_parts.append(df_part)
        species_df = pd.concat(df_parts).drop_duplicates()
        species_df.index = species_df['name']
        species_df.drop(columns=['name'], inplace=True)
        return species_df

    def fit(self, X: pd.DataFrame, y=None):
        self.check_consistency(X)
        species_dataset = self.distinct_species_df(X)
        names = species_dataset.index
        assert len(names) == len(set(names)), 'Dataset has inconsistent species attributes values!'
        self.imputer.fit(species_dataset)
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        self.check_consistency(X)
        species_dataset = self.distinct_species_df(X)
        names = species_dataset.index
        assert len(names) == len(set(names)), 'Dataset has inconsistent species attributes values!'
        imputed = self.imputer.transform(species_dataset)
        imputed_df = pd.DataFrame(imputed, columns=species_dataset.columns, index=species_dataset.index)
        for attr in self.non_neg_attrs:
            imputed_df.loc[imputed_df[attr] < 0, attr] = 0

        # fill in the imputed species attributes to X:
        X = X.copy()
        prefixes = [f'{rp}_{num}' for rp in ['reactant', 'product'] for num in [1, 2]]
        for prefix in prefixes:
            sp_names = X.loc[:, f'{prefix}_name']
            for attr in self.sp_attrs:
                col = f'{prefix}_{attr}'
                nan_mask = X.loc[:, col].isna()
                if nan_mask.any():
                    sp_names_nan = sp_names[nan_mask]
                    X.loc[sp_names_nan.index, col] = imputed_df.loc[sp_names_nan, attr].values
        # drop the species attributes which were used in fitting but no longer needed
        X.drop(columns=self.cols_to_drop_after_transform, inplace=True)
        return X


# noinspection PyPep8Naming,PyUnusedLocal
class Scaler(BaseEstimator, TransformerMixin):
    """A standard scaler transformer preserving the dataframe type.
    """
    def __init__(self):
        super().__init__()
        self.scaler = StandardScaler()
        self.columns = []

    def fit(self, X: pd.DataFrame, y=None):
        self.columns = list(X.columns)
        self.scaler.fit(X)
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        # sanity check:
        assert list(X.columns) == self.columns
        X_orig = X.copy()
        return pd.DataFrame(self.scaler.transform(X), columns=X_orig.columns, index=X_orig.index)


# noinspection PyPep8Naming
class FinalModelWrapper(BaseEstimator, RegressorMixin):
    """A custom regressor, which only transforms the vector of predicted values - caps the logarithms of rate
    coefficients to the effective zero and returns their exponents (true rate coefficients in cm3.s-1).
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):  # cap and exp:
        return 10**cap_y(self.model.predict(X))


def get_final_regression_pipeline():
    return joblib.load(final_regression_pipeline_name)
