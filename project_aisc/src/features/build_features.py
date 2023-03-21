"""
Module used to process data. AtomData process chemical species data while  SuperConData process superconductivity data.
train_test_split is a custom function that split data for test/train, it works on list representing superconductivity
materials containing element's data (a material is thought as an ensemble of elements; in our implementation we use list
to manage multi-input model)
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split as sk_train_test_split
from utils import utils


class AtomData:
    """Class that holds the data of the periodic table and processes them.

        Categorical data is encoded with both natural number mapping and one hot encoding.
        Numerical data is imported with the mean value and standardized.
        Lanthanides and antanides have group set equal to 0 before build_features

        Attributes:
            periodic_table: pandas DataFrame that holds non-processed data
            atom_data: pandas DataFrame that holds processed data

        Methods:
            get_atom_data: return processed periodic table data
            build_atom_data: process and store periodic table data in atom_data
            get_numerical_data_processed: return a pd.DataFrame with numerical data processed
            get_categorical_data_processed: return a pd.DataFrame with categorical data processed
            one_hot_encode: one hot encode a feature available in self.periodic_table
            natural_encode: map feature's value into natural number

    """

    def __init__(self, periodic_table):
        """Units AtomData with periodic table data"""

        self.periodic_table = periodic_table
        self.atom_data = None

    def one_hot_encode(self, atomic_feature):
        """Select atomic_feature in self.periodic_table and one hot encode it"""

        one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        # Nan value are set equals to Void (imputation) and encoded as a separate class
        uncoded_atomic_feature = np.array(self.periodic_table[atomic_feature].fillna(value='Void')).reshape(-1, 1)
        one_hot_feature_encoded = pd.DataFrame(one_hot_encoder.fit_transform(uncoded_atomic_feature))

        # Put index back after one hot encode action (OneHotEncoder remove index)
        one_hot_feature_encoded.index = self.periodic_table.index

        return one_hot_feature_encoded

    def natural_encode(self, atomic_feature, atomic_map):
        """Map feature's value into natural number in according to atomic_map (dict)"""

        natural_feature_encoded = self.periodic_table[atomic_feature].map(atomic_map)

        return natural_feature_encoded

    def get_categorical_data_processed(self, categorical_encode_plan=dict(lattice_structure='one-hot',
                                                                          block={'s': 0, 'p': 1, 'd': 2, 'f': 3})):
        """Return categorical data encoded"""

        categorical_dataset_list = []

        for categorical_feature in categorical_encode_plan.keys():
            if categorical_encode_plan[categorical_feature] == 'one-hot':
                categorical_dataset_list.append(self.one_hot_encode(categorical_feature))
            else:
                categorical_dataset_list.append(self.natural_encode(atomic_feature=categorical_feature,
                                                                    atomic_map=categorical_encode_plan[
                                                                        categorical_feature]))

        categorical_dataset = pd.concat(categorical_dataset_list, axis=1)

        return categorical_dataset

    def get_numerical_data_processed(self, numerical_columns):
        """Return a DataFrame with numerical data processed: Imputed by mean and standardized"""

        numeric_periodic_table = self.periodic_table[numerical_columns].copy()
        # Impute the missing value with the mean of the available data and standardize them
        my_imputer = SimpleImputer(strategy='mean')

        transformer = preprocessing.StandardScaler()
        imputed_periodic_table = pd.DataFrame(my_imputer.fit_transform(numeric_periodic_table))
        imputed_standardized_periodic_table = pd.DataFrame(transformer.fit_transform(imputed_periodic_table))

        # Imputation removed column names so we put them back
        imputed_standardized_periodic_table.columns = numeric_periodic_table.columns

        return imputed_standardized_periodic_table

    def build_atom_data(self):
        """
        Fill self.atom_data with periodic table data imputed, encoded and processed.
        """

        # Select numerical columns for imputations and procession
        numerical_columns = [feature for feature in self.periodic_table.columns if
                             self.periodic_table[feature].dtype in ['int64', 'float64', 'int32', 'float32']]

        # Lanthanides and antanides don't have group.
        # We choose to set them to 0 before build_features (it's an unique value for them)

        if 'group_id' in numerical_columns:
            self.periodic_table.loc[:, 'group_id'] = self.periodic_table['group_id'].fillna(0)

        categorical_data = self.get_categorical_data_processed(
            categorical_encode_plan={'lattice_structure': 'one-hot', 'block': {'s': 0, 'p': 1, 'd': 2, 'f': 3}})
        numerical_data = self.get_numerical_data_processed(numerical_columns)

        self.atom_data = pd.concat([numerical_data, categorical_data], axis=1)

    def get_atom_data(self):
        """Return periodic table data processed"""

        # if periodic table data is not processed, we process it and then return it
        if self.atom_data is None:
            self.build_atom_data()

        return self.atom_data


class SuperConData:
    """Class processes SuperCon dataset to make it ready as model input.

        The class holds the relevant data to build the input for the machine learning
        model. It can build vector representation of the material, i.e. assemble
        features computed by function from atomic features (analytic dataset).
        In this case the input is an array.
        Or it can build list of atomic representation: in this case the input is a list
        and each element in the list is an array representing the chemical element.

        Attributes:
            atom_data: pandas DataFrame that holds processed data of atoms
            supercon_dataset: pandas DataFrame that holds SuperCon data
            dataset: dataset ready to be processed by the model
            padding: (default auto) pads input if needed. Set it manual or let it infer the value

        Methods:
            get_dataset: return processed periodic table data
            build_dataset: assemble and store dataset ready for model input
            select_atom_in_formula: select the atoms and relative quantity written in the chemical formula
            get_atom_arrays: return arrays filled with atom data and padding
            expand_row_into_model_input: turn a row into a model input


    """

    def __init__(self, atom_data, supercon_dataset, padding='auto'):
        """Init SuperConData. Processed atoms data and SuperCon are mandatory"""
        self.atom_data = atom_data
        self.supercon_dataset = supercon_dataset
        self.supercon_dataset_processed = None
        self.analytical_supercon_dataset_processed = None
        self._padding = padding

    def get_dataset(self):
        """Return dataset ready for model"""

        if self.supercon_dataset_processed is None:
            self.build_dataset()

        return self.supercon_dataset_processed

    def build_dataset(self):
        """Iterate over the rows, build model input and store it into dataset attributes"""

        supercon_dataset = self.supercon_dataset.copy()
        # We keep only chemical symbol columns(are 96)
        supercon_dataset = supercon_dataset.iloc[:, :96]
        # Map atom symbol into natural number:  symbol-> Z-1 where Z: atomic number
        supercon_dataset.columns = range(96)

        max_elements_in_formula = (supercon_dataset > 0).sum(axis=1).max()
        # Set padding if needed for input generation
        if self._padding == 'auto':
            self._padding = max_elements_in_formula

        dataset_ready_for_model = supercon_dataset.apply(self.expand_row_into_model_input, axis=1)
        # Rearrange type and dimension order to fit model input
        # It's a list of #samples and each samples has associated an array
        # list (np.array.shape = (#samples,atom_representation) )

        dataset_ready_for_model = np.moveaxis(np.array(list(dataset_ready_for_model)), 0, 1)
        self.supercon_dataset_processed = list(dataset_ready_for_model)

    def select_atom_in_formula(self, formula_array):
        """Return a list of tuples with atomic number shifted by 1(Z-1) and atomic quantity"""

        # remove chemical elements not present in the chemical formula
        formula_array_other_than_0 = formula_array[formula_array > 0]
        # atoms_index is the atomic number (Z) -1
        atoms_index = formula_array_other_than_0.index
        # quantity of the relative element in the formula
        atoms_value = formula_array_other_than_0.values

        atom_symbol_quantity = [(i, j) for i, j in zip(atoms_index, atoms_value)]

        return atom_symbol_quantity

    def get_atom_arrays(self, atom_symbol_quantity):
        """Return an array(numpy) of length max_length filled by atomic arrays and 0's arrays (padding)"""

        list_atom_features = []
        # The symbol is not a string (like 'H') but an index (like 0 for 'H')
        # symbol = Z-1 where Z = atomic number
        for symbol, quantity in atom_symbol_quantity:
            atom_features = self.atom_data.iloc[symbol, :].to_numpy()
            complete_atom_features = np.append(atom_features, quantity)
            list_atom_features.append(complete_atom_features)

        padding_value = self._padding - len(atom_symbol_quantity)
        # Padding need to be > 0. It is ensured by construction normally
        assert padding_value >= 0, f'padding_value: {padding_value} and atoms in formula: {atom_symbol_quantity}'

        array_atom_features_padded = np.pad(list_atom_features, [(0, padding_value), (0, 0)], )

        return array_atom_features_padded

    def expand_row_into_model_input(self, row):
        """Expand a row (pandas Series) into a model ready input"""
        # Select atom with quantity different from 0
        # And put them into a list of tuples (index_symbol,quantity)
        atom_symbol_quantity = self.select_atom_in_formula(row)

        expanded_row = self.get_atom_arrays(atom_symbol_quantity)

        return expanded_row

    def get_analytical_dataset(self):
        """Return dataset ready for model"""

        if self.analytical_supercon_dataset_processed is None:
            self.build_analytical_dataset()

        return self.analytical_supercon_dataset_processed

    def build_analytical_dataset(self):

        supercon_dataset = self.supercon_dataset.copy()
        # We keep only chemical symbol columns(are 96)
        supercon_dataset = supercon_dataset.iloc[:, :96]
        # Map atom symbol into natural number:  symbol-> Z-1 where Z: atomic number
        supercon_dataset.columns = range(96)
        string_mono_functions = ['mean', 'geometric_average', 'entropy', 'range_feature', 'std']
        string_bi_functions = ['weighted_mean', 'weighted_geo_mean', 'weighted_entropy', 'weighted_range_feature',
                               'weighted_std']
        selected_features = ['atomic_weight', 'ionenergies', 'valence', 'thermal_conductivity', 'atomic_radius_rahm',
                             'density', 'electron_affinity', 'fusion_heat']

        dataset_columns = [
            *[[func + '_' + feature for func in string_mono_functions] for feature in selected_features],
            *[[func + '_' + feature for func in string_bi_functions] for feature in selected_features]
        ]
        dataset_columns = [feature for sublist in dataset_columns for feature in sublist]

        dataset_ready_for_model = supercon_dataset.apply(self.expand_row_into_analytical_model_input, axis=1)
        self.analytical_supercon_dataset_processed = pd.DataFrame(list(dataset_ready_for_model))
        self.analytical_supercon_dataset_processed.columns = dataset_columns

    def expand_row_into_analytical_model_input(self, row):
        """Expand a row (pandas Series) into an analytical model ready input with length 80"""
        # Select atom with quantity different from 0
        # And put them into a list of tuples (index_symbol,quantity)
        atom_symbol_quantity = self.select_atom_in_formula(row)
        expanded_row = []
        selected_features = ['atomic_weight', 'ionenergies', 'valence', 'thermal_conductivity', 'atomic_radius_rahm',
                             'density', 'electron_affinity', 'fusion_heat']
        atoms_quantity = [quantity for _, quantity in atom_symbol_quantity]
        atoms_symbol = [symbol for symbol, _ in atom_symbol_quantity]
        selected_atom = self.atom_data.loc[atoms_symbol, selected_features]

        analytic_mono_functions = [np.mean, utils.geo_mean, utils.entropy, utils.range_feature, np.std, ]
        for foo in analytic_mono_functions:
            expanded_row.append(selected_atom.apply(foo).values)

        analytic_bi_functions = [utils.weighted_average, utils.weighted_geo_mean, utils.weighted_entropy,
                                 utils.weighted_range_feature, utils.weighted_std]
        for foo in analytic_bi_functions:
            expanded_row.append(selected_atom.apply(lambda x: foo(x, atoms_quantity), axis=0).values)

        expanded_row = np.reshape(np.array(expanded_row), (-1,))
        return expanded_row


def train_test_split(data, label, test_size=0.2, seed=42):
    """Custom train-test split.

       Args:
           data: A list of numpy array containing atom's representation
           label: List or numpy array or pandas Series containing labels
           test_size: fraction of data to use as test set (default 0.2)
           seed: Int controlling the data's shuffle. Useful to make experiments reproducible.
       Returns:
           X,X_test,y,y_test: A tuple containing in order data, test's data, label, test's label
       """

    X, X_test, Y, Y_test = sk_train_test_split(np.moveaxis(np.array(data), 0, 1), label, test_size=test_size,
                                               random_state=seed)
    X = list(np.moveaxis(X, 0, 1))
    X_test = list(np.moveaxis(X_test, 0, 1))

    return X, X_test, Y, Y_test


def remove_columns_with_missing_elements(dataset, max_missing_value=30, exceptions=None):
    """remove columns that has more than max_missing_value with exception for except columns"""

    empty_columns = [column for column in dataset.columns if dataset[column].isna().sum() > max_missing_value]

    # remove from the list exceptional columns even if they have too many missing values
    if exceptions:
        for column in exceptions:
            if column in empty_columns:
                empty_columns.remove(column)

    return dataset.drop(columns=empty_columns)
