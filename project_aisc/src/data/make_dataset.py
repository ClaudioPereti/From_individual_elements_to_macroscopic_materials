"""
Module containing functions to load the necessary data. The data is divided into data on superconductors and data on
chemical species. For superconductivity data there are function to load data in the appropriate format, ie a pandas
DataFrame with columns counting chemical species belonging to the chemical formula, chemical formula and the critical
temperature. For non-superconductors the critical temperature is set to 0. The function to build the dataset of
superconductors and non-superconductors must have two columns; one column containing che chemical formula and the other
the critical temperature. For non-superconductors the critical temperature is set to 0
"""
import pandas as pd
import numpy as np
from mendeleev import element
from mendeleev import get_table
import pathlib
from features.build_features import remove_columns_with_missing_elements
from chela.formula_handler import build_dataframe


def get_supercon(name='supercon.csv'):
    """ Load pandas DataFrame of superconductors and relative critical temperatures"""
    name = str(pathlib.Path(__file__).parent.parent.parent) + '/data/raw/' + name
    sc_dataframe = pd.read_csv(name)
    return sc_dataframe


def get_periodictable(max_atomic_number=96, max_missing_value=30):
    """
    Return merged periodic table with the following features:
       'atomic_number', 'atomic_volume', 'block', 'density',
       'dipole_polarizability', 'electron_affinity', 'evaporation_heat',
       'fusion_heat', 'group_id', 'lattice_constant', 'lattice_structure',
       'melting_point', 'period', 'specific_heat', 'thermal_conductivity',
       'vdw_radius', 'covalent_radius_pyykko', 'en_pauling', 'atomic_weight',
       'atomic_radius_rahm', 'valence', 'ionenergies'

    Args:
          max_atomic_number: (default = 96) Holds atoms with atomic number minor of a fixed value
          max_missing_value: (default = 30) Drop features with more than a fixed value

    Returns:
        periodic_table: pd.DataFrame containing periodic table data
    """

    periodic_table = get_mendeleev_periodictable(max_atomic_number=max_atomic_number)

    exceptions = ['thermal_conductivity', 'fusion_heat', 'electron_affinity', 'specific_heat']
    periodic_table = remove_columns_with_missing_elements(dataset=periodic_table,
                                                          exceptions=exceptions,
                                                          max_missing_value=max_missing_value,
                                                          )

    path = str(pathlib.Path(__file__).parent.parent.parent) + "/data/raw/"
    features = ['thermal_conductivity', 'specific_heat', 'electron_affinity', 'density']
    features_and_scale = {'thermal_conductivity': 1, 'specific_heat': 1 / 1000, 'electron_affinity': 1 / 100,
                          'density': 1 / 1000}

    atomic_dataset_dict = _get_external_periodictable_data(path=path, features=features)

    periodic_table = _merge_periodictable_data(features_and_scale=features_and_scale,
                                               atomic_dataset_dict=atomic_dataset_dict, periodic_table=periodic_table)

    return periodic_table


def get_mendeleev_periodictable(max_atomic_number=96):
    """get periodic table from mendeleev software up to atomic number = max_atomic_number"""

    periodic_table = get_table('elements')
    periodic_table = periodic_table.iloc[:max_atomic_number, :]

    # Atomic features we don't  think are linked to superconductivity
    atomic_features_to_drop = ['annotation', 'description', 'name', 'jmol_color', 'symbol', 'is_radioactive',
                               'vdw_radius_mm3',
                               'cpk_color', 'uses', 'sources', 'name_origin', 'discovery_location',
                               'covalent_radius_cordero',
                               'discoverers', 'cas', 'goldschmidt_class', 'molcas_gv_color', 'discovery_year',
                               'atomic_radius', 'series_id',
                               'electronic_configuration', 'glawe_number', 'en_ghosh', 'heat_of_formation',
                               'covalent_radius_pyykko_double',
                               'vdw_radius_alvarez', 'abundance_crust', 'abundance_sea', 'c6_gb', 'vdw_radius_uff',
                               'dipole_polarizability_unc', 'boiling_point', 'pettifor_number', 'mendeleev_number',
                               'geochemical_class',
                               'covalent_radius_pyykko_triple', 'en_allen', 'atomic_weight_uncertainty']

    # There isn't data for elements heavier than (Z)109 however we restrinc the analisys to 96
    # ionenergies and valence are not presence in get_table so we add them with another function
    ionenergies_column = [element(index).ionenergies[1] for index in range(1, max_atomic_number + 1)]
    valence_column = [element(index).nvalence() for index in range(1, max_atomic_number + 1)]

    periodic_table = periodic_table.drop(atomic_features_to_drop, axis=1)

    periodic_table['valence'] = valence_column
    periodic_table['ionenergies'] = ionenergies_column

    return periodic_table


def _get_external_periodictable_data(path, features):
    """Load a list of pandas DataFrame containing single atomic features"""

    # These Dataset are meant to be merged with periodic table data from mendeleev software
    # They have no header and contain only a single atomic feature
    # There is a specific clean process where we replace some strings (they indicate no avaible data) with Nan value
    atomic_dataset_list = [pd.read_csv(path + feature + '.csv', header=0) for feature in features]

    atomic_dataset_list = list(
        map(lambda x: x.replace('QuantityMagnitude[Missing["NotAvailable"]]', np.nan), atomic_dataset_list))
    atomic_dataset_list = list(
        map(lambda x: x.replace('QuantityMagnitude[Missing["Unknown"]]', np.nan), atomic_dataset_list))

    # turn the list into a dictionary to be more manageable
    atomic_dataset_dict = {feature: atomic_dataset_list[index] for index, feature in enumerate(features)}

    return atomic_dataset_dict


def _merge_periodictable_data(features_and_scale, atomic_dataset_dict, periodic_table):
    """Merge periodic table from get_external_periodic_table_data (and scale them) and mendeleev data"""

    for feature in features_and_scale.keys():
        scaled_feature = atomic_dataset_dict[feature][feature].astype('float32') * features_and_scale[feature]
        periodic_table[feature] = periodic_table[feature].fillna(scaled_feature)

    return periodic_table


def build_supercon(path, index_col=0):
    """Create a dataset of superconductors and non-superconductors from a csv file with chemical formulas,
    under column name 'formula', and critical temperature (as a float)

    Args:
        path (str): path to read raw material formulas
        index_col: if index is present put 0, otherwise None
    """
    data = pd.read_csv(path, index_col=index_col)
    data = data.dropna()
    supercon = build_dataframe(data)
    # We use material as label for chemical formulas
    supercon = supercon.rename(columns={'formula': 'material'})
    # build_dataframe create a dataframe with 1-118 chemical elements,
    # but we want only the first 96
    chemical_elements_to_drop = supercon.columns[96:118]
    supercon = supercon.drop(columns=chemical_elements_to_drop)
    # We swap the last 2 columns because we want 'material' as last one
    chemical_symbols = supercon.columns[:96]
    chemical_symbols = list(chemical_symbols)
    chemical_symbols.append('material')
    if "critical_temp" in data.columns:
        chemical_symbols.append('critical_temp')
    supercon = supercon[chemical_symbols]

    return supercon
