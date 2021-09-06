import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from pyvalem.formula import Formula
from mendeleev import element

file_dir = Path(__file__).parent.resolve().absolute()


def load_data(cache: bool = True) -> dict[str, dict[int, dict[str, Any]]]:
    file_name = 'data_final.yaml'
    file_name_root = ''.join(file_name.split('.')[:-1])
    cache_path = file_dir.joinpath('_cache', f'{file_name_root}.pkl')
    yaml_path = file_dir.joinpath(file_name)

    if cache and cache_path.is_file():
        with open(cache_path, 'rb') as stream:
            data = pickle.load(stream)
    else:
        with open(yaml_path, 'r') as stream:
            data = yaml.load(stream, yaml.FullLoader)
        if cache:
            cache_path.parent.mkdir(exist_ok=True, parents=True)
            with open(cache_path, 'wb') as stream:
                pickle.dump(data, stream)
    return data


def get_db_url(r_dict):
    if r_dict['database'] == 'kida':
        return r_dict['db_id']
    elif r_dict['database'] == 'nfri':
        return f"https://dcpp.kfe.re.kr/search/popupViewBasic.do?plBiDataNum={r_dict['db_id']}"
    elif r_dict['database'] == 'qdb':
        return f"https://quantemoldb.com/reactions/dataset/details/{int(r_dict['db_id'])}"
    elif r_dict['database'] == 'umist':
        return f"RATE12 Index: {r_dict['db_id']}"
    else:
        raise ValueError('This should not happen!')


sp_attributes = [
    'name',
    'mass',
    'charge',
    'hform',
    'hform_neutral',
    'polarizability',
    'dipole_moment',
    'db',
    'db_id'
]
formulas_memo = {}


def build_species_attributes(sp_dict: dict[str, Any]) -> pd.Series:
    el_map = {'2H': 'D', '3H': 'T'}
    formula_str = ''.join([f'{el_map.get(el, el)}{num}' for el, num in sp_dict['elements'].items()])
    formula = formulas_memo.get(formula_str, Formula(formula_str))
    if formula_str not in formulas_memo:
        formulas_memo[formula_str] = formula

    return pd.Series([
        sp_dict['name'],
        formula.mass,
        sp_dict.get('charge', 0),
        sp_dict.get('hform', np.nan),
        sp_dict.get('hform_n0', np.nan),
        sp_dict.get('polarizability', np.nan),
        sp_dict.get('dipole_moment', np.nan),
        sp_dict['database'],
        str(sp_dict.get('db_id', ''))
    ], index=sp_attributes)


blocks = 's p d f'.split()
groups = [f'{gr1}{gr2}' for gr1 in 'I II III IV V VI VII VIII'.split() for gr2 in ['A', 'B']]
gr_bl_index = [f'block_{bl}' for bl in blocks] + [f'group_{gr}' for gr in groups]
element_memo = {}


def build_species_elemental(elements_dict: dict[str, int]) -> pd.Series:
    el_map = {'2H': 'H', '3H': 'H', 'D': 'H', 'T': 'H'}
    blocks_counts = {}
    groups_counts = {}
    for atom, count in elements_dict.items():
        if atom in element_memo:
            el = element_memo[atom]
        else:
            el = element(el_map.get(atom, atom))
            element_memo[atom] = el
        if el.block in blocks_counts:
            blocks_counts[el.block] += count
        else:
            blocks_counts[el.block] = count
        if el.group.symbol in groups_counts:
            groups_counts[el.group.symbol] += count
        else:
            groups_counts[el.group.symbol] = count
    return pd.Series(
        [blocks_counts.get(bl, 0) for bl in blocks] + [groups_counts.get(gr, 0) for gr in groups],
        index=gr_bl_index
    )


def get_elements_interchanged(r_dict: dict[str, Any], species: dict[int: dict]) -> dict[str, int]:
    """Uff, I'm way too tired to write this in a nice way... fuck it!"""
    el_map = {'2H': 'D', '3H': 'T'}
    r1, _, p1, p2 = [species[sp_id]['elements'] for sp_id in r_dict['reactants'] + r_dict['products']]
    r1 = {el_map.get(el, el): num for el, num in r1.items()}
    p1 = {el_map.get(el, el): num for el, num in p1.items()}
    p2 = {el_map.get(el, el): num for el, num in p2.items()}

    o1, o2 = r1.copy(), r1.copy()

    for el, num in p1.items():
        if el in o1:
            o1[el] -= num
        else:
            o1[el] = -num

    for el, num in p2.items():
        if el in o2:
            o2[el] -= num
        else:
            o2[el] = -num

    options = [o1, o2]
    total_masses = []
    num_fragments = []
    for o in options:
        nf = set()
        for val in o.values():
            if val > 0:
                nf.add(1)
            elif val < 0:
                nf.add(-1)
        num_fragments.append(len(nf))

        formula_str = ''.join([f'{el}{num}' for el, num in o.items()])
        formula = formulas_memo.get(formula_str, Formula(formula_str))
        if formula_str not in formulas_memo:
            formulas_memo[formula_str] = formula
        total_masses.append(formula.mass)

    if len(set(num_fragments)) != 1:
        res = options[num_fragments.index(min(num_fragments))]
    else:
        res = options[total_masses.index(min(total_masses))]
    return {el: int(abs(val)) for el, val in res.items() if val}


def build_exchanged(r_dict: dict[str, Any], species: dict[int: dict]) -> pd.Series:
    exchanged_elements = get_elements_interchanged(r_dict, species)
    formula_str = ''.join([f'{el}{num}' for el, num in exchanged_elements.items()])
    if formula_str:
        formula = formulas_memo.get(formula_str, Formula(formula_str))
        if formula_str not in formulas_memo:
            formulas_memo[formula_str] = formula
        mass = formula.mass
    else:
        mass = 0
    ser1 = pd.Series([mass, sum(exchanged_elements.values())], index=['exchanged_mass', 'exchanged_atoms'])
    ser2 = build_species_elemental(exchanged_elements)
    ser2.index = [f'exchanged_{attr}' for attr in ser2.index]
    return pd.concat([ser1, ser2])


r_attributes = [
    'reaction_string',
    'database',
    'database_url',
    'source',
    'arrhenius_alpha',
    'arrhenius_beta',
    'arrhenius_gamma',
    'log_k(300K)'
]


def build_reaction_attributes(r_dict: dict[str, Any], species: dict[int: dict]) -> pd.Series:
    lhs_str = ' + '.join(species[sp_id]['name'] for sp_id in r_dict['reactants'])
    rhs_str = ' + '.join(species[sp_id]['name'] for sp_id in r_dict['products'])
    r_str = f'{lhs_str} -> {rhs_str}'
    a, b, c = r_dict['arrh_a'], r_dict.get('arrh_b', np.nan), r_dict.get('arrh_c', np.nan)
    return pd.Series([
        r_str,
        r_dict['database'],
        get_db_url(r_dict),
        r_dict.get('source', np.nan),
        a,
        b,
        c,
        np.log10(a * np.exp(-(c if not np.isnan(c) else 0) / 300))
    ], index=r_attributes)


def build_reaction_entry(r_id: int, r_dict: dict[str, Any], species: dict[int: dict]) -> pd.Series:
    print(f'building entry for reaction R{r_id}.')
    subseries = [build_reaction_attributes(r_dict, species)]
    for side, sp_label in zip([r_dict['reactants'], r_dict['products']], ['reactant', 'product']):
        for i, sp_id in enumerate(side, start=1):
            prefix = f'{sp_label}_{i}'
            sp_ser_1 = build_species_attributes(species[sp_id])
            sp_ser_1.index = [f'{prefix}_{attr}' for attr in sp_ser_1.index]
            subseries.append(sp_ser_1)
            sp_ser_2 = build_species_elemental(species[sp_id]['elements'])
            sp_ser_2.index = [f'{prefix}_{attr}' for attr in sp_ser_2.index]
            subseries.append(sp_ser_2)
    subseries.append(build_exchanged(r_dict, species))
    return pd.concat(subseries)


def build_raw_dataset() -> pd.DataFrame:
    data = load_data()
    r_pks = list(sorted(data['reactions'].keys()))
    dataset = pd.DataFrame(
        [build_reaction_entry(i, data['reactions'][i], data['species']) for i in r_pks],
        index=r_pks,
    )

    return dataset


if __name__ == '__main__':

    def main():
        ds = build_raw_dataset()
        ds.to_csv(file_dir / 'dataset_raw.csv')

    main()
