"""Utilities for the "Baromètre de la Science Ouverte" dataset.

BSO :

https://ministeresuprecherche.github.io/bso/

https://data.enseignementsup-recherche.gouv.fr/explore/dataset/open-access-monitor-france/

https://hal.archives-ouvertes.fr/hal-02141819v1

https://visiarchives.sciencesconf.org/data/pages/2018_11_19_CouperinScanR.pdf

https://jnso2018.sciencesconf.org/resource/page/id/9
https://jnso2018.sciencesconf.org/data/pages/poster_BSO_FRANCE_20181119.pdf

Unpaywall:

https://unpaywall.org/data-format
"""

from os.path import splitext

import numpy as np
import pandas as pd


JSON_SCHEMA = 'data/open-access-monitor-france_schema.json'
JSON_FILE = 'data/open-access-monitor-france.json'

CSV_FILE = 'data/open-access-monitor-france.csv'
# fixed CSV file
bname, ext = splitext(CSV_FILE)
FIX_CSV_FILE = bname + '_fix' + ext

DTYPE = {
    'oa_host_type': 'category',
    'scientific_field': 'category',
    'oa_version': 'category',
    'genre': 'category',
    'oa_host_type_genre': 'category',
    'oa_host_type_scientific_field': 'category',
}


def drop_duplicate_titles(df_bso):
    """Drop lines whose titles are not unique.

    The most common 'title' is "Introduction", appearing eg. in books and
    journal issues.
    The 'source_title' is the name of the journal or book series, which is
    often not very helpful for very general collections that cover more than
    one scientific fields (ex: "SpringerBriefs in Applied Sciences and
    Technology").
    A solution would be to use the 'booktitle' but Unpaywall does not provide
    it.

    Parameters
    ----------
    df_bso

    Returns
    -------
    df_bso
    """
    df_title_cnt = df_bso['title'].value_counts().to_frame()
    df_title_cnt = df_title_cnt.reset_index()
    df_title_cnt.columns = ['title', 'count']
    print(df_title_cnt[df_title_cnt['count'] > 1]['title'].to_string())
    # return df_bso


def fix_bso_csv():
    """Fix the BSO CSV file"""
    df_bso = pd.read_csv(CSV_FILE, sep=';', dtype=str)
    # values in 'scientific_field' (and derived columns) contain '\n'
    cols_nl = [
        'scientific_field',
        'oa_host_type_year_scientific_field',
        'oa_host_type_scientific_field'
    ]
    for col_name in cols_nl:
        df_bso[col_name] = df_bso[col_name].str.replace('\n', '', regex=False)
    # dump
    df_bso.to_csv(FIX_CSV_FILE, sep=';', index=False)


def load_bso_csv(fixed=True):
    """Load the CSV version of the BSO

    Parameters
    ----------
    fixed : boolean, defaults to True
        If True, use the fixed version of the CSV file.

    Returns
    -------
    df_bso : DataFrame
        Data from the BSO dataset.
    """
    fname = FIX_CSV_FILE if fixed else CSV_FILE
    df_bso = pd.read_csv(fname, sep=';', dtype=DTYPE)
    return df_bso


def load_bso_json():
    """Load the JSON version of the BSO

    Does not work great at the moment as the content is nested under "fields".
    """
    df_bso = pd.read_json(JSON_FILE, orient='records')
    return df_bso


def get_abstracts_crossref():
    """Get abstracts from CrossRef with their DOI.

    https://github.com/CrossRef/rest-api-doc/issues/98
    """
    return


def examine_bso(df_bso):
    """Examine BSO dataset"""
    # Check assumptions:
    # * every work should have a title
    print("Missing 'title': {}".format(
        sum(df_bso['title'].isna())))
    print(np.where(df_bso['title'].isna()))
    # * every work has been given a scientific field
    print("Missing 'scientific_field': {}".format(
        sum(df_bso['scientific_field'].isna())))
    print(np.where(df_bso['scientific_field'].isna()))


if __name__ == '__main__':
    if True:
        fix_bso_csv()
    if True:
        df_bso = load_bso_csv(fixed=True)
        print("Number of entries: {}".format(df_bso.shape[0]))
        examine_bso(df_bso)
        # print(df_bso.columns)
        # print(df_bso.info())
        # print(df_bso.memory_usage(deep=True))
        # print(df_bso.describe(include='all').transpose())
        # print(df_bso['title'].describe())
        drop_duplicate_titles(df_bso)
    # print(df_bso['title'].value_counts().to_frame())
