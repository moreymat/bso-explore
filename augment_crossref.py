"""Augment BSO with CrossRef

CrossRef's REST API doc:
https://github.com/CrossRef/rest-api-doc

https://habanero.readthedocs.io/en/latest/modules/crossref.html
"""

from habanero import Crossref
import pandas as pd
import requests.exceptions

from bso_dataset import load_bso_csv


def query_crossref(cr, dois, fn):
    """Query CrossRef

    Parameters
    ----------
    cr : Crossref

    dois : List[str]
        DOIs

    Returns
    -------
    cr_works : List[Dict]
        List of works returned by CrossRef
    """
    cr_works = []
    for i, doi in enumerate(dois):
        try:
            cr_work = cr.works(ids=doi)
        except requests.exceptions.HTTPError as e:
            cr_work = None
        cr_works.append(fn(cr_work))
    return cr_works


def query_crossref_batch(cr, dois, fn, batch_size=10):
    """Query CrossRef by batch"""
    cr_works = []
    batch_begs = list(range(0, len(dois), batch_size)) + [len(dois)]
    beg_ends = zip(batch_begs[:-1], batch_begs[1:])
    for b_e in beg_ends:
        dois_sel = dois[b_e[0]:b_e[1]].tolist()
        try:
            cr_works_batch = cr.works(ids=dois_sel)
        except requests.exceptions.HTTPError as e:
            # delegate to the variant doing individual queries
            # to handle missing DOIs
            cr_works_batch = query_crossref(cr, dois_sel)
        cr_works.extend(fn(x) for x in cr_works_batch)
    return cr_works


if __name__ == '__main__':
    # setup
    cr = Crossref(mailto="mathieu@datactivist.coop")
    # extract list of DOIs and source_title
    df_bso = load_bso_csv()
    # query CrossRef for DOIs
    # limit how much we take
    df_bso_sel = df_bso.loc[:200,:]
    sel_abs = lambda x: (x['message'].get('abstract', None) if x else None)
    df_bso_sel['abstract'] = query_crossref(cr, df_bso_sel['doi'], sel_abs)
    print(df_bso_sel['abstract'].describe())
    # RESUME HERE dump CSV with new col
    
