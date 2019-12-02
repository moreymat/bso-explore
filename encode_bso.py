"""Encode BSO dataset: titles.

Maybe journal, possibly abstract when present?

Google's Universal Sentence Encoder Multilingual:
https://tfhub.dev/google/universal-sentence-encoder-multilingual/2

t-SNE + viz:
https://towardsdatascience.com/plotting-text-and-image-vectors-using-t-sne-d0e43e55d89

See also:
https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/semantic_similarity_with_tf_hub_universal_encoder.ipynb#scrollTo=MkqPOxH3EL1j
https://cloud.google.com/solutions/machine-learning/building-real-time-embeddings-similarity-matching-system?hl=fr
https://blog.floydhub.com/automate-customer-support-part-two/
"""

import numba
import numpy as np
import pandas as pd
import tensorflow_hub as hub
import tensorflow_text
# t-sne
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import LabelEncoder
# umap
import umap
# plotting
import datashader as ds
import datashader.utils as ds_utils
import datashader.transfer_functions as ds_tf
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns

from bso_dataset import load_bso_csv


# random state for t-SNE and UMAP
RAND_STATE = 42


def ds_scatter(df, lbl, img_name):
    """Draw a scatter plot of the data with datashader.

    Parameters
    ----------
    df : DataFrame

    lbl : str
        Name of the column that contains labels.
    """
    lbls = df[lbl].cat.categories
    # choose a color palette with seaborn
    palette = np.array(sns.color_palette("hls", len(lbls)).as_hex())
    color_key = {lbl: color for lbl, color in zip(lbls, palette)}
    # draw and export image
    cvs = ds.Canvas(plot_width=400, plot_height=400)
    agg = cvs.points(df, 'x', 'y', ds.count_cat(lbl))
    img = ds_tf.shade(agg, color_key=color_key, how='eq_hist')
    ds_utils.export_image(img, filename=img_name, background='white')
    # display image
    image = plt.imread(img_name + '.png')
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.imshow(image)
    plt.setp(ax, xticks=[], yticks=[])
    plt.title("BSO titles embedded\n"
              "into two dimensions by UMAP\n"
              "visualised with Datashader",
              fontsize=12)
    plt.show()


def scatter(x, y):
    """Draw a scatter plot of the data with matplotlib.

    Parameters
    ----------
    x : 

    y : 

    Returns
    -------
    f

    ax

    sc

    txts
    """
    # encode labels as integers
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    # choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", len(le.classes_)))

    # create a scatter plot
    f = plt.figure(figsize=(32, 32))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=20,
                    c=palette[y_enc])
    #plt.xlim(-25, 25)
    #plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the label for each class
    txts = []
    for lbl in list(le.classes_):
        # Position of each label
        xtext, ytext = np.median(x[y == lbl, :], axis=0)
        txt = ax.text(xtext, ytext, lbl, fontsize=50)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    #
    return f, ax, sc, txts


if __name__ == '__main__':
    # load BSO data
    # we'll encode and project 'title' and use 'scientific_field' as label
    # (even if it's a noisy, predicted, label)
    df_bso = load_bso_csv(fixed=True)
    df_bso['title'] = df_bso['title'].fillna('')
    # take care of missing data in categorical column
    df_bso['scientific_field'] = df_bso['scientific_field'].fillna('unknown')
    # sample entries
    x_lim = 20000
    titles = df_bso['title'][:x_lim]
    sci_genres = df_bso['scientific_field'][:x_lim]

    # load multilingual universal sentence encoder
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/2")
    sci_genres_nb = len(sci_genres.unique())

    # embed titles
    titles_emb = embed(titles)["outputs"]
    # alternative: embed 'title, source_title'
    df_bso['source_title'] = df_bso['source_title'].fillna('')
    df_bso['full_title'] = df_bso['title'] + ', ' + df_bso['source_title']
    full_titles = df_bso['full_title'][:x_lim]
    full_titles_emb = embed(full_titles)["outputs"]

    # choose method
    reduce_dim = 'umap'  # 'tsne' | 'umap'
    # default for t-SNE and UMAP implementations: metric='euclidean'
    metric = 'cosine'
    #
    if reduce_dim == 'tsne':
        # reduce dimension with t-SNE
        # * title
        titles_proj = (TSNE(n_components=2, random_state=RAND_STATE,
                            metric=metric)
                       .fit_transform(titles_emb))
        # * title, source_title
        full_titles_proj = (TSNE(n_components=2, random_state=RAND_STATE,
                                 metric=metric)
                            .fit_transform(full_titles_emb))
    else:
        # 'cosine' is a good default metric for sentence embeddings
        # (the "Universal Sentence Encoder" paper uses angular similarity
        # instead of cosine in order to penalize small differences heavily
        # for the pairwise Semantic Textual Similarity task)
        # https://math.stackexchange.com/questions/2874940/cosine-similarity-vs-angular-distance
        titles_proj = (umap.UMAP(n_neighbors=5,
                                 n_components=2,
                                 metric=metric,
                                 min_dist=0.3,
                                 random_state=RAND_STATE)
                       .fit_transform(titles_emb))
        full_titles_proj = (umap.UMAP(n_neighbors=5,
                                      n_components=2,
                                      metric=metric,
                                      min_dist=0.3,
                                      random_state=RAND_STATE)
                            .fit_transform(full_titles_emb))

    sns.set(context="paper", style="white")
    # pack 2-D projections and scientific_field in a DataFrame
    df_titles_proj = pd.DataFrame(titles_proj, columns=('x', 'y'))
    df_titles_proj['scientific_field'] = sci_genres
    img_name = 'img/title_guse_{}_{}_{}_ds'.format(reduce_dim, metric, x_lim)
    ds_scatter(df_titles_proj, 'scientific_field', img_name)
    #
    df_full_titles_proj = pd.DataFrame(full_titles_proj, columns=('x', 'y'))
    df_full_titles_proj['scientific_field'] = sci_genres
    img_name = 'img/fulltitle_guse_{}_{}_{}_ds'.format(reduce_dim, metric, x_lim)
    ds_scatter(df_full_titles_proj, 'scientific_field', img_name)

    # plot
    # * title
    # sns.palplot(np.array(sns.color_palette("hls", sci_genres_nb)))
    scatter(titles_proj, sci_genres)
    plt.savefig('img/title_guse_{}_{}_{}.png'.format(
        reduce_dim, metric, x_lim),
                dpi=120)
    # * title, source_title
    # sns.palplot(np.array(sns.color_palette("hls", sci_genres_nb)))
    scatter(full_titles_proj, sci_genres)
    plt.savefig('img/fulltitle_guse_{}_{}_{}.png'.format(
        reduce_dim, metric, x_lim),
                dpi=120)
