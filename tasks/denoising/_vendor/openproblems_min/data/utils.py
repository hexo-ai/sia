from . import TEMPDIR

import anndata
import functools
import hashlib
import logging
import numpy as np
import os
import scipy.sparse

log = logging.getLogger("openproblems")


def _func_to_bytes(func):
    return bytes(".".join([func.__module__, func.__name__]), encoding="utf-8")


def _obj_to_bytes(obj):
    return bytes(str(obj), encoding="utf-8")


def _hash_function(func, *args, **kwargs):
    hash = hashlib.sha256()
    hash.update(_func_to_bytes(func))
    hash.update(_obj_to_bytes(args))
    hash.update(_obj_to_bytes(kwargs))
    return hash.hexdigest()


def _cache_path(func, *args, **kwargs):
    try:
        os.mkdir(TEMPDIR)
    except OSError:
        pass
    if hasattr(func, "__wrapped__"):
        func = func.__wrapped__
    filename = "openproblems_{}.h5ad".format(_hash_function(func, *args, **kwargs))
    return os.path.join(TEMPDIR, filename)


def _fix_matrix_format(X):
    if scipy.sparse.issparse(X) and not isinstance(X, scipy.sparse.csr_matrix):
        X = X.tocsr()
    if isinstance(X, np.matrix):
        X = X.A
    return X


def _fix_adata(adata):
    adata.strings_to_categoricals()
    if "var_names_all" not in adata.uns:
        adata.uns["var_names_all"] = adata.var.index.to_numpy()
    adata.X = _fix_matrix_format(adata.X)
    for layer in adata.layers:
        adata.layers[layer] = _fix_matrix_format(adata.layers[layer])
    for obsm in adata.obsm:
        adata.obsm[obsm] = _fix_matrix_format(adata.obsm[obsm])
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X


def loader(data_url, data_reference):
    """Decorate a data loader function.

    Parameters
    ----------
    data_url : str
        Link to the original source of the dataset
    data_reference : str
        Link to the paper describing how the dataset was generated
    """

    def decorator(func):
        @functools.wraps(func)
        def apply_func(*args, **kwargs):
            filepath = _cache_path(func, *args, **kwargs)
            dataset_name = f"{func.__name__}({args}, {kwargs})"
            if os.path.isfile(filepath):
                log.debug(f"Loading cached {dataset_name} dataset")
                adata = anndata.read_h5ad(filepath)
                adata.uns["_from_cache"] = True
                return adata
            else:
                log.debug(f"Downloading {dataset_name} dataset")
                adata = func(*args, **kwargs)
                adata.uns["_from_cache"] = False
                write_h5ad(adata, filepath)
                return adata

        apply_func.metadata = dict(data_url=data_url, data_reference=data_reference)
        return apply_func

    return decorator


def filter_genes_cells(adata):
    """Remove empty cells and genes."""
    import scanpy as sc

    if "var_names_all" not in adata.uns:
        # fill in original var names before filtering
        adata.uns["var_names_all"] = adata.var.index.to_numpy()
    sc.pp.filter_genes(adata, min_cells=1)
    sc.pp.filter_cells(adata, min_counts=2)


def subsample_even(adata, n_obs, even_obs):
    """Subsample a dataset evenly across an obs.

    Parameters
    ----------
    adata : AnnData
    n_obs : int
        Total number of cells to retain
    even_obs : str
        `adata.obs[even_obs]` to be subsampled evenly across partitions.

    Returns
    -------
    adata : AnnData
        Subsampled AnnData object
    """
    import scanpy as sc

    values = adata.obs[even_obs].unique()
    adatas = []
    n_obs_per_value = n_obs // len(values)
    for v in values:
        adata_subset = adata[adata.obs[even_obs] == v].copy()
        sc.pp.subsample(adata_subset, n_obs=min(n_obs_per_value, adata_subset.shape[0]))
        adatas.append(adata_subset)

    adata_out = anndata.concat(adatas, label="_obs_batch")

    adata_out.uns = adata.uns
    adata_out.varm = adata.varm
    adata_out.varp = adata.varp
    return adata_out


def write_h5ad(adata, filepath):
    if os.path.isfile(filepath):
        os.unlink(filepath)
    _fix_adata(adata)
    adata.write_h5ad(filepath)
