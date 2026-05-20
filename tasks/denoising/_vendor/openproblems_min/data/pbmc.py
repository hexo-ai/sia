from . import utils

import os
import scprep
import tempfile

# 1k PBMCs sparsified from figshare (OpenProblems v1)
URL = "https://ndownloader.figshare.com/files/36088667"


@utils.loader(data_url=URL, data_reference="10x2018pbmc")
def load_tenx_1k_pbmc(test=False):
    """Download 1k PBMC data from Figshare."""
    import scanpy as sc

    if test:
        adata = load_tenx_1k_pbmc(test=False)
        sc.pp.subsample(adata, n_obs=100)
        adata = adata[:, :1000]
        utils.filter_genes_cells(adata)
        return adata

    with tempfile.TemporaryDirectory() as tempdir:
        filepath = os.path.join(tempdir, "pbmc_1k.h5ad")
        scprep.io.download.download_url(URL, filepath)
        adata = sc.read_h5ad(filepath)

    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
    adata.X = adata.layers["counts"]

    utils.filter_genes_cells(adata)
    return adata
