from . import utils

import numpy as np
import os
import scprep
import tempfile

# sparsified from https://figshare.com/articles/dataset/Benchmarking_atlas-level_data_integration_in_single-cell_genomics_-_integration_task_datasets_Immune_and_pancreas_/12420968/2 # noqa: E501
URL = "https://ndownloader.figshare.com/files/36086813"


@utils.loader(data_url=URL, data_reference="luecken2022benchmarking")
def load_pancreas(test=False, keep_techs=None):
    """Download pancreas data from figshare."""
    import scanpy as sc

    if test:
        # load full data first, cached if available
        adata = load_pancreas(
            test=False,
            keep_techs=keep_techs or ["celseq", "inDrop4", "smarter"],
        )

        keep_celltypes = adata.obs["celltype"].dtype.categories[[0, 3]]
        adata = adata[adata.obs["celltype"].isin(keep_celltypes)].copy()

        # Subsample pancreas data
        adata = adata[:, :500].copy()
        # Note: could also use 200-500 HVGs rather than 200 random genes
        utils.filter_genes_cells(adata)

        # select 250 cells from each celltype
        keep_cell_idx = np.concatenate(
            [
                np.random.choice(
                    np.argwhere(adata.obs["celltype"].to_numpy() == ct).flatten(),
                    min(250, np.sum(adata.obs["celltype"].to_numpy() == ct)),
                    replace=False,
                )
                for ct in keep_celltypes
            ]
        )
        adata = adata[adata.obs.index[keep_cell_idx]]

        # Ensure there are no cells or genes with 0 counts
        utils.filter_genes_cells(adata)

        return adata

    with tempfile.TemporaryDirectory() as tempdir:
        filepath = os.path.join(tempdir, "pancreas.h5ad")
        scprep.io.download.download_url(URL, filepath)
        adata = sc.read(filepath)

    if keep_techs is not None:
        adata = adata[adata.obs["tech"].isin(keep_techs)].copy()

    # NOTE: adata.X contains log-normalized data, so we're moving it
    adata.layers["log_normalized"] = adata.X
    adata.X = adata.layers["counts"]

    # Ensure there are no cells or genes with 0 counts
    utils.filter_genes_cells(adata)

    return adata
