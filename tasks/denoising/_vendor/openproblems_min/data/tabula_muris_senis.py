from . import utils

import anndata as ad
import os
import scprep
import tempfile
import time

# CellXGene collection hosting Tabula Muris Senis
COLLECTION_ID = "0b9d8a04-bb9d-44da-aa27-705bb65b54eb"
API_BASE = "https://api.cellxgene.cziscience.com"
METHOD_ALIASES = {"10x 3' v2": "droplet", "Smart-seq2": "facs"}


def _get_json(url, retries=5, sleep=0.1, backoff=2):
    import requests
    try:
        res = requests.get(url=url, headers={"Content-Type": "application/json"})
        return res.json()
    except Exception:
        if retries > 0:
            time.sleep(sleep)
            return _get_json(url, retries - 1, sleep * backoff, backoff)
        raise


def _matching(dataset, method_list, organ_list):
    if len(dataset["assay"]) > 1 or len(dataset["tissue"]) > 1:
        return False
    method = METHOD_ALIASES.get(dataset["assay"][0]["label"])
    if method is None:
        return False
    if organ_list and dataset["tissue"][0]["label"].lower() not in organ_list:
        return False
    if method_list and method not in method_list:
        return False
    return True


def _download_dataset(dataset):
    import scanpy as sc
    assets = [a for a in dataset.get("assets", []) if a["filetype"] == "H5AD"]
    assert len(assets) == 1, f"Expected 1 H5AD asset, got {len(assets)}"
    url = assets[0]["url"]
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, os.path.basename(url))
        scprep.io.download.download_url(url, filepath)
        adata = sc.read_h5ad(filepath)
    utils.filter_genes_cells(adata)
    if getattr(adata, "raw", None) is not None:
        return adata.raw.to_adata()
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
    return adata


@utils.loader(
    data_url="https://tabula-muris-senis.ds.czbiohub.org/",
    data_reference="tabula2020single",
)
def load_tabula_muris_senis(test=False, method_list=None, organ_list=None):
    """Load Tabula Muris Senis lung droplet data from CellXGene."""
    method_list = [x.lower() for x in (method_list or [])]
    organ_list = [x.lower() for x in (organ_list or [])]

    datasets = _get_json(f"{API_BASE}/curation/v1/collections/{COLLECTION_ID}")["datasets"]
    matched = [d for d in datasets if _matching(d, method_list, organ_list)]
    assert len(matched) > 0, f"No datasets matched organ_list={organ_list} method_list={method_list}"

    adatas = [_download_dataset(d) for d in matched]
    adata = ad.concat(adatas, join="outer")
    if "is_primary_data" in adata.obs.columns:
        del adata.obs["is_primary_data"]

    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
    adata.X = adata.layers["counts"]

    if test:
        import scanpy as sc
        sc.pp.subsample(adata, n_obs=min(500, adata.n_obs))
        adata = adata[:, :1000]
        utils.filter_genes_cells(adata)

    utils.filter_genes_cells(adata)
    return adata
