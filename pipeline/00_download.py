"""Download the Criteo uplift dataset (311 MB) from Hugging Face into data/raw."""
from huggingface_hub import hf_hub_download
from config import ROOT

p = hf_hub_download("criteo/criteo-uplift", "criteo-research-uplift-v2.1.csv.gz",
                    repo_type="dataset", local_dir=str(ROOT / "data" / "raw"))
print(p)
