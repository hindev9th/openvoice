from cached_path import cached_path
from huggingface_hub import hf_hub_download

DOWNLOAD_CKPT_URL = 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/EN/checkpoint.pth'

DOWNLOAD_CONFIG_URL = 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/EN/config.json'

CONVERT_TO_HF_REPO_ID = 'nguyenhienlnh/openvoice_convert'

def load_or_download_config(use_hf=True, config_path=None):
    if config_path is None:
        if use_hf:
            config_path = hf_hub_download(repo_id=CONVERT_TO_HF_REPO_ID, filename="config.json")
        else:
            config_path = cached_path(DOWNLOAD_CONFIG_URL)
    return config_path

def load_or_download_model( use_hf=True, ckpt_path=None):
    if ckpt_path is None:
        if use_hf:
            ckpt_path = hf_hub_download(repo_id=CONVERT_TO_HF_REPO_ID, filename="checkpoint.pth")
        else:
            ckpt_path = cached_path(DOWNLOAD_CKPT_URL)
    return ckpt_path

def load_or_download_speaker(use_hf=True, speaker_id=None):
    if config_path is None:
        config_path = hf_hub_download(repo_id=CONVERT_TO_HF_REPO_ID, filename=f"base_speakers/ses/{speaker_id}.pth")
    return config_path