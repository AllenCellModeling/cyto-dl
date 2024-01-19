import fire

from cyto_dl.utils.download_test_data import delete_test_data, download_test_data

if __name__ == "__main__":
    fire.Fire(download_test_data)
