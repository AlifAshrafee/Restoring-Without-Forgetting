from huggingface_hub import upload_folder, HfApi
from huggingface_hub import upload_file

api = HfApi()
repo_id = "AlifAshrafee/shift-variant-blur-div2k"

for variant in range(1, 7):
    print(f"\nUploading variant: {variant}")
    print('-'*50)

    upload_folder(
        folder_path=f"datasets/shift_variant_blur/variant_{variant}",
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo=f"variant_{variant}",
    )
    print(f"✓ Variant {variant} uploaded successfully")

upload_file(
    path_or_fileobj="datasets/shift_variant_blur/dataset_info.json",
    path_in_repo="dataset_info.json",
    repo_id=repo_id,
    repo_type="dataset",
)

print("✓ Dataset metadata uploaded")