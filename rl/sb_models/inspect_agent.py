import zipfile
import torch
import io

# Change this to your agent zip file path
zip_path = "bytefight_logs/ppo_bytefight_final.zip"

def inspect_zip(zip_path):
    with zipfile.ZipFile(zip_path, "r") as zf:
        file_list = zf.namelist()
        print("Files in the zip archive:")
        for file in file_list:
            print(" -", file)

        # Inspect system_info.txt if available
        if "system_info.txt" in file_list:
            print("\nContents of system_info.txt:")
            with zf.open("system_info.txt") as f:
                content = f.read().decode("utf-8")
                print(content)

        # Optionally, inspect the data file (if it's a text or small binary file)
        if "data" in file_list:
            print("\nContents of data file:")
            with zf.open("data") as f:
                # Attempt to decode as text; if not, show first few bytes
                try:
                    content = f.read().decode("utf-8")
                    print(content)
                except UnicodeDecodeError:
                    f.seek(0)
                    print("Binary data (first 100 bytes):", f.read(100))

        # List of .pth files to inspect
        pth_files = ["policy.pth", "policy.optimizer.pth", "pytorch_variables.pth"]
        for pth_file in pth_files:
            if pth_file in file_list:
                print(f"\nLoading {pth_file}:")
                with zf.open(pth_file) as f:
                    # torch.load expects a file-like object, so we wrap the bytes in a BytesIO stream
                    buffer = io.BytesIO(f.read())
                    try:
                        data = torch.load(buffer, map_location="cpu")
                    except Exception as e:
                        print(f"Error loading {pth_file}: {e}")
                        continue
                    # Display basic info about the loaded object
                    print("Type:", type(data))
                    if isinstance(data, dict):
                        print("Keys:", list(data.keys()))
                    else:
                        print(data)

if __name__ == "__main__":
    inspect_zip(zip_path)
