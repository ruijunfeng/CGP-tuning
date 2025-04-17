import json

# Function to read a JSON file
def read_json_file(file_path: str) -> dict:
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

# Function to write a JSON file
def write_json_file(data: list, file_path: str):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)
