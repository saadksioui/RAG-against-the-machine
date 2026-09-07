import ast
import pathlib

with open("src/models.py", 'r') as f:
    content = f.read()

print(list(pathlib.Path("data/raw").rglob("*.py")))