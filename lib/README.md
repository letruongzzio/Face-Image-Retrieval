# Instruction

```bash
# Prepare a separate conda env
conda create --name new_env_name --file requirements.txt

# Download data from Kaggle, you need to authenticate the kaggle API beforehand
python3 prepare.py

# Process data
python3 process.py

# Build the embedding query tree
python3 embedding.py
```