# Download data

. scripts/download_glove.sh
. scripts/download_quac_data.sh

# Download pretrained models into ./trained

. scripts/download_models.sh

# Install Python requirements

pip install -r requirements.txt

# Preprocess data

python -m utils.preprocess_quac --min_freq 10 && python -m utils.preprocess_quac --file_name val_v0.2.json --eval
