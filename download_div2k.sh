#!/bin/bash

mkdir -p datasets/DIV2K
cd datasets/DIV2K

echo "Downloading DIV2K dataset"

# training images (800 images, ~3.5GB)
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip

# validation images (100 images, ~450MB) 
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip

echo "Extracting archives"
unzip DIV2K_train_HR.zip
unzip DIV2K_valid_HR.zip

rm DIV2K_train_HR.zip
rm DIV2K_valid_HR.zip

echo "DIV2K dataset downloaded!"
echo "Train: $(ls DIV2K_train_HR/*.png | wc -l) images"
echo "Valid: $(ls DIV2K_valid_HR/*.png | wc -l) images"

cd ../..
