#!/bin/bash

mkdir -p data/coco-wholebody/images
mkdir -p data/coco-wholebody/annotations
cd ./data/coco-wholebody/images || exit

echo "Downloading Train2017 (18GB)..."
wget -c -N http://images.cocodataset.org/zips/train2017.zip

echo "Downloading Val2017 (1GB)..."
wget -c -N http://images.cocodataset.org/zips/val2017.zip

echo "Unzipping..."
unzip -q -n train2017.zip
unzip -q -n val2017.zip
