# Robust Scene Text Recognition Through Adaptive Image Enhancement

Official PyTorch implementation of our TextREN.

## Dependendency
```
conda env create -f environment.yml
```

## Training
```
bash scripts/stn_att_rec.sh
```

## Evaluation
Test with LMDB dataset by
```
bash scripts/main_test_all.sh
```
Test with single image by
```
bash scripts/main_test_image.sh
```

## Acknowledgements
This implementation has been based on this repository [Aster.pytorch](github.com/ayumiymk/aster.pytorch)