# Methodology

## Data sources
- Satellite: Sentinel-2 L2A (10 m)
- IoT sensors: Commercial probes at 5/15/30 cm
- Farm logs: Mobile app
- Lab samples: 200 samples

## Quality control
1. Range check
2. IQR outliers
3. Gap filling (cubic interpolation <= 48h)

## Model
CNN-LSTM with soft-attention fusion.

## Validation
- GroupKFold by farm.
- Block spatial CV.
