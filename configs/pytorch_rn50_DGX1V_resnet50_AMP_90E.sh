python ./launch.py --model resnet50 --precision AMP --mode benchmark_training --platform DGX1V /imagenet --epochs 4 -b 32 --prof 1851 --mixup 0.0 --workspace ./ --raport-file raport.json
