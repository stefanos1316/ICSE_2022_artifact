python ./launch.py --model resnet50 --precision AMP --mode benchmark_inference --platform DGX1V /imagenet --epochs 1 -b 32 --prof 100 --mixup 0.0 --workspace ./ --raport-file raport.json
