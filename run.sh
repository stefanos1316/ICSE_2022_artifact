#!/bin/sh

# A script that executes all of our expriments
# and collects the required measurements

# Text fonts for Linux distros
bold=$(tput bold)
underline=$(tput smul)
default=$(tput sgr0)
greenlabel=$(tput setab 2)
redlabel=$(tput setab 1)
yellowlabel=$(tput setab 3)

# Set default values
repetitions=1
test_type="train"

# Help
help_info()
{
  echo "-r <repeitions number> or --repetitions <repeitions number> are used to define the number of repetitions to run each task"
  echo "-t <train | infer> or --test <train | infer> to perform the correspoding type of test"
  exit
}

# Log with a timestamp
log()
{
  # Output is redirected to the log file if needed at the script's lop level
  date +'%F %T ' | tr -d \\n 1>&2
  echo "$@" 1>&2
}

# Function that executes
# $1 is the name of the task (e.g., Transformer-XL)
# $2 is the name of the framework (e.g., PyTorch)
# $3 is the number of times to run each task
# $4 is the command to execute a task for the corresponding task ($1) that is written with a framework ($2)
collect_energy_measurements()
{
  log "Obtaining energy and run-time performance measurements"
 
  for i in $(seq 1 $3); do
    # Collect the energy consumption of the GPU
    nvidia-smi -i 0 --loop-ms=1000 --format=csv,noheader --query-gpu=power.draw >> "$measurements"/"$1"_"$2"_nvidia_smi_"$i".txt &

    # Get nvidia-smi's PID
    nvidia_smi_PID=$!
    
    # Collect the energy consumption of the processor package and main memory
    perf stat -e power/energy-pkg/,power/energy-ram/ $4 2>> "$measurements"/"$1"_"$2"_perf_"$i".txt

    # Remove cached model
    if [ "$1" == "transformer_xl" ]; then
      rm data/wikitext-103/cache.pt
    fi

    if [ "$1" == "gnmt" ]; then
      rm -rf results
    fi

    # When the experiment is elapsed, terminate the nvidia-smi process
    kill -9 "$nvidia_smi_PID"

    log "Small sleep time to reduce power tail effecs"
    sleep 60
  
  done
}

# Get command-line arguments
OPTIONS=$(getopt -o r:t: --long repetitions:test -n 'run_experiments' -- "$@")
eval set -- "$OPTIONS"
while true; do
  case "$1" in
    -r|--repetitions) repetitions="$2"; shift 2;;
    -t|--test) test_type="$2"; shift 2;;
    -h|--help) help_info; shift;;
    --) shift; break;;
    *) >&2 log "${redlabel}[ERROR]${default} Wrong command line argument, please try again."; exit 1;;
  esac
done

# Switching to perfomrance mode
log "Switching to performance mode"
sudo ./governor.sh pe

# Test can be train or inference
if [ "$test_type" == "train" ]; then
  measurements="$PWD/measurements_train"
else
  measurements="$PWD/measurements_inference"
fi

[ -d "$measurements" ] && rm -rf "$measurements" && mkdir "$measurements"
[ ! -d "$measurements" ] && mkdir "$measurements"

# Go into the DeepLearning Examples repository
cd DeepLearningExamples

declare -a arr=("PyTorch" "TensorFlow")

# Executing SSD for PyTorch and TensorFlow
for i in "${arr[@]}"; do
  log "Executing SSD for $i"
  cd "$i"/Detection/SSD/
  docker build . -t nvidia_ssd

  if [ "$i" == "PyTorch" ]; then
    if [ "$test_type" == "train" ]; then
      log "Training SSD for PyTorch"
      collect_energy_measurements "ssd" "$i" "$repetitions" "nvidia-docker run --rm -it --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}/coco/:/coco/ --ipc=host nvidia_ssd python ./main.py --backbone resnet50 --mode training --bs 8 --epochs 1 --seed 1 --amp --data /coco"
    else
      log "Testing SSD for PyTorch"
      collect_energy_measurements "ssd" "$i" "$repetitions" "nvidia-docker run --rm -it --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}/coco/:/coco/ --ipc=host nvidia_ssd python ./main.py --backbone resnet50 --mode evaluation --checkpoint ./pre_train_model/epoch_*.pt --eval-batch-size 8 --epochs 1 --seed 1 --amp --data /coco"
    fi
    cd ../../../   
  else
    if [ "$test_type" == "train" ]; then
      log "Training SSD for TensorFlow"
      collect_energy_measurements "ssd" "$i" "$repetitions" "nvidia-docker run --rm -it --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}/data/:/data/coco2017_tfrecords -v ${PWD}/checkpoints/:/checkpoints --ipc=host nvidia_ssd bash ./examples/SSD320_FP16_1GPU.sh"
    else
      log "Testing SSD for TensorFlow"
      collect_energy_measurements "ssd" "$i" "$repetitions" "nvidia-docker run --rm -it --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}/data/:/data/coco2017_tfrecords -v ${PWD}/checkpoints/:/checkpoints --ipc=host nvidia_ssd bash ./examples/SSD320_evaluate.sh"
    fi
    cd ../../../
  fi
done

# Executing GNMT for PyTorch and TensorFlow
for i in "${arr[@]}"; do
  log "Executing GNMT for $i"
  cd "$i"/Translation/GNMT/
  bash scripts/docker/build.sh
  
  if [ "$i" == "PyTorch" ]; then
    if [ "$test_type" == "train" ]; then
      log "Training GNMT for PyTorch"
      collect_energy_measurements "gnmt" "$i" "$repetitions" "docker run --gpus all --init -it --rm --network=host --ipc=host -v $PWD:/workspace/gnmt/ gnmt python3 -m torch.distributed.launch --nproc_per_node=1  train.py --seed 1 --train-global-batch-size 128"
    else
      log "Testing GNMT for PyTorch"
      collect_energy_measurements "gnmt" "$i" "$repetitions" "docker run --gpus all --init -it --rm --network=host --ipc=host -v $PWD:/workspace/gnmt/ gnmt python3 translate.py --model gnmt/model_best.pth --input data/wmt16_de_en/newstest2014.en --reference data/wmt16_de_en/newstest2014.de --output /tmp/output --batch-size 128 --tables"
    fi
    cd ../../../
  else
    if [ "$test_type" == "train" ]; then
      log "Training GNMT for TensorFlow"
      collect_energy_measurements "gnmt" "$i" "$repetitions" "nvidia-docker run -it --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v $PWD:/workspace/gnmt/ gnmt_tf python nmt.py --output_dir=results --batch_size=128 --learning_rate=2e-3"
    else
      log "Testing GNMT for TensorFlow"
      collect_energy_measurements "gnmt" "$i" "$repetitions" "nvidia-docker run -it --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v $PWD:/workspace/gnmt/ gnmt_tf python nmt.py --mode=infer --output_dir=results --infer_batch_size=128 --learning_rate=2e-3"
    fi
    cd ../../../
  fi
done

# Execute Transformer-XL for PyToch and TensorFlow
for i in "${arr[@]}"; do
  log "Executing Transofrmer-XL for $i"
  cd "$i"/LanguageModeling/Transformer-XL/

  if [ "$i" == "PyTorch" ]; then
    bash pytorch/scripts/docker/build.sh
    if [ "$test_type" == "train" ]; then
      log "Training Transformer-XL for PyTorch"
      collect_energy_measurements "transformer_xl" "$i" "$repetitions" "docker run --gpus all --init -it --rm --network=host --ipc=host -v $PWD:/workspace/transformer-xl transformer-xl bash run_wt103_base.sh train 1"
    else
      log "Testing Transformer-XL for PyTorch"
      collect_energy_measurements "transformer_xl" "$i" "$repetitions" "docker run --gpus all --init -it --rm --network=host --ipc=host -v $PWD:/workspace/transformer-xl transformer-xl bash run_wt103_base.sh eval 1"
    fi
    cd ../../../
  else
    bash tf/scripts/docker/build.sh
    if [ "$test_type" == "train" ]; then
      log "Training Transformer-XL for TensorFlow"
      collect_energy_measurements "transformer_xl" "$i" "$repetitions" "docker run --gpus all --init -it --rm --network=host --ipc=host -v $PWD:/workspace/transformer-xl transformer-xl bash run_wt103_base.sh train 1"
    else
      log "Testing Transformer-XL for TensorFlow"
      collect_energy_measurements "transformer_xl" "$i" "$repetitions" "docker run --gpus all --init -it --rm --network=host --ipc=host -v $PWD:/workspace/transformer-xl transformer-xl bash run_wt103_base.sh eval 1"
    fi
    cd ../../../
  fi
done

# Executing NCF for PyTorch and TensorFlow
for i in "${arr[@]}"; do
  log "Executing NCF for $i"
  cd "$i"/Recommendation/NCF/
  docker build . -t nvidia_ncf
  if [ "$i" == "PyTorch" ]; then
    if [ "$test_type" == "train" ]; then
      log "Training NCF for PyTorch"
      collect_energy_measurements "ncf" "$i" "$repetitions" "docker run --runtime=nvidia -it --rm --ipc=host -v ${PWD}/data:/data nvidia_ncf python -m torch.distributed.launch --nproc_per_node=1 --use_env ncf.py --data ../../data/cache/ml-20m --checkpoint_dir ../../data/checkpoints/  -s 1 -e 1 -f 1 -b 64 --mode train"
    else
      log "Testing NCF for PyTorch"
      collect_energy_measurements "ncf" "$i" "$repetitions" "docker run --runtime=nvidia -it --rm --ipc=host -v ${PWD}/data:/data nvidia_ncf python -m torch.distributed.launch --nproc_per_node=1 --use_env ncf.py --data ../../data/cache/ml-20m --checkpoint_dir ../../data/checkpoints/  -s 1 -e 1 -f 1 -b 64 --valid_batch_size 64 --mode test"
    fi
    cd ../../../
  else
    if [ "$test_type" == "train" ]; then
      log "Training NCF for TensorFlow"
      collect_energy_measurements "ncf" "$i" "$repetitions" "docker run --runtime=nvidia -it --rm --ipc=host -v ${PWD}/data:/data nvidia_ncf mpirun -np 1 --allow-run-as-root python ncf.py --amp --data ../../data/cache/ml-20m --checkpoint-dir ../../data/checkpoints/ -s 1 -e 1 -f 1 -b 64 --mode train"
    else
      log "Testing NCF for TensorFlow"
      collect_energy_measurements "ncf" "$i" "$repetitions" "docker run --runtime=nvidia -it --rm --ipc=host -v ${PWD}/data:/data nvidia_ncf mpirun -np 1 --allow-run-as-root python ncf.py --amp --data ../../data/cache/ml-20m --checkpoint-dir ../../data/checkpoints/ -s 1 -e 1 -f 1 -b 64 --mode test"
    fi
    cd ../../../
  fi
done

# Executing Resnet50 for PyTorch and TensorFlow
for i in "${arr[@]}"; do
  log "Executing ResNet50 for $i"
  cd "$i"/Classification/ConvNets
 # docker build . -t nvidia_rn50

  if [ "$i" == "PyTorch" ]; then
    if [ "$test_type" == "train" ]; then
      log "Training ResNet50 for PyTorch"
      collect_energy_measurements "resnet50" "$i" "$repetitions" "nvidia-docker run --rm -it -v ${PWD}/imagenet/:/imagenet --ipc=host nvidia_rn50 python ./launch.py --model resnet50 --precision AMP --mode benchmark_training --platform DGX1V /imagenet --epochs 4 -b 32 --prof 1851 --mixup 0.0 --workspace ./ --raport-file raport.json"
    else
      log "Testing ResNet50 for PyTorch"
      collect_energy_measurements "resnet50" "$i" "$repetitions" "nvidia-docker run --rm -it -v ${PWD}/imagenet/:/imagenet --ipc=host nvidia_rn50 python ./launch.py --model resnet50 --precision AMP --mode benchmark_inference --platform DGX1V /imagenet --prof 100 --epochs 50 -b 32  --workspace ./ --raport-file inference.json"
    fi
    cd ../../../   
  else
    if [ "$test_type" == "train" ]; then
      log "Training ResNet50 for TensorFlow"
      collect_energy_measurements "resnet50" "$i" "$repetitions" "nvidia-docker run --rm -it -v ${PWD}/imagenet/:/data/tfrecords --ipc=host nvidia_rn50 bash resnet50v1.5/training/DGX1_RN50_AMP_90E.sh"
    else
      log "Training ResNet50 for TensorFlow"
      collect_energy_measurements "resnet50" "$i" "$repetitions" "nvidia-docker run --rm -it -v ${PWD}/imagenet/:/data/tfrecords --ipc=host nvidia_rn50 python main.py --mode=inference_benchmark --arch=resnet50 --num_iter=5000 --iter_unit batch --batch_size 32  --results_dir=./"
    fi
    cd ../../../
  fi
done

# Executing MaskRCNN for PyTorch and TensorFlow
for i in "${arr[@]}"; do
  log "Executing MaskRCNN for $i"
  cd "$i"/Segmentation/MaskRCNN/
  
  if [ "$i" == "PyTorch" ]; then
    cd pytorch && bash scripts/docker/build.sh && cd ..
    if [ "$test_type" == "train" ]; then
      log "Training MaskRCNN for PyTorch"
      collect_energy_measurements "mask_rcnn" "$i" "$repetitions" "docker run --runtime=nvidia -v $PWD/data:/datasets/data --rm --name=maskrcnn_interactice --shm-size=10g --ulimit memlock=-1 --ulimit stack=67108864 --ipc=host -t -i nvidia_joc_maskrcnn_pt python -m torch.distributed.launch tools/train_net.py --skip-test --amp --config-file configs/e2e_mask_rcnn_R_50_FPN_1x_1GPU.yaml"
    else
      log "Testing MaskRCNN for PyTorch"
      collect_energy_measurements "mask_rcnn" "$i" "$repetitions" "docker run --runtime=nvidia -v $PWD/data:/datasets/data --rm --name=maskrcnn_interactice --shm-size=10g --ulimit memlock=-1 --ulimit stack=67108864 --ipc=host -t -i nvidia_joc_maskrcnn_pt python -m torch.distributed.launch tools/test_net.py --amp --config-file configs/e2e_mask_rcnn_R_50_FPN_1x_1GPU.yaml"
    fi
    cd ../../../
  else
    nvidia-docker build -t nvidia_mrcnn_tf2 .
    if [ "$test_type" == "train" ]; then
      log "Training MaskRCNN for TensorFlow"
      collect_energy_measurements "mask_rcnn" "$i" "$repetitions" "docker run --gpus 1 -it --rm --shm-size=10g --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}/dataset/data/:/data -v ${PWD}/dataset/weights/:/weights nvidia_mrcnn_tf2 python main.py train --epochs 1 --steps_per_epoch 1000 --amp --train_batch_size 4"
    else
      log "Testing MaskRCNN for TensorFlow"
      collect_energy_measurements "mask_rcnn" "$i" "$repetitions" "docker run --gpus 1 -it --rm --shm-size=10g --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}/dataset/data/:/data -v ${PWD}/dataset/weights/:/weights nvidia_mrcnn_tf2 python main.py infer --amp  --eval_samples 5000 --eval_batch_size 4"
    fi
    cd ../../../
  fi
done

log "Done with all tests"
return 0
