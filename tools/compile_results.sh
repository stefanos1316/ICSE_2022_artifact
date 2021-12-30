#!/bin/sh

if [ -z "$1" ]; then
  echo Please provide the path with the results/measurements
  exit
fi

declare -a models=("gnmt" "ncf" "transformer_xl" "mask_rcnn" "ssd" "resnet50")
declare -a frameworks=("PyTorch" "TensorFlow")
declare -a typeOfMeasurements=("perf" "nvidia_smi")

[ -f finalResults.txt ] && rm -f finalResults.txt

for i in "${models[@]}"; do
  for j in "${frameworks[@]}"; do
    for k in "${typeOfMeasurements[@]}"; do
	  
      totalPkgEnergy=0
      totalRamEnergy=0
      totalRunTime=0
      totalGpuEnergy=0
      for l in $(seq 1 10); do
  	
	      # Create file path of the current report
        currentReport=$1/${i}"_"${j}"_"${k}"_"${l}".txt"
        
        # Compile perf results
        if [ "$k" == "perf" ]; then
          # Grep the pkg measurements
          pkgEnergy=$(grep "energy-pkg" $currentReport | awk '{print $1}' | tr -d ',')
          ramEnergy=$(grep "energy-ram" $currentReport | awk '{print $1}' | tr -d ',')
          runTime=$(grep "seconds time" $currentReport | awk '{print $1}' | tr -d ',')
          
          tailPkgEnergy="$tailPkgEnergy, $pkgEnergy"
          tailRamEnergy="$tailRamEnergy, $ramEnergy"
          tailRunTime="$tailRunTime, $runTime"

          totalPkgEnergy=$(echo "$totalPkgEnergy + $pkgEnergy" | bc)
          totalRamEnergy=$(echo "$totalRamEnergy + $ramEnergy" | bc)
          totalRunTime=$(echo "$totalRunTime + $runTime" | bc)
        fi

        # Compile nvidia-smi results
        if [ "$k" == "nvidia_smi" ]; then
          # Calculate total energy consumption
          gpuEnergy=$(awk '{sum+=$1}END{printf "%.2f\n", sum}' "$currentReport")

	        tailGpuEnergy="$tailGpuEnergy, $gpuEnergy"

          totalGpuEnergy=$(echo "$totalGpuEnergy + $gpuEnergy" | bc)
        fi

      done
        # Get mean results
        meanPkgEnergy=$(echo "scale=2; $totalPkgEnergy / 10" | bc)
        meanRamEnergy=$(echo "scale=2; $totalRamEnergy / 10" | bc)
        meanGpuEnergy=$(echo "scale=2; $totalGpuEnergy / 10" | bc)
        meanRunTime=$(echo "scale=2; $totalRunTime / 10" | bc)
        total=$(echo "scale=2; $totalPkgEnergy + $totalRamEnergy" | bc)
	total_in_watts_perf=$(echo "scale=2; $total / $totalRunTime" | bc)
	totalTimeInHours=$(echo "scale=2; $totalRunTime / 3600 "| bc)

        # Print means to file
        if [ "$k" == "nvidia_smi" ]; then
          echo "GPU mean energy consumption: $meanGpuEnergy [$tailGpuEnergy]" >> finalResults.txt
          echo "total of GPU (in Watts): $totalGpuEnergy" >> finalResults.txt
          echo >> finalResults.txt
        else
          echo "Measurements for $i framework $j" >> finalResults.txt
          echo "Package mean energy consumption: $meanPkgEnergy  [$tailPkgEnergy]" >> finalResults.txt
          echo "RAM mean energy consumption: $meanRamEnergy [$tailRamEnergy]" >> finalResults.txt
          echo "Run-time performance: $meanRunTime [$tailRunTime]" >> finalResults.txt
	  echo "total of PKG and RAM (in Joules): $total" >> finalResults.txt
	  echo "total time (in seconds): $totalRunTime" >> finalResults.txt
	  echo "total time (in hours): $totalTimeInHours" >> finalResults.txt
	  echo "total watts in perf $total_in_watts_perf" >> finalResults.txt
        fi
      
	tailPkgEnergy=0
        tailRamEnergy=0
        tailGpuEnergy=0
        tailRunTime=0
    done
  done
done

echo Done, check results in ./finalResults.txt
exit
