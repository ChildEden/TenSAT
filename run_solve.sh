grp=1
date="1001"
task="solve"

if [ ! -d "run/${task}_1009" ]; then
  mkdir -p "run/${task}_1009"
fi

#rm -rf run/$task/for_plot/
if [ ! -d "run/${task}_1009/for_plot" ]; then
  mkdir -p "run/${task}_1009/for_plot"
fi

#for baseModel in "neurosat" "nnsat"; do
for baseModel in "nnsat"; do
#  for rnnType in "rnn" "gru" "lstm"; do
  for rnnType in "lstm"; do
    for normal in 1; do
#    for normal in 0 1; do

      if [ $normal == 1 ]; then
        normalStr="normal"
        if [ $baseModel == "neurosat" ]; then
          modelName="ggcn"
        else
          modelName="nnsat-ggcn"
        fi
      else
        normalStr="non-normal"
        modelName=$baseModel
      fi

      singleFolder="run/${task}_1009/for_plot/ep50_round24_${modelName}-${rnnType}_${date}_g${grp}"
      if [ ! -d "$singleFolder" ]; then
        mkdir -p $singleFolder
      fi
      echo $singleFolder
      echo "model/${baseModel}_${rnnType}_${normalStr}_${date}_g${grp}_sr5to10_pairs10000_ep50_nr24_d128_best.pth.tar"

      for nv in 30 40 50 60 70 80; do
        for iterRound in 10 20 30 40 50 60; do
          python3 "src/${task}_${baseModel}.py" \
            --task-name "${modelName}-${rnnType}_${date}_g${grp}" \
            --is-normal $normal \
            --rnn-type $rnnType \
            --dim 128 \
            --n_rounds $iterRound \
            --restore "model/group_12/${baseModel}_${rnnType}_${normalStr}_${date}_g${grp}_sr5to10_pairs10000_ep50_nr24_d128_best.pth.tar" \
            --data-dir "data/test/grp$grp/sr$nv" \
            --output-dir $singleFolder
        done
      done

    done
  done
done
