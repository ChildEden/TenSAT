grp=1
date="1025"
#baseModel="nnsat"
#normal=1
#rnnType="lstm"

if [ ! -d "run/test_1025" ]; then
  mkdir -p run/test_1025
fi

#rm -rf run/test_1025/for_plot/
if [ ! -d "run/test_1025/for_plot" ]; then
  mkdir -p run/test_1025/for_plot
fi

for baseModel in "neurosat" "nnsat"; do
  for rnnType in "rnn" "gru" "lstm"; do
   for normal in 0 1; do

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

      singleFolder="run/test_1025/for_plot/ep25_round24_${modelName}-${rnnType}_${date}_g${grp}"
      if [ ! -d "$singleFolder" ]; then
        mkdir -p $singleFolder
      fi
      echo $singleFolder
      echo "model/${baseModel}_${rnnType}_${normalStr}_${date}_g${grp}_sr5to10_pairs50000_ep25_nr24_d128_best.pth.tar"

      for nv in 30 40 50 60 70 80; do
        for iterRound in 10 20 30 40 50 60; do
          python3 "src/test_${baseModel}.py" \
            --task-name "${modelName}-${rnnType}_${date}_g${grp}" \
            --is-normal $normal \
            --rnn-type $rnnType \
            --dim 128 \
            --n_rounds $iterRound \
            --restore "model/group_12_1025/${baseModel}_${rnnType}_${normalStr}_${date}_g${grp}_sr5to10_pairs50000_ep25_nr24_d128_best.pth.tar" \
            --data-dir "data/test/grp$grp/sr$nv" \
            --output-dir $singleFolder
#            --test-temperature 0
        done
      done

    done
  done
done
