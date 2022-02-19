grp=1
baseModel="nnsat"
date="1025"
normal=1 # GGCN
#rnnType="lstm"

if [ $normal == 1 ]; then
  normalStr="normal"
else
  normalStr="non-normal"
fi

if [ ! -d "log/train" ]; then
  mkdir -p log/train
fi

for rnnType in "lstm" "rnn" "gru"; do
  echo "src/train_${baseModel}.py"
  echo "${baseModel}_${rnnType}_${normalStr}_${date}_g${grp}"

  python3 "src/train_${baseModel}.py" \
    --task-name "${baseModel}_${rnnType}_${normalStr}_${date}_g${grp}" \
    --epochs 25 \
    --n_pairs 50000 \
    --n_rounds 24 \
    --max_nodes_per_batch 12000 \
    --min_n 5 \
    --max_n 10 \
    --is-normal $normal \
    --rnn-type $rnnType \
    --model-dir "model/group_12_1025/" \
    --val-file "data/val/grp$grp" \
    --train-file "data/train/grp$grp" \
    --log-dir 'log/train'
done
