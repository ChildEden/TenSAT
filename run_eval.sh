python3 src/eval.py \
  --task-name 'neurosat_eval_sr4' \
  --dim 128 \
  --n_rounds 16 \
  --restore 'model/dev_sr5to10_pairs10000_ep25_nr16_d128_best.pth.tar' \
  --data-dir 'data/test'
