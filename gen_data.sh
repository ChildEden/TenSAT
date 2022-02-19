grp=1
for ty in "train" "val" "test" ; do
    rm -rf data/$ty/grp$grp
    mkdir -p data/$ty/grp$grp

    if [ ! -d "data/$ty/grp$grp" ]; then
      mkdir -p data/$ty/grp$grp
    fi

    if [ ! -d "log/$ty/grp$grp" ]; then
      mkdir -p log/$ty/grp$grp
    fi

    for group in $(seq $grp); do
      if [ $ty = "train" ]; then
        python3 src/data_maker.py data/$ty/grp$grp log/$ty/grp$grp 50000 12000 --group "$group" --min_n 5 --max_n 10
      fi

      if [ $ty = "val" ]; then
        python3 src/data_maker.py data/$ty/grp$grp log/$ty/grp$grp 15000 12000 --group "$group" --min_n 5 --max_n 10
      fi
    done

   if [ $ty = "test" ]; then
     rm -rf data/$ty/grp$grp
     mkdir -p data/$ty/grp$grp
     for nv in 30 40 50 60 70 80; do
       mkdir -p data/$ty/grp$grp/sr$nv
       for group in $(seq $grp); do
         python3 src/data_maker.py data/$ty/grp$grp/sr$nv log/data/$ty 10000 12000 --group "$group" --min_n $nv --max_n $nv
       done
     done
   fi

done;
