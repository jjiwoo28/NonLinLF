stanford_path="/data/NeuLF_rgb/stanford_half"


test_day="240823_benchmark_"
result_path="/data/result/${test_day}"

width="256"
epoch="10"



datasets=("knights")  
nonlins=("relu" "relu_skip2" "wire" "siren" "gauss")
batch_sizes=("8192" "65536")
depths=("2" "4" "8")

#nomlin=("relu_decom" "wire_decom")

#relu
for depth in "${depths[@]}"; do
    for dataset in "${datasets[@]}"; do
        for batch_size in "${batch_sizes[@]}"; do
            for nonlin in "${nonlins[@]}"; do
                echo "Processing $dataset , $nonlin , $batch_size , $depth"
                python wire_lf_new.py \
                    --data_dir "${stanford_path}/${dataset}" \
                    --exp_dir "${result_path}/${test_day}_${nonlin}_${depth}_${batch_size}_${dataset}" \
                    --depth $depth \
                    --width $width \
                    --whole_epoch $epoch \
                    --test_freq 1 \
                    --nonlin $nonlin \
                    --benchmark \
                    --batch_size $batch_size

                python asem_json.py  "/data/result/${test_day}" "result_json/${test_day}"
            done
        done
    done
done

test_day="240823_benchmark_"
result_path="/data/result/${test_day}"

width="256"
epoch="10"



datasets=("knights")  
batch_sizes=("8192" "65536")
depths=("2" "4" "8")

nomlin=("relu_decom" "wire_decom")





for depth in "${depths[@]}"; do
    for dataset in "${datasets[@]}"; do
        for batch_size in "${batch_sizes[@]}"; do
            for nonlin in "${nonlins[@]}"; do
                echo "Processing $dataset , $nonlin , $batch_size , $depth"
                python nlf_decompose.py \
                    --data_dir "${stanford_path}/${dataset}" \
                    --exp_dir "${result_path}/${test_day}_${nonlin}_${depth}_${batch_size}_${dataset}" \
                    --depth $depth \
                    --width $width \
                    --whole_epoch $epoch \
                    --test_freq 1 \
                    --nonlin $nonlin \
                    --benchmark \
                    --batch_size $batch_size

                python asem_json.py  "/data/result/${test_day}" "result_json/${test_day}"
            done
        done
    done
done




