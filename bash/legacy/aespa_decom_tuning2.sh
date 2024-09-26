stanford_path="/data/NeuLF_rgb/stanford_half"

test_day="240823_decom_tuning_"
result_path="/data/result/${test_day}"

width="256"
epoch="10"

datasets=( "knights" "bracelet" "bunny" "tarot")
batch_sizes=("8192" "65536")
depths=("2")

lrs=("0.005" "0.001" "0.0005" "0.0001")

nomlin=("relu_decom")

for depth in "${depths[@]}"; do
    for dataset in "${datasets[@]}"; do
        for batch_size in "${batch_sizes[@]}"; do
            for nonlin in "${nomlin[@]}"; do  # 여기를 수정
                for lr in "${lrs[@]}"; do
                    echo "Processing $dataset , $nonlin , $batch_size , $depth , $lr"
                    python nlf_decompose_epi.py \
                        --data_dir "${stanford_path}/${dataset}" \
                        --exp_dir "${result_path}/${test_day}_epi_${nonlin}_${depth}_${batch_size}_${dataset}" \
                        --depth $depth \
                        --width $width \
                        --whole_epoch $epoch \
                        --test_freq 1 \
                        --nonlin $nonlin \
                        --lr $lr \
                        --benchmark \
                        --batch_size $batch_size \
                        --gpu 1

                    python asem_json.py "/data/result/${test_day}" "result_json/${test_day}"
                done
            done
        done
    done
done
