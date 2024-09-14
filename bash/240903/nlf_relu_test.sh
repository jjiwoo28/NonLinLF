stanford_path="/data/hmjung/NeuLF_rgb/dataset/stanford_half"


test_day="240913_nlf_relu_test"
result_path="/data/result/${test_day}"


depths=("4" "6" "8" "2")
widths=("256")

epoch="300"

#datasets=( "knights" "bracelet" "bunny" "tarot")
datasets=( "knights" )
batch_sizes=("8192")
Rs=("1")

decom_dims=("uv")
lrs=("0.0005")

nomlin=("relu")

for R in "${Rs[@]}"; do
    for depth in "${depths[@]}"; do
        for width in "${widths[@]}"; do
            for dataset in "${datasets[@]}"; do
                for nonlin in "${nomlin[@]}"; do  # 여기를 수정
                    for lr in "${lrs[@]}"; do
                        for batch_size in "${batch_sizes[@]}"; do
                            
                            echo "Processing $dataset , $nonlin , $depth , $width ,$epoch"
                            python wire_lf_new.py \
                                --data_dir "${stanford_path}/${dataset}" \
                                --exp_dir "${result_path}/${test_day}_${nonlin}_d${depth}_w${width}_${batch_size}_lr${lr}_${dataset}" \
                                --depth $depth \
                                --width $width \
                                --whole_epoch $epoch \
                                --test_freq 10 \
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
    done
done
