stanford_path="/data/NeuLF_rgb/stanford_half"

test_day="240910_all_decom_tuned_with_schdule_type"
result_path="/data/result/${test_day}"

depth="0"
width="3"
coord_depths=("2" )
coord_widths=("1024")
epoch="1000"

#datasets=( "knights" "bracelet" "bunny" "tarot")
datasets=( "knights" )
batch_sizes=("65536" )
schdule_types=("linear")
Rs=("1")

decom_dims=("uv")
lrs=("0.0005")

nomlin=("relu")

for R in "${Rs[@]}"; do
    for coord_depth in "${coord_depths[@]}"; do
        for coord_width in "${coord_widths[@]}"; do
            for dataset in "${datasets[@]}"; do
                for nonlin in "${nomlin[@]}"; do  # 여기를 수정
                    for lr in "${lrs[@]}"; do
                        for batch_size in "${batch_sizes[@]}"; do
                            for decom_dim in "${decom_dims[@]}"; do
                                for schdule_type in "${schdule_types[@]}"; do

                                    echo "Processing $dataset , $nonlin , $coord_depth , $coord_width ,$decom_dim"
                                    python nlf_decom_3decom.py \
                                        --data_dir "${stanford_path}/${dataset}" \
                                        --exp_dir "${result_path}/${test_day}_${nonlin}_d${depth}_w${width}_cd${coord_depth}_cd${coord_width}_R${R}_${batch_size}_decom_dim_${decom_dim}_lr${lr}sample_type_${sample_type}_schdule_type_${schdule_type}_${dataset}" \
                                        --depth $depth \
                                        --width $width \
                                        --coord_depth $coord_depth \
                                        --coord_width $coord_width \
                                        --whole_epoch $epoch \
                                        --test_freq 10 \
                                        --nonlin $nonlin \
                                        --lr $lr \
                                        --benchmark \
                                        --batch_size $batch_size \
                                        --gpu 0 \
                                        --decom_dim $decom_dim \
                                        --R $R \
                                        --schdule_type $schdule_type\
                                        #--sample_type $sample_type

                                    python asem_json.py "/data/result/${test_day}" "result_json/${test_day}"
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
