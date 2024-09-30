stanford_path="/data/NeuLF_rgb/stanford_half" #데이터셋 경로


test_day="240930_all_decom_nonlin" #실험결과 파일 (json파일 ) 중간 이름을 결정합니다.
result_path="/data/result/${test_day}" #실험 결과 파일 저장 경로 , 기본적으로는 최상위 data 폴더의 하위 폴더로 지정되어있습니다.


depths=("4" "6" "8" "2")
depths=("0")
widths=("256") 
coord_depths=("8")
coord_widths=("256" "512")
coord_widths=("256")
epoch="500"

#datasets=( "knights" "bracelet" "bunny" "tarot")
datasets=( "knights" )
batch_sizes=("8192" "65536" "131072" "262144")
batch_sizes=("8192")
Rs=("2" "3")
Rs=("1")

decom_dims=("uv")
lrs=("0.005"  "0.001" "0.0005"  "0.0001"  "0.00005"   "0.00001")
lrs=("0.0005")

nomlin=("gauss" "siren" "finer" "relu")
nomlin=("finer")
nomlin=("siren" "finer" "gauss" )

for R in "${Rs[@]}"; do
    for depth in "${depths[@]}"; do
        for width in "${widths[@]}"; do
            for coord_width in "${coord_widths[@]}"; do
                for coord_depth in "${coord_depths[@]}"; do
                    for dataset in "${datasets[@]}"; do
                        for nonlin in "${nomlin[@]}"; do  
                            for lr in "${lrs[@]}"; do  
                                for batch_size in "${batch_sizes[@]}"; do
                                    for decom_dim in "${decom_dims[@]}"; do
                                        echo "Processing $dataset , $nonlin , $depth , $width , $lr "
                                        echo "Processing  $coord_depth , $coord_width ,$decom_dim"
                                        python nlf_decom_3decom.py \
                                            --data_dir "${stanford_path}/${dataset}" \
                                            --exp_dir "${result_path}/${test_day}_${nonlin}_d${depth}_w${width}_cd${coord_depth}_cd${coord_width}_R${R}_${batch_size}_decom_dim_${decom_dim}_lr${lr}_${dataset}" \
                                            --depth $depth \
                                            --width $coord_width \
                                            --coord_depth $coord_depth \
                                            --coord_width $coord_width \
                                            --whole_epoch $epoch \
                                            --test_freq 1 \
                                            --nonlin $nonlin \
                                            --lr $lr \
                                            --benchmark \
                                            --batch_size $batch_size \
                                            --gpu 1 \
                                            --schdule_type linear\
                                            --decom_dim $decom_dim \
                                            --R $R \
                                            --lr_batch_preset

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
done
