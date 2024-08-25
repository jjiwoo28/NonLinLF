stanford_path="/data/hmjung/NeuLF_rgb/dataset/stanford_half"




test_day="240808_wire_d4w256_"
result_path="/data/result/${test_day}"
depth="4"
width="256"
epoch="300"

datasets=( "knights" "bracelet" "bunny" "tarot")  
batch_sizes=("32768" "65536" "262144" "524288" "1048576")

result_path="/data/result/${test_day}"

for dataset in "${datasets[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do
        
        echo "Processing $dataset"
        
        python wire_lf_new.py --data_dir "${stanford_path}/${dataset}" --exp_dir "${result_path}/${dataset}_batchsize_${batch_size}" --depth $depth --width $width --whole_epoch $epoch --test_freq 10 --nonlin wire --lr 5e-3 --batch_size $batch_size
        
        python asem_json.py  "/data/result/${test_day}" "result_json/${test_day}"

    done
done

# test_day="240808_relu_skip2_d4w256_"
# depth="4"
# width="256"
# epoch="300"
# result_path="/data/result/${test_day}"

# datasets=( "knights" "bracelet" "bunny" "tarot")  
# batch_sizes=("2048" "4092" "16384" )

# for batch_size in "${batch_sizes[@]}"; do
#     for dataset in "${datasets[@]}"; do
        
#         echo "Processing $dataset"
        
#         python wire_lf_new.py --data_dir "${stanford_path}/${dataset}" --exp_dir "${result_path}/${dataset}_batchsize_${batch_size}" --depth $depth --width $width --whole_epoch $epoch --test_freq 10 --nonlin relu_skip2 --lr 4e-4 --batch_size $batch_size
        
#         python asem_json.py  "/data/result/${test_day}" "result_json/${test_day}"

#     done
# done

# python asem_json.py  "/data/result/${test_day}" "result_json/${test_day}"


# test_day="240808_wire_d8w256_lr_1e-3"
# depth="8"
# width="256"
# epoch="300"



# result_path="/data/result/${test_day}"

# for batch_size in "${batch_sizes[@]}"; do
#     for dataset in "${datasets[@]}"; do
        
#         echo "Processing $dataset"
        
#         python wire_lf_new.py --data_dir "${stanford_path}/${dataset}" --exp_dir "${result_path}/${dataset}_batchsize_${batch_size}" --depth $depth --width $width --whole_epoch $epoch --test_freq 10 --nonlin wire --lr 1e-3 --batch_size $batch_size
        
#         python asem_json.py  "/data/result/${test_day}" "result_json/${test_day}"

#     done
# done



# python asem_json.py  "/data/result/${test_day}" "result_json/${test_day}"


# test_day="240808_relu_skip2_d8w256_"
# depth="8"
# width="256"
# epoch="300"

# datasets=( "knights" "bracelet" "bunny" "tarot")  
# batch_sizes=("4092" "16384" )

# result_path="/data/result/${test_day}"


# for batch_size in "${batch_sizes[@]}"; do
#     for dataset in "${datasets[@]}"; do
        
#         echo "Processing $dataset"
        
#         python wire_lf_new.py --data_dir "${stanford_path}/${dataset}" --exp_dir "${result_path}/${dataset}_batchsize_${batch_size}" --depth $depth --width $width --whole_epoch $epoch --test_freq 10 --nonlin relu_skip2 --lr 4e-4 --batch_size $batch_size
        
#         python asem_json.py  "/data/result/${test_day}" "result_json/${test_day}"

#     done
# done


