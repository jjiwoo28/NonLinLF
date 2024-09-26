stanford_path="/data/NeuLF_rgb/stanford_half"
# llff_path="/data/nerf_llff_data"
# bmw_path="/data/bmw_dataset"


test_day="240817_debug_relu_epi_2f2w256"
depth="2"
width="256"
epoch="300"

result_path="/data/result/${test_day}"
datasets=( "knights" "gem"  "bracelet" "chess" "flowers" "tarot")  
datasets=( "knights" "bracelet" "bunny" "tarot")  
#datasets=("gem")  


for dataset in "${datasets[@]}"; do
    echo "Processing $dataset"
    
    python nlf_decompose_epi.py --data_dir "${stanford_path}/${dataset}" --exp_dir "${result_path}/${test_day}_${dataset}" --depth $depth --width $width --whole_epoch $epoch --test_freq 1 --nonlin relu_decom --lr 5e-4 --benchmark --gpu 1 
    
    python asem_json.py  "/data/result/${test_day}" "result_json_debug/${test_day}"

done
# test_day="240817_decom_relu_2f2w256_lr_1e-4"5
# depth="2"
# width="256"
# epoch="300"

# result_path="/data/result/${test_day}"
# datasets=("knights")  
# datasets=( "knights" "bracelet" "bunny" "tarot")  

# for dataset in "${datasets[@]}"; do
#     echo "Processing $dataset"
    
#     python nlf_decompose.py --data_dir "${stanford_path}/${dataset}" --exp_dir "${result_path}/${test_day}_${dataset}" --depth $depth --width $width --whole_epoch $epoch --test_freq 1 --nonlin relu_decom --lr 1e-4 --benchmark  
    
#     python asem_json.py  "/data/result/${test_day}" "result_json_debug/${test_day}"

# done

#python asem_json.py  "/data/result/${test_day}" "result_json/${test_day}"

# test_day="240719_new_relu_d4w256"

# #datasets=( "knights" "gem" "bunny" "bracelet" "chess" "flowers" "tarot")  
# datasets=( "flowers" "tarot")  
# result_path="/data/result/${test_day}"


# for dataset in "${datasets[@]}"; do
#     echo "Processing $dataset"
    
#     python wire_lf_new.py --data_dir "${stanford_path}/${dataset}" --exp_dir "${result_path}/${test_day}_${dataset}" --depth $depth --width $width --whole_epoch $epoch --test_freq 5 --nonlin relu
    
#     python asem_json.py  "/data/result/${test_day}" "result_json/${test_day}"

# done

# python asem_json.py  "/data/result/${test_day}" "result_json/${test_day}"

