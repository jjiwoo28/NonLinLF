stanford_path="/data/NeuLF_rgb/stanford_half"
# llff_path="/data/nerf_llff_data"
# bmw_path="/data/bmw_dataset"


test_day="240724_new_wire_d4w256_wire_tunable"
depth="2"
width="256"
epoch="300"


#datasets=("beans" "gem" "bunny" "bracelet")  
#datasets=("beans")  

#datasets=( "knights" "gem" "bunny" "bracelet" "chess" "flowers" "tarot")  
result_path="/data/result/${test_day}"
#datasets=("bunny")  
datasets=( "knights" "gem"  "bracelet" "chess" "flowers" "tarot")  


for dataset in "${datasets[@]}"; do
    echo "Processing $dataset"
    
    python wire_lf_new.py --data_dir "${stanford_path}/${dataset}" --exp_dir "${result_path}/${test_day}_${dataset}" --depth $depth --width $width --whole_epoch $epoch --test_freq 1 --nonlin wire --wire_tunable --gpu 1
    
    python asem_json.py  "/data/result/${test_day}" "result_json/${test_day}"

done

# python asem_json.py  "/data/result/${test_day}" "result_json/${test_day}"

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

