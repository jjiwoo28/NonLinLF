stanford_path="/data/NeuLF_rgb/stanford_half"
# llff_path="/data/nerf_llff_data"
# bmw_path="/data/bmw_dataset"


test_day="240802_debug_44"
depth="2"
width="256"
epoch="5"


#datasets=("beans" "gem" "bunny" "bracelet")  
#datasets=("beans")  

#datasets=( "knights" "gem" "bunny" "bracelet" "chess" "flowers" "tarot")  
result_path="/data/result/${test_day}"
#datasets=("bunny")  
datasets=( "knights" "gem"  "bracelet" "chess" "flowers" "tarot")  
datasets=("knights")  


for dataset in "${datasets[@]}"; do
    echo "Processing $dataset"
    
    python wire_lf_new.py --data_dir "${stanford_path}/${dataset}" --exp_dir "${result_path}/${test_day}_${dataset}" --depth $depth --width $width --whole_epoch $epoch --test_freq 1 --nonlin relu --lr 1e-3 --benchmark --real_gabor --batch_size 262144 
    
    python asem_json.py  "/data/result/${test_day}" "result_json/${test_day}"

done

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

