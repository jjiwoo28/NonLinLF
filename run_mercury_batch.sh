stanford_path="/data/hmjung/NeuLF_rgb/dataset/stanford_half"
# llff_path="/data/nerf_llff_data"
# bmw_path="/data/bmw_dataset"


test_day="240802_relu_skips_d8w256_lr4e-4_exp_sc_batch8192"
depth="8"
width="256"
epoch="300"


#datasets=("beans" "gem" "bunny" "bracelet")  
#datasets=("beans")  

#datasets=( "knights" "gem" "bunny" "bracelet" "chess" "flowers" "tarot")  
result_path="/data/result/${test_day}"
#datasets=("bunny")  
datasets=( "knights" "bracelet" "bunny" "tarot")  


for dataset in "${datasets[@]}"; do
    echo "Processing $dataset"
    
    python wire_lf_new.py --data_dir "${stanford_path}/${dataset}" --exp_dir "${result_path}/${test_day}_${dataset}" --depth $depth --width $width --whole_epoch $epoch --test_freq 10 --nonlin relu_skip --lr 4e-4 --batch_size 8192
    
    python asem_json.py  "/data/result/${test_day}" "result_json/${test_day}"

done

python asem_json.py  "/data/result/${test_day}" "result_json/${test_day}"

test_day="240802_relu_skips_d4w256_lr4e-4_exp_sc_batch8192"
depth="4"
width="256"
epoch="300"

datasets=( "knights" "bracelet" "bunny" "tarot")  

result_path="/data/result/${test_day}"



for dataset in "${datasets[@]}"; do
    echo "Processing $dataset"
    
    python  wire_lf_new.py --data_dir "${stanford_path}/${dataset}" --exp_dir "${result_path}/${test_day}_${dataset}" --depth $depth --width $width --whole_epoch $epoch --test_freq 10 --nonlin relu_skip --lr 4e-4 --batch_size 8192
    
    python asem_json.py  "/data/result/${test_day}" "result_json/${test_day}"

done

python asem_json.py  "/data/result/${test_day}" "result_json/${test_day}"


test_day="240802_relu_skips_d2w256_lr4e-4_exp_sc_batch8192"
depth="2"
width="256"
epoch="300"

datasets=( "knights" "bracelet" "bunny" "tarot")  

result_path="/data/result/${test_day}"



for dataset in "${datasets[@]}"; do
    echo "Processing $dataset"
    
    python  wire_lf_new.py --data_dir "${stanford_path}/${dataset}" --exp_dir "${result_path}/${test_day}_${dataset}" --depth $depth --width $width --whole_epoch $epoch --test_freq 10 --nonlin relu_skip --lr 4e-4 --batch_size 8192
    
    python asem_json.py  "/data/result/${test_day}" "result_json/${test_day}"

done

python asem_json.py  "/data/result/${test_day}" "result_json/${test_day}"


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

