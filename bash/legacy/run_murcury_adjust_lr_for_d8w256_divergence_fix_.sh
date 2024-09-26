stanford_path="/data/hmjung/NeuLF_rgb/dataset/stanford_half"
# llff_path="/data/nerf_llff_data"
# bmw_path="/data/bmw_dataset"


test_day="240721_new_wire_d8w256_adjust_lr_1e-3"
depth="2"
width="256"
epoch="301"


#datasets=("beans" "gem" "bunny" "bracelet")  
#datasets=("beans")  

datasets=( "knights" "gem" "bunny" "bracelet" "chess" "flowers" "tarot")  
result_path="/data/result/${test_day}"
#datasets=("bunny")  


for dataset in "${datasets[@]}"; do
    echo "Processing $dataset"
    
    python wire_lf_new.py --data_dir "${stanford_path}/${dataset}" --exp_dir "${result_path}/${test_day}_${dataset}" --depth $depth --width $width --whole_epoch $epoch --test_freq 10 --nonlin wire --lr 1e-3 --gpu 1
    
    python asem_json.py  "/data/result/${test_day}" "result_json/${test_day}"

done
python asem_json.py  "/data/result/${test_day}" "result_json/${test_day}"

# python asem_json.py  "/data/result/${test_day}" "result_json/${test_day}"

# test_day="240719_new_wire_d8w256_adjust_lr_5e-4"

# datasets=( "knights" "gem" "bunny" "bracelet" "chess" "flowers" "tarot")  
# result_path="/data/result/${test_day}"


# for dataset in "${datasets[@]}"; do
#     echo "Processing $dataset"
    
#     python wire_lf_new.py --data_dir "${stanford_path}/${dataset}" --exp_dir "${result_path}/${test_day}_${dataset}" --depth $depth --width $width --whole_epoch $epoch --test_freq 1 --nonlin wire --lr 5e-4 --gpu 1
    
#     python asem_json.py  "/data/result/${test_day}" "result_json/${test_day}"

# done

# python asem_json.py  "/data/result/${test_day}" "result_json/${test_day}"

