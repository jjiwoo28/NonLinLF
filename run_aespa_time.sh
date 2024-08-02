stanford_path="/data/NeuLF_rgb/stanford_half"
# llff_path="/data/nerf_llff_data"
# bmw_path="/data/bmw_dataset"


test_day="240802_debug_44"
result_path="/data/result/${test_day}"
depth="2"
width="256"
epoch="10"


#datasets=("beans" "gem" "bunny" "bracelet")  
#datasets=("beans")  

#datasets=( "knights" "gem" "bunny" "bracelet" "chess" "flowers" "tarot")  
#datasets=("bunny")  
datasets=( "knights" "gem"  "bracelet" "chess" "flowers" "tarot")  
datasets=("knights")  


#relu


depth="2"
width="256"
test_day="240802_time_relu_d2w256"
result_path="/data/result/${test_day}"


for dataset in "${datasets[@]}"; do
    echo "Processing $dataset"
    
    python wire_lf_new.py --data_dir "${stanford_path}/${dataset}" --exp_dir "${result_path}/${test_day}_${dataset}" --depth $depth --width $width --whole_epoch $epoch --test_freq 1 --nonlin relu --lr 1e-3 --benchmark --real_gabor --batch_size 262144 
    
    python asem_json.py  "/data/result/${test_day}" "result_json/${test_day}"

done

depth="4"
width="256"
test_day="240802_time_relu_d4w256"
result_path="/data/result/${test_day}"


for dataset in "${datasets[@]}"; do
    echo "Processing $dataset"
    
    python wire_lf_new.py --data_dir "${stanford_path}/${dataset}" --exp_dir "${result_path}/${test_day}_${dataset}" --depth $depth --width $width --whole_epoch $epoch --test_freq 1 --nonlin relu --lr 1e-3 --benchmark --real_gabor --batch_size 262144 
    
    python asem_json.py  "/data/result/${test_day}" "result_json/${test_day}"

done



depth="8"
width="256"
test_day="240802_time_relu_d8w256"
result_path="/data/result/${test_day}"


for dataset in "${datasets[@]}"; do
    echo "Processing $dataset"
    
    python wire_lf_new.py --data_dir "${stanford_path}/${dataset}" --exp_dir "${result_path}/${test_day}_${dataset}" --depth $depth --width $width --whole_epoch $epoch --test_freq 1 --nonlin relu --lr 1e-3 --benchmark --real_gabor --batch_size 262144 
    
    python asem_json.py  "/data/result/${test_day}" "result_json/${test_day}"

done


#wire

depth="2"
width="256"
test_day="240802_time_wire_d2w256"
result_path="/data/result/${test_day}"


for dataset in "${datasets[@]}"; do
    echo "Processing $dataset"
    
    python wire_lf_new.py --data_dir "${stanford_path}/${dataset}" --exp_dir "${result_path}/${test_day}_${dataset}" --depth $depth --width $width --whole_epoch $epoch --test_freq 1 --nonlin wire --lr 1e-3 --benchmark --real_gabor --batch_size 262144 
    
    python asem_json.py  "/data/result/${test_day}" "result_json/${test_day}"

done

depth="4"
width="256"
test_day="240802_time_wire_d4w256"
result_path="/data/result/${test_day}"


for dataset in "${datasets[@]}"; do
    echo "Processing $dataset"
    
    python wire_lf_new.py --data_dir "${stanford_path}/${dataset}" --exp_dir "${result_path}/${test_day}_${dataset}" --depth $depth --width $width --whole_epoch $epoch --test_freq 1 --nonlin wire --lr 1e-3 --benchmark --real_gabor --batch_size 262144 
    
    python asem_json.py  "/data/result/${test_day}" "result_json/${test_day}"

done



depth="8"
width="256"
test_day="240802_time_wire_d8w256"
result_path="/data/result/${test_day}"


for dataset in "${datasets[@]}"; do
    echo "Processing $dataset"
    
    python wire_lf_new.py --data_dir "${stanford_path}/${dataset}" --exp_dir "${result_path}/${test_day}_${dataset}" --depth $depth --width $width --whole_epoch $epoch --test_freq 1 --nonlin wire --lr 1e-3 --benchmark --real_gabor --batch_size 262144 
    
    python asem_json.py  "/data/result/${test_day}" "result_json/${test_day}"

done

#relu_skip2

depth="2"
width="256"
test_day="240802_time_relu_skip2_d2w256"
result_path="/data/result/${test_day}"


for dataset in "${datasets[@]}"; do
    echo "Processing $dataset"
    
    python wire_lf_new.py --data_dir "${stanford_path}/${dataset}" --exp_dir "${result_path}/${test_day}_${dataset}" --depth $depth --width $width --whole_epoch $epoch --test_freq 1 --nonlin rrelu_skip2 --lr 1e-3 --benchmark --real_gabor --batch_size 262144 
    
    python asem_json.py  "/data/result/${test_day}" "result_json/${test_day}"

done


depth="4"
width="256"
test_day="240802_time_relu_skip2_d4w256"
result_path="/data/result/${test_day}"


for dataset in "${datasets[@]}"; do
    echo "Processing $dataset"
    
    python wire_lf_new.py --data_dir "${stanford_path}/${dataset}" --exp_dir "${result_path}/${test_day}_${dataset}" --depth $depth --width $width --whole_epoch $epoch --test_freq 1 --nonlin relu_skip2 --lr 1e-3 --benchmark --real_gabor --batch_size 262144 
    
    python asem_json.py  "/data/result/${test_day}" "result_json/${test_day}"

done



depth="8"
width="256"
test_day="240802_time_relu_skip2_d8w256"
result_path="/data/result/${test_day}"


for dataset in "${datasets[@]}"; do
    echo "Processing $dataset"
    
    python wire_lf_new.py --data_dir "${stanford_path}/${dataset}" --exp_dir "${result_path}/${test_day}_${dataset}" --depth $depth --width $width --whole_epoch $epoch --test_freq 1 --nonlin relu_skip2 --lr 1e-3 --benchmark --real_gabor --batch_size 262144 
    
    python asem_json.py  "/data/result/${test_day}" "result_json/${test_day}"

done


#wire








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

