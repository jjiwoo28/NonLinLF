stanford_path="/data/NeuLF_rgb/stanford_half"
# llff_path="/data/nerf_llff_data"
# bmw_path="/data/bmw_dataset"


test_day="240816_gussian_d4w256"
depth="4"
width="256"
epoch="300"
datasets=( "knights" "bracelet" "bunny" "tarot")

#datasets=("bunny")  

batch_sizes=("65536" "8192")

result_path="/data/result/${test_day}"

for dataset in "${datasets[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do
        echo "Processing $dataset"
    
        python wire_lf_new.py --data_dir "${stanford_path}/${dataset}" --exp_dir "${result_path}/${test_day}_${dataset}" --depth $depth --width $width --whole_epoch $epoch --test_freq 10 --nonlin gauss --lr 0.005  --batch_size $batch_size
        
        python asem_json.py  "/data/result/${test_day}" "result_json/${test_day}"
    done
done



test_day="240816_siren_d4w256"
depth="4"
width="256"
epoch="300"
datasets=( "knights" "bracelet" "bunny" "tarot")

#datasets=("bunny")  

batch_sizes=("65536" "8192")

result_path="/data/result/${test_day}"

for dataset in "${datasets[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do
        echo "Processing $dataset"
    
        python wire_lf_new.py --data_dir "${stanford_path}/${dataset}" --exp_dir "${result_path}/${test_day}_${dataset}" --depth $depth --width $width --whole_epoch $epoch --test_freq 10 --nonlin siren --lr 0.0005  --batch_size $batch_size
        
        python asem_json.py  "/data/result/${test_day}" "result_json/${test_day}"
    done
done



python asem_json.py  "/data/result/${test_day}" "result_json/${test_day}"

