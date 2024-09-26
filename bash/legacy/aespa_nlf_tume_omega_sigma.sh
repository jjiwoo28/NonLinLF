stanford_path="/data/NeuLF_rgb/stanford_half"

test_day="240823_tune_omega_sigma_"
result_path="/data/result/${test_day}"

width="256"
epoch="300"

#datasets=( "knights" )
datasets=( "knights" )
depths=("4")

omega_sigmas=("30" "10" "50")

nomlin=("siren" "gauss" "finer" "wire")

for depth in "${depths[@]}"; do
    for dataset in "${datasets[@]}"; do
        for nonlin in "${nomlin[@]}"; do  # 여기를 수정
            for om_sig in "${omega_sigmas[@]}"; do
                echo "Processing $dataset , $nonlin , depth : $depth , om_sig : $om_sig"
                python wire_lf_new.py \
                    --data_dir "${stanford_path}/${dataset}" \
                    --exp_dir "${result_path}/${test_day}_${nonlin}_depth${depth}_om_sig_${om_sig}_${dataset}" \
                    --depth $depth \
                    --width $width \
                    --whole_epoch $epoch \
                    --test_freq 10 \
                    --nonlin $nonlin \
                    --lr_batch_preset \
                    --gpu 1 \
                    --omega $om_sig \
                    --sigma $om_sig \

                python asem_json.py "/data/result/${test_day}" "result_json/${test_day}"
            done
        done
    done
done
