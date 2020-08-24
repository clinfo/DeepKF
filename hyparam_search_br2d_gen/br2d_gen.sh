#k=$1
#for i in `seq 1 10`
#do
#    for j in `seq 1 10`
#    do
#        dmm train,test --config config_adw_gen.json\
#        --hyperparam hyparams/hyparam_adw_gen_${i}_${j}_${k}.json\
#        --save-config config_results/config_adw_gen_${i}_${j}_${k}.result.json --gpu ${k}
        dmm train,test --config config_br2d_gen.json\
        --hyperparam hyparams/hyparam_br2d_gen_1_1_0.json\
        --save-config config_results/config_br2d_gen_1_1_0.result.json --gpu 0
#        dmm-plot infer --config config_results/config_adw_gen_${i}_${j}_${k}.result.json --limit_all 5            
#    done
#done
