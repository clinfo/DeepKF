k=$1
for i in `seq 1 10`
do
    for j in `seq 1 10`
    do
        dmm train,test --config config_adw_gen.json\
         --hyperparam hyparams/hyparam_adw_gen_${i}_${j}_${k}.json\
         --save-config config_results/config_adw_gen_${i}_${j}_${k}.result.json\
         --seed 0
         --gpu 0
        dmm train,test --config config_adw_gen.json\
         --hyperparam hyparams/hyparam_adw_gen_${i}_${j}_${k}.json\
         --save-config config_results/config_adw_gen_${i}_${j}_${k}.result.json\
         --seed 1
         --gpu 1
        dmm train,test --config config_adw_gen.json\
         --hyperparam hyparams/hyparam_adw_gen_${i}_${j}_${k}.json\
         --save-config config_results/config_adw_gen_${i}_${j}_${k}.result.json\
         --seed 0
         --gpu 0
    done
done
