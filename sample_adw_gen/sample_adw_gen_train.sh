for i in `seq 1 10`
do
alpha=`echo ${i}| awk '{print $1*0.1}'`
    for j in `seq 1 10`
    do
        beta=`echo ${j}| awk '{print $1*0.1}'`
        hyparam=hyparam_adw_gen_${i}_${j}.json
        rm ${hyparam}
        touch ${hyparam}
        echo "{" >> ${hyparam} 
        echo "\t \"alpha\": ${alpha}," >> ${hyparam}
        echo "\t \"beta\": ${beta}," >> ${hyparam}
        echo "\t \"result_path\": \"result_${i}_${j}\"" >> ${hyparam}
        echo "}" >> ${hyparam}
        dmm train --config config_adw_gen.json\
         --hyperparam hyparam_adw_gen_${i}_${j}.json\
         --save-config config_adw_gen_${i}_${j}.result.json --gpu 0
        dmm-plot train --config config_adw_gen_${i}_${j}.result.json --limit_all 5
    done
done