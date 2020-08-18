for k in `seq 0 2`
do
    for i in `seq 1 10`
    do
        alpha=`echo ${i}| awk '{print $1*0.1}'`
        for j in `seq 1 10`
        do
            beta=`echo ${j}| awk '{print $1*0.1}'`
            hyparam=hyparams/hyparam_adw_gen_${i}_${j}_${k}.json
            rm ${hyparam}
            touch ${hyparam}
            echo "{" >> ${hyparam} 
            echo "\t \"alpha\": ${alpha}," >> ${hyparam}
            echo "\t \"beta\": ${beta}," >> ${hyparam}
            echo "\t \"result_path\": \"results/result_${i}_${j}/ex${k}\"" >> ${hyparam}
            echo "}" >> ${hyparam}
        done
    done
done