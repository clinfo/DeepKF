for alpha in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    for beta in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
    do
        hyparam=hyparam_adw_gen_${alpha}_${beta}.json
        rm ${hyparam}
        touch ${hyparam}
        echo "{" >> ${hyparam} 
        echo "\t \"alpha\": ${alpha}," >> ${hyparam}
        echo "\t \"beta\": ${beta}," >> ${hyparam}
        echo "\t \"result_path\": \"result_${alpha}_${beta}\"" >> ${hyparam}
        echo "}" >> ${hyparam}
    done
done