alpha=0.1
beta=0.2
hyparam=hyparam_adw_gen_${alpha}_${beta}.json
rm ${json}
echo "{" >> ${hyparam} 
echo "\t \"alpha\": ${alpha}," >> ${hyparam}
echo "\t \"beta\": ${beta}," >> ${hyparam}
echo "\t \"result_path\": \"result_${alpha}_${beta}\"" >> ${hyparam}
echo "}" >> ${hyparam}
