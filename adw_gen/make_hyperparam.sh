alpha=${1}
beta=${2}
seed=${3}
jsonfile=hyperparams/${alpha}_${beta}_${seed}.json
rm ${jsonfile}
touch ${jsonfile}
echo "{" >> ${jsonfile} 
echo "  \"alpha\": ${alpha}," >> ${jsonfile}
echo "  \"beta\": ${beta}," >> ${jsonfile}
echo "  \"result_path\": \"results/${alpha}_${beta}_${seed}\"" >> ${jsonfile}
echo "}" >> ${jsonfile}
