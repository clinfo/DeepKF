rm alpha_beta.md
touch alpha_beta.md
echo '| alpha |beta|ex0|ex1|ex2|' >> alpha_beta.md
echo '|      -|   -| -| -| -| -| -| -| -| -|  -|' >> alpha_beta.md
#|    .1 |   .1| ![](results/result_1_1/ex0/plot/0_data0_infer.png) |![](results/result_1_1/ex1/plot/0_data0_infer.png) |![](results/result_1_1/ex2/plot/0_data0_infer.png) |
for i in `seq 1 10`
do
    alpha=`echo ${i}| awk '{print $1*0.1}'`
    for j in `seq 1 10`
    do
        beta=`echo ${j}| awk '{print $1*0.1}'`
        echo \|${alpha}\|${beta}\|![]\(results/result_${i}_${j}/ex0/plot/0_data0_infer.png\) \|![]\(results/result_${i}_${j}/ex1/plot/0_data0_infer.png\) \|![]\(results/result_${i}_${j}/ex2/plot/0_data0_infer.png\) \| >> alpha_beta.md
    done
done