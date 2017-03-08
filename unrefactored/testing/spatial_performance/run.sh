OUTPUT_DIR=.
PARAMS_FILE=${OUTPUT_DIR}/params.txt
RESULT_FILE=${OUTPUT_DIR}/result.txt
DELIMITER=##########################################################################################################
echo $PARAMS_FILE
cat /dev/null > $RESULT_FILE
while IFS= read -r opt
do
    for((i=36;i>0;i=i-3))
	do
		export OMP_NUM_THREADS=$i
		#echo $OMP_NUM_THREADS
		for j in `seq 10`
		do
			echo ./spatial_performance $opt $j $OMP_NUM_THREADS >>\
				 $RESULT_FILE
			./spatial_performance.out $opt $j >> ${RESULT_FILE}
		done
		echo $DELIMITER >> $RESULT_FILE
		echo $DELIMITER >> $RESULT_FILE
		echo $DELIMITER >> $RESULT_FILE
		echo DONE $opt $i
	done
done <$PARAMS_FILE
