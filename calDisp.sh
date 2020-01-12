#!/bin/bash 

# for (( i = 0; i < 10; i++ )); do
# 	python3 main.py --input-left ./data/Synthetic/TL${i}.png --input-right ./data/Synthetic/TR${i}.png --output ./data/Synthetic/TL${i}.pfm
# done

Error=$(echo 0+0 | bc)
for (( i = 0; i < 10; i++ )); do
	err=$(python3 countErr.py --GT ./data/Synthetic/TLD${i}.pfm --input ./data/Synthetic/TL${i}.pfm)
	# Error=$(bc -l <<<"${Error}+${err}")
	Error=`expr $Error+$err`
	echo $err
	Error=$(echo "`bc <<< $Error`")
done

Error=$(echo "$Error / 10" | bc -l)
echo "Average error: 0$Error"
