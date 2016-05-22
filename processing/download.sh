#!/bin/bash

if [ $# -lt 2 ]; then
        echo "Usage: $0 seq_start seq_end [wget_args]"
        exit
fi

url_format='https://loanperformancedata.fanniemae.com/lppub/publish?file='
seq_start=$1
seq_end=$2
number=0
for year in `seq $seq_start $seq_end`;do
    for quarter in `seq 1 4`;do
	filenames[$number]="${year}Q${quarter}.zip";
	number=$(($number + 1));
	#url = "$url_format${year}Q$quarter"
	#echo "$url_format_y${year}Q${quarter}.zip"
    done
done

for filename in ${filenames[*]};do
    echo "$url_format$filename"
    wget --load-cookies=cookies.txt --output-document=$filename "$url_format$filename" 
done



#shift 3

#printf "$url_format\\n" `seq $seq_start $seq_end` `seq 1 4`  #| wget -i- "$@"
