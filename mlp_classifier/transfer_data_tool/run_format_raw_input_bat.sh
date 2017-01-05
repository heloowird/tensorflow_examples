#!/bin/sh

raw_file_dir="./train_data_line_text"
format_file_dir="./tf_record_for_train"

if [ -d $format_file_dir ]
then
	rm -f $format_file_dir/*
else
	mkdir $format_file_dir
fi

max_process_nums=10

i=0
file_lst=`ls $raw_file_dir`
for ele in $file_lst
do
	python format_line_to_tf_record.py $raw_file_dir/$ele $format_file_dir/$ele && echo "$ele finish" &
	((i++))
	if [ $i -eq $max_process_nums ]
	then
		wait
		i=0
	fi
done
wait
