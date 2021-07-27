#!/bin/sh

#  Script.sh
#
#  Created by Mandy on 23.07.21.
#  


v=$(pwd)

#first move all test-data into train folder


for dir in "$v"/data/car_data/test/*
do
    for file in "$dir"*/*
    do
        #echo "$dir"
        if [[ -f $file ]]
        then

            echo "$file"
            generated_file_name=${file/test/train}

            echo "$generated_file_name"
            mv "$file" "$generated_file_name"
        fi
    done
done
  
  

# calculate the test-split and move it back to test directory

for dir in "$v"/data/car_data/train/* #~/#./car_data/car_data/train/*
do

    cd "$dir"
    n_files=$(ls  -1q  | wc -l)

    #calculate no. of test-files and round to integer
    test_files=$(bc <<< "$n_files /10.0")


#    #ls | head -$test_files
    generated_file_name=${dir/train/test}
    
    for file in $(ls | head -$test_files)
    do
        echo "$generated_file_name/$file"
        mv $file "$generated_file_name/$file"
    done
done
