#!/bin/bash
arr=(read_comment like click_avatar forward)
for item in ${arr[*]}
do
echo "$item"
python fm.py $item
done
read -p 'over！'
