#!/usr/bin/env bash

declare -A map=( [1]=854 [2]=825 [3]=796 [4]=767 [5]=736 [6]=707 [7]=677 [8]=647 [9]=617 [10]=587 [11]=557 [12]=527 [13]=497 )

for r in {1..100};
do

for i in {0..5};
do

for s in {1..13};
do

echo $r $i $s

# echo ${map[${s}]}
# echo run${r}/alpha/iter${i}/CIalpha_stack${s}.ascii_n${map[${s}]}
mv run${r}/alpha/iter${i}/CIalpha_stack${s}.ascii run${r}/alpha/iter${i}/CIalpha_only_n${map[${s}]}.ascii

done

done

i=6

for s in {1..13};
do

echo $r $i $s

# echo ${map[${s}]}
# echo run${r}/alpha/iter${i}/CIalpha_stack${s}.ascii_n${map[${s}]}
mv run${r}/alpha/iter${i}/CIalpha_only_stack${s}.ascii run${r}/alpha/iter${i}/CIalpha_only_n${map[${s}]}.ascii

done

done
