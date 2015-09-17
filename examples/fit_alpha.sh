#!/usr/bin/env bash

for i in {1..100};
do

if [ ! -d run${i}/alpha/ ]
then
mkdir run${i}/alpha/
fi

cp fit_stacks_1c.py run${i}/alpha/
cd run${i}/alpha/

count=`ls -1 *.ascii 2>/dev/null | wc -l`

if [ ${count} != 0 ]
then

echo "Will fit in run"${i}
python fit_stacks_1c.py
cp CIalpha_-47kms_nomod_1c.log ../../all/iter7/CIalpha_-47kms_run${i}.log

fi

cd ../..

done