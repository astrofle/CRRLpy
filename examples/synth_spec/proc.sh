#!/usr/bin/env bash

# Change according to where the scripts are located
SPECPATH=/data2/psalas/python/CRRLpy/scripts

# Create subdirectory structure
mkdir logs
mkdir medges
mkdir stacks
mkdir models
mkdir plots

function make_dirs_iter()
{
i=$1
mkdir -p lines/iter${i}
mkdir lines/iter${i}/sbs
mkdir lines/iter${i}/malpha
mkdir lines/iter${i}/mbeta
mkdir lines/iter${i}/mgamma
mkdir lines/iter${i}/mdelta
mkdir lines/iter${i}/bcorr
mkdir plots/iter${i}
mkdir stacks/iter${i}
}

function process_transition() 
{

trans=$1
iter=$2
rmslog=$3

if [ $trans == 'alpha' ]
then
transs=RRL_CIalpha,RRL_CIbeta,RRL_CIgamma,RRL_CIdelta
stackmax=18
folder0=lines/iter${iter}/sbs
fi

if [ $trans == 'beta' ]
then
transs=RRL_CIbeta,RRL_CIgamma,RRL_CIdelta
stackmax=15
folder0=lines/iter${iter}/malpha
fi

if [ $trans == 'gamma' ]
then
transs=RRL_CIgamma,RRL_CIdelta
stackmax=9
folder0=lines/iter${iter}/mbeta
fi

if [ $trans == 'delta' ]
then
transs=RRL_CIdelta
stackmax=6
folder0=lines/iter${iter}/mgamma
fi

# Convert the lines to velocity
${SPECPATH}/spec2vel.py ${folder0}'/lba_sim_SB*.ascii' 'lines/iter'${iter}'/lba_sim' -t RRL_CI${trans} --z=-1.567e-4 --f_col=0
# Determine which lines are more than 50 km/s from other CI line
${SPECPATH}/goodlines.py ${folder0}'/lba_sim_SB*.ascii' 'lines/iter'${iter}'/lba_sim' -t ${transs} --x_col=0 --z=-1.567e-4 -s 10
mv *_good_lines.log logs
# sleep 1
# Make a list with the lines that go into each stack and which subband gets which stack removed
${SPECPATH}/makestacklist.py logs/RRL_CI${trans}_good_lines.log ${rmslog} logs/RRL_CI${trans} ${stackmax} -p Last --path=${folder0} --first=SB000
# Make the stacks and remove them from the spectra
for (( i=1; i<=$stackmax; i++ ))
do
echo $i
${SPECPATH}/matchweightlist.py ${rmslog} 'logs/RRL_CI'${trans}'_stack'${i}'.log' 'logs/RRL_CI'${trans}'_stack'${i}'_w.log'
${SPECPATH}/stack.py 'logs/RRL_CI'${trans}'_stack'${i}'.log' 'stacks/iter'${iter}'/RRL_CI'${trans}'_stack'${i}'.ascii' -m interpol --weight='list' \
                      --weight_list='logs/RRL_CI'${trans}'_stack'${i}'_w.log' --x_col=1 --y_col=2 --v_max=800 --v_min=-800
${SPECPATH}/sbsplot.py 'stacks/iter'${iter}'/RRL_CI'${trans}'_stack'${i}'.ascii' -x 'Velocity (km s$^{-1}$)' plots/iter${iter}/stack_RRL_CI${trans}${i}.pdf
python makemodel.py 'stacks/iter'${iter}'/RRL_CI'${trans}'_stack'${i}'.ascii' 'models/RRL_CI'${trans}'_stack'${i}'.ascii' 'plots/iter'${iter}/'RRL_CI'${trans}'_stack'${i}'.pdf'
${SPECPATH}/sbsplot.py 'models/RRL_CI'${trans}'_stack'${i}'.ascii' -x 'Velocity (km s$^{-1}$)' plots/iter${iter}/models_RRL_CI${trans}${i}.pdf
${SPECPATH}/removemodel.py 'logs/RRL_CI'${trans}'_stack'${i}'_SBs.log' 'models/RRL_CI'${trans}'_stack'${i}'.ascii' 'lines/iter'${iter}'/m'${trans}'/lba_sim' -t RRL_CI${trans} \
                            --x_col=0 --y_col=1 --z=-1.567e-4 -p --plot_file=plots/iter${iter}/lba_sim_RRL_CI${trans}_mstack${i}.pdf -m 'subtract'
done
}

# Plot the raw spectra showing the CI lines at the velocity of -47 km/s
${SPECPATH}/sbsplot.py 'raw/lba_sim_SB*.ascii' plots/raw_sbs.pdf --x_col=0 --y_col=1 -l --z=-1.567e-4 -t RRL_CIalpha,RRL_CIbeta,RRL_CIgamma,RRL_CIdelta
cp mcont/lba_sim_SB*.ascii medges/.
# Get the rms of each subband
${SPECPATH}/makeweightlist.py 'medges/lba_sim_SB*.ascii' logs/medges_sbs_rms.log --f_col=0 --y_col=1 -t RRL_CIalpha,RRL_CIbeta,RRL_CIgamma,RRL_CIdelta --z=-1.567e-4 -m 1/rms2 -d 50

make_dirs_iter 0
cp medges/* lines/iter0/sbs
# Move the spectra to the starting folder
process_transition alpha 0 logs/medges_sbs_rms.log
process_transition beta 0 logs/medges_sbs_rms.log
process_transition gamma 0 logs/medges_sbs_rms.log
process_transition delta 0 logs/medges_sbs_rms.log

# Remove the baseline
${SPECPATH}/baselinecorr.py 'lines/iter0/mdelta/lba_sim_SB*.ascii' 'lines/iter0/bcorr/lba_sim' -k 1 --x_col=0 --y_col=1 -s -b 'models/lba_sim_b0'
${SPECPATH}/sbsplot.py 'lines/iter0/bcorr/lba_sim_SB*.ascii' plots/bcorr0_sbs.pdf --x_col=0 --y_col=1 -l --z=-1.567e-4 -t RRL_CIalpha,RRL_CIbeta,RRL_CIgamma,RRL_CIdelta

iter=1
piter=`expr $iter - 1`
make_dirs_iter ${iter}
for i in {000..359}
do
echo $i
if [ -f lines/iter${piter}/sbs/lba_sim_SB${i}.ascii ];
then
${SPECPATH}/removemodel.py lines/iter${piter}/sbs/lba_sim_SB${i}.ascii models/lba_sim_b${piter}_SB${i}.ascii lines/iter${iter}/sbs/lba_sim --x_col=0 --y_col=1 --freq \
                           -p --plot_file=plots/iter${iter}/lba_sim_SB${i}_mbaseline${piter}.pdf -m 'divide'
fi
done
${SPECPATH}/sbsplot.py 'lines/iter'${iter}'/sbs/lba_sim_SB*.ascii' plots/iter${iter}_sbs.pdf --x_col=0 --y_col=1 -l --z=-1.567e-4 -t RRL_CIalpha,RRL_CIbeta,RRL_CIgamma,RRL_CIdelta
${SPECPATH}/makeweightlist.py 'lines/iter'${iter}'/sbs/lba_sim_SB*.ascii' logs/iter${iter}_sbs_rms.log --f_col=0 --y_col=1 -t RRL_CIalpha,RRL_CIbeta,RRL_CIgamma,RRL_CIdelta --z=-1.567e-4 -m 1/rms2 -d 50
process_transition alpha ${iter} logs/iter${iter}_sbs_rms.log
process_transition beta  ${iter} logs/iter${iter}_sbs_rms.log
process_transition gamma ${iter} logs/iter${iter}_sbs_rms.log
process_transition delta ${iter} logs/iter${iter}_sbs_rms.log
${SPECPATH}/baselinecorr.py 'lines/iter'${iter}'/mdelta/lba_sim_SB*.ascii' 'lines/iter'${iter}'/bcorr/lba_sim' -k 2 -m --x_col=0 --y_col=1 -s -b 'models/lba_sim_b'${iter}
${SPECPATH}/sbsplot.py 'lines/iter'${iter}'/bcorr/lba_sim_SB*.ascii' plots/bcorr${iter}_sbs.pdf --x_col=0 --y_col=1 -l --z=-1.567e-4 -t RRL_CIalpha,RRL_CIbeta,RRL_CIgamma,RRL_CIdelta

iter=2
piter=`expr $iter - 1`
make_dirs_iter ${iter}
for i in {000..359}
do
echo $i
if [ -f lines/iter${piter}/sbs/lba_sim_SB${i}.ascii ];
then
${SPECPATH}/removemodel.py lines/iter${piter}/sbs/lba_sim_SB${i}.ascii models/lba_sim_b${piter}_SB${i}.ascii lines/iter${iter}/sbs/lba_sim --x_col=0 --y_col=1 --freq \
                           -p --plot_file=plots/iter${iter}/lba_sim_SB${i}_mbaseline${piter}.pdf -m 'divide'
fi
done
${SPECPATH}/sbsplot.py 'lines/iter'${iter}'/sbs/lba_sim_SB*.ascii' plots/iter${iter}_sbs.pdf --x_col=0 --y_col=1 -l --z=-1.567e-4 -t RRL_CIalpha,RRL_CIbeta,RRL_CIgamma,RRL_CIdelta
${SPECPATH}/makeweightlist.py 'lines/iter'${iter}'/sbs/lba_sim_SB*.ascii' logs/iter${iter}_sbs_rms.log --f_col=0 --y_col=1 -t RRL_CIalpha,RRL_CIbeta,RRL_CIgamma,RRL_CIdelta --z=-1.567e-4 -m 1/rms2 -d 50
process_transition alpha ${iter} logs/iter${iter}_sbs_rms.log
process_transition beta  ${iter} logs/iter${iter}_sbs_rms.log
process_transition gamma ${iter} logs/iter${iter}_sbs_rms.log
process_transition delta ${iter} logs/iter${iter}_sbs_rms.log
${SPECPATH}/baselinecorr.py 'lines/iter'${iter}'/mdelta/lba_sim_SB*.ascii' 'lines/iter'${iter}'/bcorr/lba_sim' -k 3 -m --x_col=0 --y_col=1 -s -b 'models/lba_sim_b'${iter}
${SPECPATH}/sbsplot.py 'lines/iter'${iter}'/bcorr/lba_sim_SB*.ascii' plots/bcorr${iter}_sbs.pdf --x_col=0 --y_col=1 -l --z=-1.567e-4 -t RRL_CIalpha,RRL_CIbeta,RRL_CIgamma,RRL_CIdelta

iter=3
piter=`expr $iter - 1`
make_dirs_iter ${iter}
for i in {000..359}
do
echo $i
if [ -f lines/iter${piter}/sbs/lba_sim_SB${i}.ascii ];
then
${SPECPATH}/removemodel.py lines/iter${piter}/sbs/lba_sim_SB${i}.ascii models/lba_sim_b${piter}_SB${i}.ascii lines/iter${iter}/sbs/lba_sim --x_col=0 --y_col=1 --freq \
                           -p --plot_file=plots/iter${iter}/lba_sim_SB${i}_mbaseline${piter}.pdf -m 'divide'
fi
done
${SPECPATH}/sbsplot.py 'lines/iter'${iter}'/sbs/lba_sim_SB*.ascii' plots/iter${iter}_sbs.pdf --x_col=0 --y_col=1 -l --z=-1.567e-4 -t RRL_CIalpha,RRL_CIbeta,RRL_CIgamma,RRL_CIdelta
${SPECPATH}/makeweightlist.py 'lines/iter'${iter}'/sbs/lba_sim_SB*.ascii' logs/iter${iter}_sbs_rms.log --f_col=0 --y_col=1 -t RRL_CIalpha,RRL_CIbeta,RRL_CIgamma,RRL_CIdelta --z=-1.567e-4 -m 1/rms2 -d 50
process_transition alpha ${iter} logs/iter${iter}_sbs_rms.log
process_transition beta  ${iter} logs/iter${iter}_sbs_rms.log
process_transition gamma ${iter} logs/iter${iter}_sbs_rms.log
process_transition delta ${iter} logs/iter${iter}_sbs_rms.log
${SPECPATH}/baselinecorr.py 'lines/iter'${iter}'/mdelta/lba_sim_SB*.ascii' 'lines/iter'${iter}'/bcorr/lba_sim' -k 4 -m --x_col=0 --y_col=1 -s -b 'models/lba_sim_b'${iter}
${SPECPATH}/sbsplot.py 'lines/iter'${iter}'/bcorr/lba_sim_SB*.ascii' plots/bcorr${iter}_sbs.pdf --x_col=0 --y_col=1 -l --z=-1.567e-4 -t RRL_CIalpha,RRL_CIbeta,RRL_CIgamma,RRL_CIdelta

iter=4
piter=`expr $iter - 1`
make_dirs_iter ${iter}
for i in {000..359}
do
echo $i
if [ -f lines/iter${piter}/sbs/lba_sim_SB${i}.ascii ];
then
${SPECPATH}/removemodel.py lines/iter${piter}/sbs/lba_sim_SB${i}.ascii models/lba_sim_b${piter}_SB${i}.ascii lines/iter${iter}/sbs/lba_sim --x_col=0 --y_col=1 --freq \
                           -p --plot_file=plots/iter${iter}/lba_sim_SB${i}_mbaseline${piter}.pdf -m 'divide'
fi
done
${SPECPATH}/sbsplot.py 'lines/iter'${iter}'/sbs/lba_sim_SB*.ascii' plots/iter${iter}_sbs.pdf --x_col=0 --y_col=1 -l --z=-1.567e-4 -t RRL_CIalpha,RRL_CIbeta,RRL_CIgamma,RRL_CIdelta
${SPECPATH}/makeweightlist.py 'lines/iter'${iter}'/sbs/lba_sim_SB*.ascii' logs/iter${iter}_sbs_rms.log --f_col=0 --y_col=1 -t RRL_CIalpha,RRL_CIbeta,RRL_CIgamma,RRL_CIdelta --z=-1.567e-4 -m 1/rms2 -d 50
process_transition alpha ${iter} logs/iter${iter}_sbs_rms.log
process_transition beta  ${iter} logs/iter${iter}_sbs_rms.log
process_transition gamma ${iter} logs/iter${iter}_sbs_rms.log
process_transition delta ${iter} logs/iter${iter}_sbs_rms.log
${SPECPATH}/baselinecorr.py 'lines/iter'${iter}'/mdelta/lba_sim_SB*.ascii' 'lines/iter'${iter}'/bcorr/lba_sim' -k 5 -m --x_col=0 --y_col=1 -s -b 'models/lba_sim_b'${iter}
${SPECPATH}/sbsplot.py 'lines/iter'${iter}'/bcorr/lba_sim_SB*.ascii' plots/bcorr${iter}_sbs.pdf --x_col=0 --y_col=1 -l --z=-1.567e-4 -t RRL_CIalpha,RRL_CIbeta,RRL_CIgamma,RRL_CIdelta

iter=5
piter=`expr $iter - 1`
make_dirs_iter ${iter}
for i in {000..359}
do
echo $i
if [ -f lines/iter${piter}/sbs/lba_sim_SB${i}.ascii ];
then
${SPECPATH}/removemodel.py lines/iter${piter}/sbs/lba_sim_SB${i}.ascii models/lba_sim_b${piter}_SB${i}.ascii lines/iter${iter}/sbs/lba_sim --x_col=0 --y_col=1 --freq \
                           -p --plot_file=plots/iter${iter}/lba_sim_SB${i}_mbaseline${piter}.pdf -m 'divide'
fi
done
${SPECPATH}/sbsplot.py 'lines/iter'${iter}'/sbs/lba_sim_SB*.ascii' plots/iter${iter}_sbs.pdf --x_col=0 --y_col=1 -l --z=-1.567e-4 -t RRL_CIalpha,RRL_CIbeta,RRL_CIgamma,RRL_CIdelta
${SPECPATH}/makeweightlist.py 'lines/iter'${iter}'/sbs/lba_sim_SB*.ascii' logs/iter${iter}_sbs_rms.log --f_col=0 --y_col=1 -t RRL_CIalpha,RRL_CIbeta,RRL_CIgamma,RRL_CIdelta --z=-1.567e-4 -m 1/rms2 -d 50
process_transition alpha ${iter} logs/iter${iter}_sbs_rms.log
process_transition beta  ${iter} logs/iter${iter}_sbs_rms.log
process_transition gamma ${iter} logs/iter${iter}_sbs_rms.log
process_transition delta ${iter} logs/iter${iter}_sbs_rms.log
${SPECPATH}/baselinecorr.py 'lines/iter'${iter}'/mdelta/lba_sim_SB*.ascii' 'lines/iter'${iter}'/bcorr/lba_sim' -k 1 --x_col=0 --y_col=1 -s -b 'models/lba_sim_b'${iter}
${SPECPATH}/sbsplot.py 'lines/iter'${iter}'/bcorr/lba_sim_SB*.ascii' plots/bcorr${iter}_sbs.pdf --x_col=0 --y_col=1 -l --z=-1.567e-4 -t RRL_CIalpha,RRL_CIbeta,RRL_CIgamma,RRL_CIdelta

# Make the final stacks
declare -A transs=( ["alpha"]=1 ["beta"]=1 ["gamma"]=1 ["delta"]=1 )
iter=6
piter=`expr $iter - 1`
make_dirs_iter ${iter}
for i in {000..359}
do
echo $i
if [ -f lines/iter${piter}/sbs/lba_sim_SB${i}.ascii ];
then
${SPECPATH}/removemodel.py lines/iter${piter}/sbs/lba_sim_SB${i}.ascii models/lba_sim_b${piter}_SB${i}.ascii lines/iter${iter}/sbs/lba_sim --x_col=0 --y_col=1 --freq \
                           -p --plot_file=plots/iter${iter}/lba_sim_SB${i}_mbaseline${piter}.pdf -m 'divide'
fi
done

# Leave only one transition per SB
# alpha
trans=alpha
rmslog=logs/iter${iter}_sbs_rms.log
mv lines/iter${iter}/malpha lines/iter${iter}/alpha
mkdir lines/iter${iter}/alpha/sbs
cp lines/iter${iter}/sbs/lba_sim_SB*.ascii lines/iter${iter}/alpha/sbs
${SPECPATH}/makeweightlist.py 'lines/iter'${iter}'/'${trans}'/sbs/lba_sim_SB*.ascii' ${rmslog} --f_col=0 --y_col=1 -t RRL_CIalpha,RRL_CIbeta,RRL_CIgamma,RRL_CIdelta --z=-1.567e-4 -m 1/rms2 -d 50
${SPECPATH}/makestacklist.py logs/RRL_CIbeta_good_lines.log  ${rmslog} logs/RRL_CIbeta  ${transs[alpha]} -p Last --path=lines/iter${iter}/alpha/sbs --first=SB000
${SPECPATH}/makestacklist.py logs/RRL_CIgamma_good_lines.log ${rmslog} logs/RRL_CIgamma ${transs[alpha]} -p Last --path=lines/iter${iter}/alpha/sbs --first=SB000
${SPECPATH}/makestacklist.py logs/RRL_CIdelta_good_lines.log ${rmslog} logs/RRL_CIdelta ${transs[alpha]} -p Last --path=lines/iter${iter}/alpha/sbs --first=SB000
for t in beta gamma delta
do
echo $t
stackmax=${transs[${t}]}
for (( i=1; i<=$stackmax; i++ ))
do
echo $i
${SPECPATH}/removemodel.py 'logs/RRL_CI'${t}'_stack'${i}'_SBs.log' 'models/RRL_CI'${t}'_stack'${i}'.ascii' 'lines/iter'${iter}'/'${trans}'/sbs/lba_sim' -t RRL_CI${t} \
                            --x_col=0 --y_col=1 --z=-1.567e-4 -p --plot_file=plots/iter${iter}/lba_sim_mstack${i}_RRL_CI${t}.pdf -m 'subtract'
done
done
${SPECPATH}/sbsplot.py 'lines/iter'${iter}'/'${trans}'/sbs/lba_sim_SB*.ascii' plots/only_RRL_CI${trans}_sbs.pdf --x_col=0 --y_col=1 -l --z=-1.567e-4 -t RRL_CIalpha,RRL_CIbeta,RRL_CIgamma,RRL_CIdelta
${SPECPATH}/spec2vel.py 'lines/iter'${iter}'/'${trans}'/sbs/lba_sim_SB*.ascii' 'lines/iter'${iter}'/'${trans}'/lba_sim' -t RRL_CI${trans} --z=-1.567e-4 --f_col=0
${SPECPATH}/goodlines.py 'lines/iter'${iter}'/'${trans}'/sbs/lba_sim_SB*.ascii' 'lines/iter'${iter}'/'${trans}'/lba_sim' -t RRL_CI${trans} --x_col=0 --z=-1.567e-4 -s 1
mv RRL_CI${trans}_good_lines.log logs/RRL_CI${trans}_only_good_lines.log
smax=${transs[${trans}]}
${SPECPATH}/makestacklist.py logs/RRL_CI${trans}_only_good_lines.log ${rmslog} logs/RRL_CI${trans}_only ${smax} -p Last --path=lines/iter${iter}/${trans}/sbs --first=SB000
for (( i=1; i<=$smax; i++ ))
do
echo $i
${SPECPATH}/matchweightlist.py ${rmslog} 'logs/RRL_CI'${trans}'_only_stack'${i}'.log' 'logs/RRL_CI'${trans}'_only_stack'${i}'_w.log'
${SPECPATH}/stack.py 'logs/RRL_CI'${trans}'_only_stack'${i}'.log' 'stacks/iter'${iter}'/RRL_CI'${trans}'_only_stack'${i}'.ascii' -m interpol --weight='list' \
                      --weight_list='logs/RRL_CI'${trans}'_only_stack'${i}'_w.log' --x_col=1 --y_col=2 --v_max=800 --v_min=-800
${SPECPATH}/sbsplot.py 'stacks/iter'${i}'/RRL_CI'${trans}'_only_stack'${i}'.ascii' -x 'Velocity (km s$^{-1}$)' plots/RRL_CI${trans}_only_stack${i}.pdf
n=`${SPECPATH}/getn.py logs/RRL_CI${trans}_only_stack${i}.log | sed -n 1p`
echo $n
cp stacks/iter${iter}/RRL_CI${trans}_only_stack${i}.ascii stacks/RRL_CI${trans}_only_n${n}.ascii
done

# beta
trans=beta
rmslog=logs/iter${iter}_sbs_rms.log
mv lines/iter${iter}/m${trans} lines/iter${iter}/${trans}
mkdir lines/iter${iter}/${trans}/sbs
cp lines/iter${iter}/sbs/lba_sim_SB*.ascii lines/iter${iter}/${trans}/sbs
${SPECPATH}/makeweightlist.py 'lines/iter'${iter}'/'${trans}'/sbs/lba_sim_SB*.ascii' ${rmslog} --f_col=0 --y_col=1 -t RRL_CIalpha,RRL_CIbeta,RRL_CIgamma,RRL_CIdelta --z=-1.567e-4 -m 1/rms2 -d 50
${SPECPATH}/makestacklist.py logs/RRL_CIalpha_good_lines.log ${rmslog} logs/RRL_CIalpha ${transs[alpha]} -p Last --path=lines/iter${iter}/${trans}/sbs --first=SB000
${SPECPATH}/makestacklist.py logs/RRL_CIgamma_good_lines.log ${rmslog} logs/RRL_CIgamma ${transs[alpha]} -p Last --path=lines/iter${iter}/${trans}/sbs --first=SB000
${SPECPATH}/makestacklist.py logs/RRL_CIdelta_good_lines.log ${rmslog} logs/RRL_CIdelta ${transs[alpha]} -p Last --path=lines/iter${iter}/${trans}/sbs --first=SB000
for t in alpha gamma delta
do
echo $t
stackmax=${transs[${t}]}
for (( i=1; i<=$stackmax; i++ ))
do
echo $i
${SPECPATH}/removemodel.py 'logs/RRL_CI'${t}'_stack'${i}'_SBs.log' 'models/RRL_CI'${t}'_stack'${i}'.ascii' 'lines/iter'${iter}'/'${trans}'/sbs/lba_sim' -t RRL_CI${t} \
                            --x_col=0 --y_col=1 --z=-1.567e-4 -p --plot_file=plots/iter${iter}/lba_sim_mstack${i}_RRL_CI${t}.pdf -m 'subtract'
done
done
${SPECPATH}/sbsplot.py 'lines/iter'${iter}'/'${trans}'/sbs/lba_sim_SB*.ascii' plots/only_RRL_CI${trans}_sbs.pdf --x_col=0 --y_col=1 -l --z=-1.567e-4 -t RRL_CIalpha,RRL_CIbeta,RRL_CIgamma,RRL_CIdelta
${SPECPATH}/spec2vel.py 'lines/iter'${iter}'/'${trans}'/sbs/lba_sim_SB*.ascii' 'lines/iter'${iter}'/'${trans}'/lba_sim' -t RRL_CI${trans} --z=-1.567e-4 --f_col=0
${SPECPATH}/goodlines.py 'lines/iter'${iter}'/'${trans}'/sbs/lba_sim_SB*.ascii' 'lines/iter'${iter}'/'${trans}'/lba_sim' -t RRL_CI${trans} --x_col=0 --z=-1.567e-4 -s 1
mv RRL_CI${trans}_good_lines.log logs/RRL_CI${trans}_only_good_lines.log
smax=${transs[${trans}]}
${SPECPATH}/makestacklist.py logs/RRL_CI${trans}_only_good_lines.log ${rmslog} logs/RRL_CI${trans}_only ${smax} -p Last --path=lines/iter${iter}/${trans}/sbs --first=SB000
for (( i=1; i<=$smax; i++ ))
do
echo $i
${SPECPATH}/matchweightlist.py ${rmslog} 'logs/RRL_CI'${trans}'_only_stack'${i}'.log' 'logs/RRL_CI'${trans}'_only_stack'${i}'_w.log'
${SPECPATH}/stack.py 'logs/RRL_CI'${trans}'_only_stack'${i}'.log' 'stacks/iter'${iter}'/RRL_CI'${trans}'_only_stack'${i}'.ascii' -m interpol --weight='list' \
                      --weight_list='logs/RRL_CI'${trans}'_only_stack'${i}'_w.log' --x_col=1 --y_col=2 --v_max=800 --v_min=-800
${SPECPATH}/sbsplot.py 'stacks/iter'${iter}'/RRL_CI'${trans}'_only_stack'${i}'.ascii' -x 'Velocity (km s$^{-1}$)' plots/RRL_CI${trans}_only_stack${i}.pdf
n=`${SPECPATH}/getn.py logs/RRL_CI${trans}_only_stack${i}.log | sed -n 1p`
echo $n
cp stacks/iter${iter}/RRL_CI${trans}_only_stack${i}.ascii stacks/RRL_CI${trans}_only_n${n}.ascii
done

# gamma
trans=gamma
rmslog=logs/iter${iter}_sbs_rms.log
mv lines/iter${iter}/m${trans} lines/iter${iter}/${trans}
mkdir lines/iter${iter}/${trans}/sbs
cp lines/iter${iter}/sbs/lba_sim_SB*.ascii lines/iter${iter}/${trans}/sbs
${SPECPATH}/makeweightlist.py 'lines/iter'${iter}'/'${trans}'/sbs/lba_sim_SB*.ascii' ${rmslog} --f_col=0 --y_col=1 -t RRL_CIalpha,RRL_CIbeta,RRL_CIgamma,RRL_CIdelta --z=-1.567e-4 -m 1/rms2 -d 50
${SPECPATH}/makestacklist.py logs/RRL_CIalpha_good_lines.log ${rmslog} logs/RRL_CIalpha ${transs[alpha]} -p Last --path=lines/iter${iter}/${trans}/sbs --first=SB000
${SPECPATH}/makestacklist.py logs/RRL_CIbeta_good_lines.log  ${rmslog} logs/RRL_CIbeta  ${transs[alpha]} -p Last --path=lines/iter${iter}/${trans}/sbs --first=SB000
${SPECPATH}/makestacklist.py logs/RRL_CIdelta_good_lines.log ${rmslog} logs/RRL_CIdelta ${transs[alpha]} -p Last --path=lines/iter${iter}/${trans}/sbs --first=SB000
for t in alpha beta delta
do
echo $t
stackmax=${transs[${t}]}
for (( i=1; i<=$stackmax; i++ ))
do
echo $i
${SPECPATH}/removemodel.py 'logs/RRL_CI'${t}'_stack'${i}'_SBs.log' 'models/RRL_CI'${t}'_stack'${i}'.ascii' 'lines/iter'${iter}'/'${trans}'/sbs/lba_sim' -t RRL_CI${t} \
                            --x_col=0 --y_col=1 --z=-1.567e-4 -p --plot_file=plots/iter${iter}/lba_sim_mstack${i}_RRL_CI${t}.pdf -m 'subtract'
done
done
${SPECPATH}/sbsplot.py 'lines/iter'${iter}'/'${trans}'/sbs/lba_sim_SB*.ascii' plots/only_RRL_CI${trans}_sbs.pdf --x_col=0 --y_col=1 -l --z=-1.567e-4 -t RRL_CIalpha,RRL_CIbeta,RRL_CIgamma,RRL_CIdelta
${SPECPATH}/spec2vel.py 'lines/iter'${iter}'/'${trans}'/sbs/lba_sim_SB*.ascii' 'lines/iter'${iter}'/'${trans}'/lba_sim' -t RRL_CI${trans} --z=-1.567e-4 --f_col=0
${SPECPATH}/goodlines.py 'lines/iter'${iter}'/'${trans}'/sbs/lba_sim_SB*.ascii' 'lines/iter'${iter}'/'${trans}'/lba_sim' -t RRL_CI${trans} --x_col=0 --z=-1.567e-4 -s 1
mv RRL_CI${trans}_good_lines.log logs/RRL_CI${trans}_only_good_lines.log
smax=${transs[${trans}]}
${SPECPATH}/makestacklist.py logs/RRL_CI${trans}_only_good_lines.log ${rmslog} logs/RRL_CI${trans}_only ${smax} -p Last --path=lines/iter${iter}/${trans}/sbs --first=SB000
for (( i=1; i<=$smax; i++ ))
do
echo $i
${SPECPATH}/matchweightlist.py ${rmslog} 'logs/RRL_CI'${trans}'_only_stack'${i}'.log' 'logs/RRL_CI'${trans}'_only_stack'${i}'_w.log'
${SPECPATH}/stack.py 'logs/RRL_CI'${trans}'_only_stack'${i}'.log' 'stacks/iter'${iter}'/RRL_CI'${trans}'_only_stack'${i}'.ascii' -m interpol --weight='list' \
                      --weight_list='logs/RRL_CI'${trans}'_only_stack'${i}'_w.log' --x_col=1 --y_col=2 --v_max=800 --v_min=-800
${SPECPATH}/sbsplot.py 'stacks/iter'${iter}'/RRL_CI'${trans}'_only_stack'${i}'.ascii' -x 'Velocity (km s$^{-1}$)' plots/RRL_CI${trans}_only_stack${i}.pdf
n=`${SPECPATH}/getn.py logs/RRL_CI${trans}_only_stack${i}.log | sed -n 1p`
echo $n
cp stacks/iter${iter}/RRL_CI${trans}_only_stack${i}.ascii stacks/RRL_CI${trans}_only_n${n}.ascii
done

# delta
trans=delta
rmslog=logs/iter${iter}_sbs_rms.log
mv lines/iter${iter}/m${trans} lines/iter${iter}/${trans}
mkdir lines/iter${iter}/${trans}/sbs
cp lines/iter${iter}/sbs/lba_sim_SB*.ascii lines/iter${iter}/${trans}/sbs
${SPECPATH}/makeweightlist.py 'lines/iter'${iter}'/'${trans}'/sbs/lba_sim_SB*.ascii' ${rmslog} --f_col=0 --y_col=1 -t RRL_CIalpha,RRL_CIbeta,RRL_CIgamma,RRL_CIdelta --z=-1.567e-4 -m 1/rms2 -d 50
${SPECPATH}/makestacklist.py logs/RRL_CIalpha_good_lines.log ${rmslog} logs/RRL_CIalpha ${transs[alpha]} -p Last --path=lines/iter${iter}/${trans}/sbs --first=SB000
${SPECPATH}/makestacklist.py logs/RRL_CIbeta_good_lines.log  ${rmslog} logs/RRL_CIbeta  ${transs[alpha]} -p Last --path=lines/iter${iter}/${trans}/sbs --first=SB000
${SPECPATH}/makestacklist.py logs/RRL_CIgamma_good_lines.log ${rmslog} logs/RRL_CIgamma ${transs[alpha]} -p Last --path=lines/iter${iter}/${trans}/sbs --first=SB000
for t in alpha beta gamma
do
echo $t
stackmax=${transs[${t}]}
for (( i=1; i<=$stackmax; i++ ))
do
echo $i
${SPECPATH}/removemodel.py 'logs/RRL_CI'${t}'_stack'${i}'_SBs.log' 'models/RRL_CI'${t}'_stack'${i}'.ascii' 'lines/iter'${iter}'/'${trans}'/sbs/lba_sim' -t RRL_CI${t} \
                            --x_col=0 --y_col=1 --z=-1.567e-4 -p --plot_file=plots/iter${iter}/lba_sim_mstack${i}_RRL_CI${t}.pdf -m 'subtract'
done
done
${SPECPATH}/sbsplot.py 'lines/iter'${iter}'/'${trans}'/sbs/lba_sim_SB*.ascii' plots/only_RRL_CI${trans}_sbs.pdf --x_col=0 --y_col=1 -l --z=-1.567e-4 -t RRL_CIalpha,RRL_CIbeta,RRL_CIgamma,RRL_CIdelta
${SPECPATH}/spec2vel.py 'lines/iter'${iter}'/'${trans}'/sbs/lba_sim_SB*.ascii' 'lines/iter'${iter}'/'${trans}'/lba_sim' -t RRL_CI${trans} --z=-1.567e-4 --f_col=0
${SPECPATH}/goodlines.py 'lines/iter'${iter}'/'${trans}'/sbs/lba_sim_SB*.ascii' 'lines/iter'${iter}'/'${trans}'/lba_sim' -t RRL_CI${trans} --x_col=0 --z=-1.567e-4 -s 1
mv RRL_CI${trans}_good_lines.log logs/RRL_CI${trans}_only_good_lines.log
smax=${transs[${trans}]}
${SPECPATH}/makestacklist.py logs/RRL_CI${trans}_only_good_lines.log ${rmslog} logs/RRL_CI${trans}_only ${smax} -p Last --path=lines/iter${iter}/${trans}/sbs --first=SB000
for (( i=1; i<=$smax; i++ ))
do
echo $i
${SPECPATH}/matchweightlist.py ${rmslog} 'logs/RRL_CI'${trans}'_only_stack'${i}'.log' 'logs/RRL_CI'${trans}'_only_stack'${i}'_w.log'
${SPECPATH}/stack.py 'logs/RRL_CI'${trans}'_only_stack'${i}'.log' 'stacks/iter'${iter}'/RRL_CI'${trans}'_only_stack'${i}'.ascii' -m interpol --weight='list' \
                      --weight_list='logs/RRL_CI'${trans}'_only_stack'${i}'_w.log' --x_col=1 --y_col=2 --v_max=800 --v_min=-800
${SPECPATH}/sbsplot.py 'stacks/iter'${iter}'/RRL_CI'${trans}'_only_stack'${i}'.ascii' -x 'Velocity (km s$^{-1}$)' plots/RRL_CI${trans}_only_stack${i}.pdf
n=`${SPECPATH}/getn.py logs/RRL_CI${trans}_only_stack${i}.log | sed -n 1p`
echo $n
cp stacks/iter${iter}/RRL_CI${trans}_only_stack${i}.ascii stacks/RRL_CI${trans}_only_n${n}.ascii
done
