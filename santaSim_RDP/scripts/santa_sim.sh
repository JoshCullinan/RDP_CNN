#!/bin/sh

GenCount=(2500 3250 4000) 
RecomProb=(0.005 0.01 0.02) 
MutRate=(4E-5 8E-5 12E-5)
SampSize=(50 100 200)
CORES=20

# 2 3 4 5

# for file in 2
# do
file=3
    mkdir -p /scratch/clljos001/XML-$file
    for GC in "${GenCount[@]}"
    do
        for RP in "${RecomProb[@]}"
        do
            for MR in "${MutRate[@]}"
            do
                for SS in "${SampSize[@]}"
                do
                    ramusage=$(free | awk '/Mem/{printf("RAM Usage: %.2f\n"), $3/$2*100}'| awk '{print $3}')
                    echo "Memory Current Usage is: $ramusage%"

                    for thread in $( seq 1 $CORES )
                    do
                        ( cd /scratch/clljos001 && 
                            mkdir -p folder-$thread && 
                            cd folder-$thread && 
                            java -jar /home/clljos001/santa.jar -generationCount=$GC -recombinationProbability=$RP -mutationRate=$MR -sampleSize=$SS /home/clljos001/xml/$file.xml &&
                            mv alignment_0.fa ../XML-$file/alignment_XML$file-$GC-$RP-$MR-$SS-$thread.fa &&
                            mv recombination_events.txt ../XML-$file/recombination_events_XML$file-$GC-$RP-$MR-$SS-$thread.txt &&
                            mv sequence_events_map.txt ../XML-$file/sequence_events_map_XML$file-$GC-$RP-$MR-$SS-$thread.txt 
                        )&
                    done
                    wait
                done
            done
        done
    done
# done