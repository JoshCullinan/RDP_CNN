#!/bin/sh

#SBATCH --account=cbio
#SBATCH --partition=curie
#SBATCH --nodes=1 --ntasks=64
#SBATCH --time=90:00:00
#SBATCH --job-name="SantaSim"
#SBATCH --mail-user=clljos001@myuct.ac.za
#SBATCH --mail-type=ALL

module load java/jdk-11

RecomProb=(0.005 0.01 0.02) 
MutRate=(4E-5 8E-5 12E-5)
SampSize=(50 100 200)
CORES=6

file=5
GC=2500

mkdir -p /scratch/clljos001/XML-$file
for Round in 1 2
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
                        mkdir -p folder-$thread-$Round && 
                        cd folder-$thread-$Round && 
                        java -jar /home/clljos001/santa.jar -generationCount=$GC -recombinationProbability=$RP -mutationRate=$MR -sampleSize=$SS /home/clljos001/xml/$file.xml &&
                        mv alignment_0.fa ../XML-$file/alignment_XML$file-$GC-$RP-$MR-$SS-$thread-$Round.fa &&
                        mv recombination_events.txt ../XML-$file/recombination_events_XML$file-$GC-$RP-$MR-$SS-$thread-$Round.txt &&
                        mv sequence_events_map.txt ../XML-$file/sequence_events_map_XML$file-$GC-$RP-$MR-$SS-$thread-$Round.txt 
                    )&
                done
                
                wait
            done
        done
    done
done