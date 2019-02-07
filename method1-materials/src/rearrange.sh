#!/bin/bash

for i in `seq 1 15`; do    
    DST="./data_"
    DST+="$i"
    
    TRAINA=$DST
    TRAINA+="/trainA"
    TRAINB=$DST
    TRAINB+="/trainB"
    TESTA=$DST
    TESTA+="/testA"
    TESTB=$DST
    TESTB+="/testB"
    
    mkdir "$DST"
    mkdir "$TRAINA"
    mkdir "$TRAINB"
    mkdir "$TESTA"
    mkdir "$TESTB"
    
    for j in `seq 1 15`; do
        SRC="./kidney_dataset_"
        SRC+="$j"
            
        FRAMES=$SRC
        FRAMES+="/left_frames/"
        cd "$FRAMES"
        for f in *.png; do
            DST_NAME="../../"
            if ! [ $j -eq $i ]
            then
                DST_NAME+="$TRAINA"
            else
                DST_NAME+="$TESTA"
            fi
            DST_NAME+=/"$j"_
            DST_NAME+="$f"
            echo $DST_NAME
            cp "$f" "$DST_NAME"            
        done
        cd "../../"
            
        GT=$SRC
        GT+="/ground_truth/"
        cd "$GT"
        for f in *.png; do
            DST_NAME="../../"
            if ! [ $j -eq $i ]
            then
                DST_NAME+="$TRAINB"
            else
                DST_NAME+="$TESTB"
            fi
            DST_NAME+=/"$j"_
            DST_NAME+="$f"
            echo $DST_NAME
            cp "$f" "$DST_NAME" 
        done
        cd "../../"

    done 
done  