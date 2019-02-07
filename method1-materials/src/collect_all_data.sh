#!/bin/bash

 
    DST="./data_all"
    
    TRAINA=$DST
    TRAINA+="/trainA"
    TRAINB=$DST
    TRAINB+="/trainB"
    
    mkdir "$DST"
    mkdir "$TRAINA"
    mkdir "$TRAINB"
    
    for j in `seq 1 15`; do
        SRC="./kidney_dataset_"
        SRC+="$j"
            
        FRAMES=$SRC
        FRAMES+="/left/"
        cd "$FRAMES"
        for f in *.png; do
            DST_NAME="../../"
            DST_NAME+="$TRAINA"

            DST_NAME+=/"$j"_
            DST_NAME+="$f"
            echo $DST_NAME
            cp "$f" "$DST_NAME"            
        done
        cd "../../"
            
        GT=$SRC
        GT+="/GT/"
        cd "$GT"
        for f in *.png; do
            DST_NAME="../../"
            DST_NAME+="$TRAINB"
            
            DST_NAME+=/"$j"_
            DST_NAME+="$f"
            echo $DST_NAME
            cp "$f" "$DST_NAME" 
        done
        cd "../../"

    done 
