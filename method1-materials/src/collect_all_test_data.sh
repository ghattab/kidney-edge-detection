#!/bin/bash

 
    DST="./data_all"
    
    TRAINA=$DST
    TRAINA+="/testA"
    
    mkdir "$DST"
    mkdir "$TRAINA"
    
    for j in `seq 16 20`; do
        SRC="./Test/kidney_dataset_"
        SRC+="$j"
            
        FRAMES=$SRC
        FRAMES+="/left/"
        cd "$FRAMES"
        for f in *.png; do
            DST_NAME="../../../"
            DST_NAME+="$TRAINA"

            DST_NAME+=/"$j"_
            DST_NAME+="$f"
            echo $DST_NAME
            cp "$f" "$DST_NAME"            
        done
        cd "../../../"

    done 
