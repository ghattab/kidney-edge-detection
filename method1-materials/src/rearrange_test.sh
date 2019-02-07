#!/bin/bash

for i in `seq 15 15`; do    
    DST="./data_"
    DST+="$i"
    
    TEST=$DST
    TEST+="/test_frames"
    
    mkdir "$TEST"    

    
    SRC="./kidney_dataset_"
    SRC+="$i"
        
    FRAMES=$SRC
    FRAMES+="/left/"
    cd "$FRAMES"
    for f in *.png; do
        DST_NAME="../../"
        DST_NAME+="$TEST"
        DST_NAME+=/"$i"_
        DST_NAME+="$f"
        echo $DST_NAME
        cp "$f" "$DST_NAME"            
    done
    cd "../../"
    
    SRC="./Test/kidney_dataset_"
    SRC+="$i"
        
    FRAMES=$SRC
    FRAMES+="/left/"
    cd "$FRAMES"
    for f in *.png; do
        DST_NAME="../../../"
        DST_NAME+="$TEST"
        DST_NAME+=/"$i"_
        DST_NAME+="$f"
        echo $DST_NAME
        cp "$f" "$DST_NAME"            
    done
    cd "../../../"


done  