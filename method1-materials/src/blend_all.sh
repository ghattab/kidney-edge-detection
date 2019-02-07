#!/bin/bash

for i in `seq 14 15`; do 
    ROOT="~/data_"
    ROOT+="$i"
    ROOT+="/results/kidney_"
    ROOT+="$i"
    
    for j in `seq 1 8`; do 
        e=$(($j * 25))
        
        DIRA="$ROOT"
        DIRA+="/"
        DIRA+="$e"
        DIRA+="_test/images/real_A"
        
        DIRB="$ROOT"
        DIRB+="/"
        DIRB+="$e"
        DIRB+="_test/images/fake_B"
        
        DST="$ROOT"
        DST+="/"
        DST+="$e"
        DST+="_test/images/combined"
        
        mkdir "$DST"
        
        python3 blend.py "$DIRA" "$DIRB" "$DST"
    done  
done
