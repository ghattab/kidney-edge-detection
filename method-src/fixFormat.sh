#!/usr/bin/env bash

baseFolder=fixedSubmission 

mkdir $baseFolder

mkdir $baseFolder/Ori-NoStrict
mkdir $baseFolder/Ori-Strict

mkdir $baseFolder/OriSp-NoStrict
mkdir $baseFolder/OriSp-Strict

mkdir $baseFolder/Ori3DSk-NoStrict
mkdir $baseFolder/Ori3DSk-Strict

mkdir $baseFolder/Ori3DSkSp-NoStrict
mkdir $baseFolder/Ori3DSkSp-Strict

targetFolder=Ori-Strict
echo $targetFolder
cp -r boundaries_original_algo/strictVersion/* $baseFolder/${targetFolder}/
for i in `seq 1 15`; do
	mkdir $baseFolder/${targetFolder}/boundary_wo${i}
	mv $baseFolder/${targetFolder}/boundary_strict_wo${i}/test/x* $baseFolder/${targetFolder}/boundary_wo${i}/
	for j in `seq 1 20`; do
		mv $baseFolder/${targetFolder}/boundary_wo${i}/x${j} $baseFolder/${targetFolder}/boundary_wo${i}/kidney_dataset_${j}
	done
done
rm -r $baseFolder/${targetFolder}/boundary_strict_wo*





targetFolder=Ori-NoStrict
echo $targetFolder
cp -r boundaries_original_algo/nonStrictVersion/* $baseFolder/${targetFolder}/
for i in `seq 1 15`; do
	mv $baseFolder/${targetFolder}/boundary_wo${i}/test/x* $baseFolder/${targetFolder}/boundary_wo${i}/
	rm -r $baseFolder/${targetFolder}/boundary_wo${i}/test
	for j in `seq 1 20`; do
		mv $baseFolder/${targetFolder}/boundary_wo${i}/x${j} $baseFolder/${targetFolder}/boundary_wo${i}/kidney_dataset_${j}
	done
done

targetFolder=OriSp-Strict
echo $targetFolder
cp -r boundaries_original_algo_spur_removal/strictVersion/* $baseFolder/${targetFolder}/
for i in `seq 1 15`; do
	mkdir $baseFolder/${targetFolder}/boundary_wo${i}
	mv $baseFolder/${targetFolder}/boundary_strict_wo${i}/test/x* $baseFolder/${targetFolder}/boundary_wo${i}/
	for j in `seq 1 20`; do
		mv $baseFolder/${targetFolder}/boundary_wo${i}/x${j} $baseFolder/${targetFolder}/boundary_wo${i}/kidney_dataset_${j}
	done
done
rm -r $baseFolder/${targetFolder}/boundary_strict_wo*

targetFolder=OriSp-NoStrict
echo $targetFolder
cp -r boundaries_original_algo_spur_removal/nonStrictVersion/* $baseFolder/${targetFolder}/
for i in `seq 1 15`; do
	mv $baseFolder/${targetFolder}/boundary_wo${i}/test/x* $baseFolder/${targetFolder}/boundary_wo${i}/
	rm -r $baseFolder/${targetFolder}/boundary_wo${i}/test
	for j in `seq 1 20`; do
		mv $baseFolder/${targetFolder}/boundary_wo${i}/x${j} $baseFolder/${targetFolder}/boundary_wo${i}/kidney_dataset_${j}
	done
done


targetFolder=Ori3DSk-Strict
echo $targetFolder
cp -r boundaries_original_algo_3d_skeletonize/strictVersion/* $baseFolder/${targetFolder}/
for i in `seq 1 15`; do
	mkdir $baseFolder/${targetFolder}/boundary_wo${i}
	mv $baseFolder/${targetFolder}/boundary_strict_wo${i}/test/x* $baseFolder/${targetFolder}/boundary_wo${i}/
	for j in `seq 1 20`; do
		mv $baseFolder/${targetFolder}/boundary_wo${i}/x${j} $baseFolder/${targetFolder}/boundary_wo${i}/kidney_dataset_${j}
	done
done
rm -r $baseFolder/${targetFolder}/boundary_strict_wo*

targetFolder=Ori3DSk-NoStrict
echo $targetFolder
cp -r boundaries_original_algo_3d_skeletonize/nonStrictVersion/* $baseFolder/${targetFolder}/
for i in `seq 1 15`; do
	mv $baseFolder/${targetFolder}/boundary_wo${i}/test/x* $baseFolder/${targetFolder}/boundary_wo${i}/
	rm -r $baseFolder/${targetFolder}/boundary_wo${i}/test
	for j in `seq 1 20`; do
		mv $baseFolder/${targetFolder}/boundary_wo${i}/x${j} $baseFolder/${targetFolder}/boundary_wo${i}/kidney_dataset_${j}
	done
done


targetFolder=Ori3DSkSp-Strict
echo $targetFolder
cp -r boundaries_original_algo_3d_skeletonize_spur_removal/strictVersion/* $baseFolder/${targetFolder}/
for i in `seq 1 15`; do
	mkdir $baseFolder/${targetFolder}/boundary_wo${i}
	mv $baseFolder/${targetFolder}/boundary_strict_wo${i}/test/x* $baseFolder/${targetFolder}/boundary_wo${i}/
	for j in `seq 1 20`; do
		mv $baseFolder/${targetFolder}/boundary_wo${i}/x${j} $baseFolder/${targetFolder}/boundary_wo${i}/kidney_dataset_${j}
	done
done
rm -r $baseFolder/${targetFolder}/boundary_strict_wo*

targetFolder=Ori3DSkSp-NoStrict
echo $targetFolder
cp -r boundaries_original_algo_3d_skeletonize_spur_removal/nonStrictVersion/* $baseFolder/${targetFolder}/
for i in `seq 1 15`; do
	mv $baseFolder/${targetFolder}/boundary_wo${i}/test/x* $baseFolder/${targetFolder}/boundary_wo${i}/
	rm -r $baseFolder/${targetFolder}/boundary_wo${i}/test
	for j in `seq 1 20`; do
		mv $baseFolder/${targetFolder}/boundary_wo${i}/x${j} $baseFolder/${targetFolder}/boundary_wo${i}/kidney_dataset_${j}
	done
done
