#!/bin/bash

lang=$1

train_file=$lang"-train.xml"
dev_file=$lang"-dev.xml"
svm_file="SVM-"$lang".answer"
knn_file="KNN-"$lang".answer"
best_file="Best-"$lang".answer"
key_file=$lang"-dev.key"

python main.py data/$train_file data/$dev_file $knn_file $svm_file $best_file $lang
