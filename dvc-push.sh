#!/bin/sh
dir="dvcfiles"
if [ ! -d $dir ]; then
    mkdir $dir
fi
# experiments name
name="default"
while getopts "n:" opt; do
    case $opt in
        n)
            name=$OPTARG
            ;;
        ?)
            echo $@ is not valid option
            exit 0
            ;;
    esac
done
ckpt="$(python dvc-script.py $name)"
#echo $ckpt
dvc add $ckpt --file "$dir/trained_models.dvc"
dvc push dvcfiles/trained_models.dvc