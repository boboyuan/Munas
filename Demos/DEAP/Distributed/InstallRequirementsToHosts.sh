#!/bin/bash

python=python3.8
directory="venv"
host_file="hosts"
requirements="requirements.txt"

if ! command -v pdsh &> /dev/null
then
    echo "pdsh could not be found, please install and rerun script"
    exit 1
fi

while getopts u:h:p:r:d: flag
do
    case "${flag}" in
        u) username=${OPTARG};;
        h) host_file=${OPTARG};;
        p) python=${OPTARG};;
        r) requirements=${OPTARG};;
        d) directory=${OPTARG};;
    esac
done

first_host=`head -n 1 hosts`
scp ../$requirements $username@$first_host:~/
scp ../CreateVEnv.sh $username@$first_host:~/
ssh $username@$first_host "bash ~/CreateVEnv.sh -p $python -r ~/$requirements -d $directory"
python_command="grep -q '$directory/env/bin' \$HOME/.bashrc || echo \"PATH=$directory/env/bin:$PATH\" >> \$HOME/.bashrc"
echo $python_command
ssh $username@$first_host $python_command
python_path="$directory/lib/$python/site-packages/"
python_path_command="grep -q \"$python_path\" \$HOME/.bashrc || echo \"PYTHONPATH=$python_path:\$PYTHONPATH\" >> \$HOME/.bashrc"
echo $python_path_command
ssh $username@$first_host $python_path_command
activate_file="$directory/bin/activate"
activate_command="grep -q \"source $activate_file\"  \$HOME/.bashrc || echo \"source $activate_file\" >> \$HOME/.bashrc"
echo $activate_command
ssh $username@$first_host $activate_command

pdsh -w ^hosts -l $username -R ssh "source ~/.bashrc"
