#!/bin/bash

while getopts c:d: flag
do
    case "${flag}" in
        c) config=${OPTARG};;
        d) demo=${OPTARG};;
    esac
done

if [ -f "$config" ]; then
  echo "Config file found"
else
  echo "Config file '$config' doesn't exist"
fi

if [ -f "$demo" ]; then
  echo "Demo script file found"
else
  echo "Demo script file '$demo' doesn't exist"
fi

setvalue() {
  sed -i "s,^\($1[ ]*=\).*,\1$2,g" $3
}

goalparamvectorstarts=(1000 5000 10000)
goalparamvectorends=(5000 25000 50000)

for i in ${!goalparamvectorstarts[*]}
do
  echo "Parameter goal vector start: ${goalparamvectorstarts[$i]}"
  echo "Parameter goal vector stop: ${goalparamvectorends[$i]}"
  setvalue GoalParamVectorStart ${goalparamvectorstarts[$i]} $config
  setvalue GoalParamVectorEnd ${goalparamvectorends[$i]} $config
  python3.8 $demo --config $config
done
