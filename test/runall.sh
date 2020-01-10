#!/bin/bash
set -e
echo "Performance TC"
testfolderlocation='/data/Seafile/GraphsAndMatrices/floridadata'

echo "Starting with : $testfolderlocation"

until ../build/performTestCase $testfolderlocation 1 0 1 256,3,2,4,4,16,512,8 3 d
do
  echo "Try again 256,3,2,4,4,16,512,8"
done

echo "Testcase done"