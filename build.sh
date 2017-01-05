#!/bin/bash

#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, You can obtain one at http://mozilla.org/MPL/2.0/.

#this will remove any partial builds, if there was a failure


if [ "$1" == "clean" ]
then
    rm *.so
    exit
fi


python setup.py build_ext --inplace 
rm -r build/
echo "~~You made it!~~"
