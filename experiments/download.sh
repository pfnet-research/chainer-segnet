#!/bin/bash

wget https://github.com/alexgkendall/SegNet-Tutorial/archive/master.zip
unzip master.zip && rm -rf master.zip
mv SegNet-Tutorial-master/CamVid data
rm -rf mv SegNet-Tutorial-master
