#!/usr/bin/env bash

# Before running this script, write the libraries you want to search for in
# ./opencv-libs.txt, e.g.:
# cat >./libs.txt <<EOF
# lib
# lib2
# EOF

cat ./opencv-libs.txt | \
  awk -F'/' '{ print $NF }' | \
  xargs -L 1 find / -name | \
  awk '!/Permission denied/' | \
  xargs readlink -f | \
  awk -F '/' '{ print $NF }' | \
  awk -F '.' '{ print $1 }' | \
  awk -F '-' '{ print "*"$1"*.so*" }' | \
  xargs -L 1 find / -name | \
  awk '!/cuda/ && !/nvcuvid/ && !/nvidia/ && !/nppc/ && !/nppial/ && !/nppicc/ && !/nppidei/ && !/nppif/ && !/nppig/ && !/nppim/ && !/nppist/ && !/nppitc/ && !/npps/ && !/cublas/ && !/cudnn/ && !/cufft/' | \
  awk -F '.so.' '{ print length($2), " " $2, " ", $0 }' | \
  sort -r | \
  awk -F ' ' '{ print $NF }' | \
  uniq
