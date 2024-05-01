#!/usr/bin/env bash

ldd /opt/conda/lib/python3.11/site-packages/cv2/cv2.abi3.so | \
  grep '=>' | \
  awk -F ' => ' '{ print $2 }' | \
  awk -F ' ' '{ print $1 }' | \
  xargs readlink -f | \
  awk -F '/' '{ print $NF }' | \
  awk -F '.' '{ print $1 }' | \
  awk -F '-' '{ print "*"$1"*.so*" }' | \
  awk '!/cuda/ && !/nvcuvid/ && !/nvidia/ && !/nppc/ && !/nppial/ && !/nppicc/ && !/nppidei/ && !/nppif/ && !/nppig/ && !/nppim/ && !/nppist/ && !/nppitc/ && !/npps/ && !/cublas/ && !/cudnn/ && !/cufft/' | \
  xargs -L 1 find /opt/conda/ -name | \
  xargs readlink -f | \
  awk '!/cuda/ && !/nvcuvid/ && !/nvidia/ && !/nppc/ && !/nppial/ && !/nppicc/ && !/nppidei/ && !/nppif/ && !/nppig/ && !/nppim/ && !/nppist/ && !/nppitc/ && !/npps/ && !/cublas/ && !/cudnn/ && !/cufft/' | \
  awk -F '.so.' '{ print length($2), " " $2, " ", $0 }' | \
  sort -r | \
  awk -F ' ' '{ print $NF }' | \
  uniq
