#!/bin/bash
locate FindCUDA.cmake| grep Modules| awk 'BEGIN {FS="/"} {printf $4}'
