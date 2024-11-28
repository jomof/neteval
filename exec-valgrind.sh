#!/bin/bash
valgrind --tool=cachegrind --cachegrind-out-file=/tmp/cachegrind.out $1   # Adjust this to your actual program path
cg_annotate /tmp/cachegrind.out