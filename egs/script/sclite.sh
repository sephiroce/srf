#!/bin/bash
sclite -h $2 -r $1 -i wsj -o pralign -o sum
cat $2.sys