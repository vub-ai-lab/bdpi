#!/bin/bash

cat $1 | grep -- out- | cut -d '"' -f 2 | cut -d ' ' -f 5 | cut -d ')' -f 1 | while read f
do
	echo "$f..."
	cp ../$f .
done
