#!/bin/bash
if [ "$#" -ne 1 ]; then
	    echo "Usage: $0 directory"
	        exit 1
fi

# Change to the specified directory
cd "$1" || exit

# Fetch the latest changes from the origin
git fetch origin

# Pull the latest changes from the current branch
git pull
