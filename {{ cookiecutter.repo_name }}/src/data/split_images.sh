#!/bin/bash
# -*- coding: utf-8 -*-
set -euo pipefail

# set default variables
script_name=$(basename $0)
script_version="0.0.1"
INPUT_DIR=0
TILE_SIZE=256

####################
# Help information
####################
Help()
{
   # Display Help
   echo -e "Usage: $script_name [-d|t|h|V]\n
   Configure git and conda for remote development on HPC cluster.\n
   Options:
   \td     input directory containing subfolders with unique classes for classification or
           'data' and 'mask' for segmentation
   \th     Print  Help.
   \tt     Tile size for splitting images
   \tV     Print software version and exit.\n"
}

####################
# Process arguments
####################
while getopts "hd:t:V" option; do
   case $option in
      h) # display Help
         Help
         exit 0;;
      d) # input directory
        INPUT_DIR="$OPTARG";;
  	  t) # tile size
  	    TILE_SIZE="$OPTARG";;
   	  V) # python version for conda
   	    echo -e "$script_name $script_version"
   	    exit 0;;
      \?) # invalid option
        echo "Error: Invalid option"
        exit 1;;
   esac
done

# Error checking: exit if the following vars not entered
if [[ "$INPUT_DIR" == "0" ]]; then echo -e "Missing or invalid input argument -d Input directory (str)."; exit 1; fi
if [[ "$TILE_SIZE" == "0" ]]; then echo -e "Invalid input argument -t Tile size (int)."; exit 1; fi

find $INPUT_DIR -type f -name "*.png" | sort | while read FILE; do
  OUTPUT_FILE=$(echo -e $FILE | sed -e s/raw/processed/g)
  OUTPUT_DIR=$(dirname $OUTPUT_FILE)
  if [ ! -d $OUTPUT_DIR ]; then
    echo -e "Creating output directory\t$OUTPUT_DIR"
    mkdir -p $OUTPUT_DIR
  fi

  convert -crop "$TILE_SIZE"x"$TILE_SIZE" $FILE ${OUTPUT_FILE%*.png}"_cropped_%04d.png"
done

