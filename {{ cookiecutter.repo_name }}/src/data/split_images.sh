#!/bin/bash
# -*- coding: utf-8 -*-
# Copyright (c) 2021 Jeffrey J. Nirschl. All rights reserved.
#
# Licensed under the MIT license. See the LICENSE.md file in the project
# root directory for full license information.
#
# Time-stamp: <2021-10-22 16:34:10 jjn>
# ======================================================================
set -euo pipefail

INPUT_DIR=${1-"/home/jjn/Documents/GitHub/{{cookiecutter.repo_name}}/data/raw/"}
TILE_SIZE=${2-256}

find $INPUT_DIR -type f -name "*.png" | sort | while read FILE; do
  OUTPUT_FILE=$(echo -e $FILE | sed -e s/raw/processed/g)
  OUTPUT_DIR=$(dirname $OUTPUT_FILE)
  if [ ! -d $OUTPUT_DIR ]; then
    echo -e "Creating output directory\t$OUTPUT_DIR"
    mkdir -p $OUTPUT_DIR
  fi

  convert -crop "$TILE_SIZE"x"$TILE_SIZE" $FILE ${OUTPUT_FILE%*.png}"_cropped_%04d.png"
done

