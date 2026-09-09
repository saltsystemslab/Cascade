#!/bin/bash

# Install `awscli`
python -m pip install awscli

# See the files and size
aws s3 ls --no-sign-request \
  s3://infini-gram-lite/index/v4_dolmasample_olmo/ --human-readable

# Fix destination
mkdir -p /scratch/$USER/corpora/dolmasample_olmo

# Download only the token array(s), skipping the suffix array (table.*) etc.
aws s3 cp --no-sign-request --recursive \
  s3://infini-gram-lite/index/v4_dolmasample_olmo/ /scratch/$USER/corpora/dolmasample_olmo/ \
  --exclude "*" --include "tokenized.*"
