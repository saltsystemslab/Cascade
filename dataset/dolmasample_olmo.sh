#!/bin/bash

# Install `awscli`
python -m pip install awscli

# See the files and size
aws s3 ls --no-sign-request \
  s3://infini-gram-lite/index/v4_dolmasample_olmo/ --human-readable

# Download only the token array(s), skipping the suffix array (table.*) etc.
aws s3 cp --no-sign-request --recursive \
  s3://infini-gram-lite/index/v4_dolmasample_olmo/ ./dolmasample_olmo/ \
  --exclude "*" --include "tokenized.*"
