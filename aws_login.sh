#!/usr/bin/env bash
set -e

export AWS_PROFILE=254105684736_DataScientist
export AWS_REGION=eu-west-1
export BEDROCK_MODEL_ID="arn:aws:bedrock:eu-west-1:254105684736:inference-profile/eu.anthropic.claude-opus-4-6-v1"

echo "Checking AWS session for profile: $AWS_PROFILE"

if aws sts get-caller-identity --profile "$AWS_PROFILE" > /dev/null 2>&1; then
  echo "AWS SSO session is active."
else
  echo "AWS SSO session expired or missing. Starting login..."
  aws sso login --profile "$AWS_PROFILE"
fi

echo "AWS identity:"
aws sts get-caller-identity --profile "$AWS_PROFILE"

echo "Ready."