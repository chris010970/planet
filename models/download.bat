@echo off
set GCS_BUCKET=gs://eo-ald-update
set REPO=planet

gsutil cp %GCS_BUCKET%/%REPO%/models/vgg16-256-128.zip .
