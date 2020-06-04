@echo off
set GCS_BUCKET=gs://eo-ald-update
set REPO=planet

gsutil cp %GCS_BUCKET%/%REPO%/data/subset-1.zip .
