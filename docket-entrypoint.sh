#!/usr/bin/env bash

echo "Waiting for MySQL..."

while ! nc -z db 3306; do
  sleep 0.5
done

echo "MySQL started"

flask db init
flask db migrate
flask db upgrade

cd /app
python3 router.py