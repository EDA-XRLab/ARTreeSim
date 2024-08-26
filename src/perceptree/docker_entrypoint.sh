#!/bin/bash

redis-server --protected-mode no --daemonize yes
/venv/bin/python /app/detection_app.py
