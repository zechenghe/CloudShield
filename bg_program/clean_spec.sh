#!/bin/bash

ps -ef | grep "run_spec" | awk '{print $2;}' | xargs -r kill
ps -ef | grep "runspec" | awk '{print $2;}' | xargs -r kill
ps -ef | grep "specinvoke" | awk '{print $2;}' | xargs -r kill
