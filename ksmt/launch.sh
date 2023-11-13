#!/bin/bash

COMMAND='./gradlew run'

while true; do
    $COMMAND
    RETURN_CODE=$?

    if [ $RETURN_CODE -ne 0 ]; then
        echo "Process exited with non-zero return code. Restarting..."
    else
        echo "Process exited successfully."
        break
    fi

done
