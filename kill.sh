ps -ef | grep infer.py | grep -v grep | awk '{print $2}' | xargs kill -9
ps -ef | grep parallel_infer.sh | grep -v grep | awk '{print $2}' | xargs kill -9