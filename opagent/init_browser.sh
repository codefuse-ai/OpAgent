echo "端口:$HOST_PORTS"
export HOST_PORTS=$HOST_PORTS
export NUM_BROWSERS=${NUM_BROWSERS:-10}
# TODO: 配置您的浏览器配置路径
export SAVE_MODEL_PATH=${SAVE_MODEL_PATH:-"./browser_config"}
TASK_ID=$TASK_ID
host_name=$(hostname)
real_ip_address=$(hostname -I | awk '{print $1}')

echo "启动浏览器配置任务..."
# 启动真实的浏览器配置任务
nohup python3 -m init_browser &
