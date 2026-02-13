import os
import paramiko
import socks
import socket
from functools import wraps
import time
from .auto_login import generate_new_cookies
from .auto_login_asynico import async_generate_new_cookies
from .utils import with_timeout_legacy
import asyncio 
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from .auto_login import URLS, SITES, SITE_PORT_MAP 
from .auto_login import ONE_YEAR_IN_SECONDS, ONE_YEAR_IN_MINUTES
from typing import List, Dict, Any, Tuple, Set
from itertools import combinations
import logging
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))
VLM_EXP_DEBUG = os.environ.get('VLM_EXP_DEBUG', '0')
VLM_EXP_DEBUG = str(VLM_EXP_DEBUG)

WEBARENA_AUTH_PATH = os.environ["WEBARENA_AUTH_PATH"]
from .auto_login_asynico import async_renew_comb, async_verif_web
# TODO: 配置您的资源目录
base_dir = os.getenv("ECS_RESOURCES_DIR", "./ecs_resources")
# 确保文件名是正确的
filename = "ecs_instance_list_cn-hangzhou_2025-11-15.csv"
full_file_path = os.path.join(base_dir, filename)

def get_proxy_settings():
    # TODO: 如需使用代理，请配置环境变量 SOCKS_PROXY
    proxy = os.getenv("SOCKS_PROXY", "")
    if not proxy:
        return None
    parts = proxy.split(':')
    if len(parts) == 2:
        return parts[0], int(parts[1])
    return None

def create_proxy_sock(target_host, target_port):
    proxy_settings = get_proxy_settings()
    if proxy_settings:
        proxy_host, proxy_port = proxy_settings
        sock = socks.create_connection(
            (target_host, target_port),
            proxy_type=socks.HTTP,
            proxy_addr=proxy_host,
            proxy_port=proxy_port
        )
        return sock
    return None

def retry(max_tries=5, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            tries = 0
            while tries < max_tries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    tries += 1
                    if tries == max_tries:
                        print(f"Max retries reached. Last error: {str(e)}")
                        raise
                    print(f"Attempt {tries} failed. Retrying in {delay} seconds...")
                    time.sleep(delay)
        return wrapper
    return decorator


@retry(max_tries=5, delay=1)
def ssh_connect(hostname=None, username="root", 
    key_filename=None, password=None,
    ):
    # TODO: 配置环境变量或使用配置文件
    if hostname is None:
        hostname = os.getenv("SSH_HOSTNAME", "localhost")
    
    if "amazonaws" in hostname:
        if key_filename is None:
            key_filename = os.path.expanduser(os.getenv("SSH_KEY_PATH", "~/.ssh/webarena_key.pem"))
    else:
        if password is None:
            password = os.getenv("SSH_PASSWORD", "")  # 已清理敏感信息，请配置环境变量
    """
    通过SSH连接到服务器，执行一个可能长时间运行的命令。
    本函数会无限期等待命令执行完毕，以确保获取完整的输出。
    """
    print("--- Starting SSH connection and command execution (no timeout) ---")

    """
    通过SSH连接到服务器，执行一个可能长时间运行的命令。
    本函数启用了SSH Keep-Alive来防止因空闲而连接中断。
    """
    ssh = None
    print("--- Starting SSH connection with Keep-Alive enabled ---")

    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        print("Attempting to create proxy socket...")
        sock = create_proxy_sock(hostname, 22)
        #sock = None

        connect_kwargs = {
            'hostname': hostname, 'username': username, 'timeout': 15
        }
        if sock:
            connect_kwargs['sock'] = sock
            print("Connecting via proxy...")
        else:
            print("Connecting directly (no proxy)...")

        if key_filename:
            connect_kwargs['key_filename'] = key_filename
        elif password:
            connect_kwargs['password'] = password
        else:
            raise ValueError("Authentication required: either key_filename or password must be provided.")
        
        ssh.connect(**connect_kwargs)
        logger.info(f"✅ Successfully connected to {hostname}")

        # --- [关键修改] ---
        # 启用 SSH Keep-Alive
        # 每 60 秒发送一个 keep-alive 包。这个时间应该小于任何可能存在的网络超时。
        # 如果 60 秒仍然被断开，可以尝试更短的时间，例如 30 秒。
        transport = ssh.get_transport()
        if transport and transport.is_active():
            print("Enabling SSH keep-alive (sending packet every 60s)...")
            transport.set_keepalive(30)
        # --------------------
    except Exception as e:
        logger.error(f"❌ FATAL ERROR during SSH connection: {hostname} {str(e)}")
        ssh = None
    return ssh

@retry(max_tries=5, delay=1)
async def ssh_connect_and_refreshweb(hostname="REPLACE_WITH_YOUR_HOST", username="root", 
    key_filename=None, password=None, webarena_auth_path=WEBARENA_AUTH_PATH, owner_actor=None
    ):
    if hostname is None:
        logger.error(f"❌ FATAL ERROR during SSH connection: Hostname is None")
        return (hostname, False, "Hostname is None")
    if True:
        #非debug 模式才会刷新
        #debug 不执行刷新web操作
        try:
            ssh = ssh_connect(hostname, username, key_filename, password)
            commands = [
                "cd /home/ubuntu && bash refersh_webs_keep_cookies_autoip.sh",
            ]
            
            for command in commands:
                logger.info(f"\n▶️ Executing command and waiting for it to complete: '{command}'")
                stdin, stdout, stderr = ssh.exec_command(command)
                start_time = time.time()
                exit_status = stdout.channel.recv_exit_status()
                end_time = time.time()
                logger.info(f"Command completed in {end_time - start_time} seconds")
                error = stderr.read().decode('utf-8').strip()

                # if error:
                #     logger.error("--- [STDERR] ---")
                #     logger.error(error)
                #     logger.error("------------------")

                if exit_status != 0:
                    logger.warning(f"⚠️ Warning: Command exited with a non-zero status ({exit_status}), indicating a possible error.")

        except Exception as e:
            logger.error(f"❌ FATAL ERROR during connection or execution: {str(e)}")
            return (hostname, False, f"❌ FATAL ERROR during connection or execution: {str(e)}")
        finally:
            if ssh and ssh.get_transport() and ssh.get_transport().is_active():
                logger.info("\nClosing SSH connection.")
                ssh.close()
        logger.info("--- SSH task finished, waiting for 60 seconds... ---")
        
        await asyncio.sleep(4 * 60)

    # 2. 将异步任务委托给 Actor
    try:
        start_time = time.time()
        
        # 使用 actor 的 submit 方法来调用 async_generate_new_cookies
        # 这会返回一个 concurrent.futures.Future
        await async_generate_new_cookies( 
            auth_folder=f"{webarena_auth_path}/.auth", 
            REPLACE_WITH_YOUR_HOST=hostname, 
            owner_actor=owner_actor
        )
        
        end_time = time.time()
        duration_msg = f"Cookie generation completed in {end_time - start_time} seconds"
        logger.info(duration_msg)
        return (hostname, True, duration_msg)
        
    except Exception as e:
        # submit 内部的异常会在 .result() 时被重新抛出
        error_msg = f"❌ FATAL ERROR during cookie generation: {e}"
        logger.error(error_msg, exc_info=True)
        return (hostname, False, str(e)) # 返回原始异常信息

@retry(max_tries=5, delay=1)
async def ssh_connect_and_refreshshopping(hostname="REPLACE_WITH_YOUR_HOST", username="root", 
    key_filename=None, password=None, webarena_auth_path=WEBARENA_AUTH_PATH, owner_actor=None
    ):
    if hostname is None:
        logger.error(f"❌ FATAL ERROR during SSH connection: Hostname is None")
        return (hostname, False, "Hostname is None")
        
    script_content = r"""#!/bin/bash
# ==============================================================================
#  Refresh Shopping and Shopping Admin Only
# ==============================================================================

ONE_YEAR_IN_SECONDS=31536000

echo "--- 正在自动获取公网 IP ---"
PUBLIC_IP=$(curl -s ifconfig.me)

if [[ -z "$PUBLIC_IP" || ! "$PUBLIC_IP" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "错误：无法获取有效的公网 IP 地址。"
    exit 1
fi

echo "成功获取公网 IP: ${PUBLIC_IP}"
YOUR_ACTUAL_HOSTNAME="http://${PUBLIC_IP}"
YOUR_ACTUAL_HOSTNAME=${YOUR_ACTUAL_HOSTNAME%/}

echo "将使用 ${YOUR_ACTUAL_HOSTNAME} 作为所有服务的基础 URL。"

# -- 清理环境 --
echo "--- 正在停止并移除 Shopping 容器 ---"
docker stop shopping || true
docker stop shopping_admin || true
docker rm shopping || true
docker rm shopping_admin || true

# -- 启动容器 --
echo "--- 正在启动 Shopping 容器 ---"
docker run --name shopping -p 7770:80 -d shopping_final_0712
docker run --name shopping_admin -p 7780:80 -d shopping_admin_final_0719

echo "正在等待 MySQL 服务就绪..."
# 循环检查直到 MySQL 可以连接（最多等待 180秒）
MAX_RETRIES=36
COUNT=0
until docker exec shopping mysql -u magentouser -pMyPassword -e "SELECT 1" > /dev/null 2>&1; do
    echo "MySQL 尚未就绪，等待 5 秒... ($COUNT/$MAX_RETRIES)"
    sleep 5
    COUNT=$((COUNT+1))
    if [ $COUNT -ge $MAX_RETRIES ]; then
        echo "错误：等待 MySQL 启动超时。"
        exit 1
    fi
done
echo "MySQL 服务已就绪。"

# -- 配置 Shopping (OSS) --
echo "--- 正在配置 Shopping (OSS) ---"
docker exec shopping /var/www/magento2/bin/magento setup:store-config:set --base-url="${YOUR_ACTUAL_HOSTNAME}:7770"
docker exec shopping mysql -u magentouser -pMyPassword magentodb -e  "UPDATE core_config_data SET value='${YOUR_ACTUAL_HOSTNAME}:7770/' WHERE path = 'web/secure/base_url';"

echo "正在为 Shopping 延长 cookie 有效期..."
docker exec shopping php /var/www/magento2/bin/magento config:set web/cookie/cookie_lifetime ${ONE_YEAR_IN_SECONDS}
docker exec shopping php /var/www/magento2/bin/magento config:set admin/security/session_lifetime ${ONE_YEAR_IN_SECONDS}
docker exec shopping sh -c "echo 'session.cookie_lifetime = ${ONE_YEAR_IN_SECONDS}' > /usr/local/etc/php/conf.d/zz-custom-session.ini"
docker exec shopping sh -c "echo 'session.gc_maxlifetime = ${ONE_YEAR_IN_SECONDS}' >> /usr/local/etc/php/conf.d/zz-custom-session.ini"
docker exec shopping pkill -HUP php-fpm
docker exec shopping /var/www/magento2/bin/magento cache:flush

echo "正在为 Shopping 禁用产品重新索引..."
docker exec shopping /var/www/magento2/bin/magento indexer:set-mode schedule catalogrule_product
docker exec shopping /var/www/magento2/bin/magento indexer:set-mode schedule catalogrule_rule
docker exec shopping /var/www/magento2/bin/magento indexer:set-mode schedule catalogsearch_fulltext
docker exec shopping /var/www/magento2/bin/magento indexer:set-mode schedule catalog_category_product
docker exec shopping /var/www/magento2/bin/magento indexer:set-mode schedule customer_grid
docker exec shopping /var/www/magento2/bin/magento indexer:set-mode schedule design_config_grid
docker exec shopping /var/www/magento2/bin/magento indexer:set-mode schedule inventory
docker exec shopping /var/www/magento2/bin/magento indexer:set-mode schedule catalog_product_category
docker exec shopping /var/www/magento2/bin/magento indexer:set-mode schedule catalog_product_attribute
docker exec shopping /var/www/magento2/bin/magento indexer:set-mode schedule catalog_product_price
docker exec shopping /var/www/magento2/bin/magento indexer:set-mode schedule cataloginventory_stock

# -- 配置 Shopping Admin (CMS) --
echo "--- 正在配置 Shopping Admin (CMS) ---"
docker exec shopping_admin /var/www/magento2/bin/magento setup:store-config:set --base-url="${YOUR_ACTUAL_HOSTNAME}:7780"
docker exec shopping_admin mysql -u magentouser -pMyPassword magentodb -e  "UPDATE core_config_data SET value='${YOUR_ACTUAL_HOSTNAME}:7780/' WHERE path = 'web/secure/base_url';"
docker exec shopping_admin php /var/www/magento2/bin/magento config:set admin/security/password_is_forced 0
docker exec shopping_admin php /var/www/magento2/bin/magento config:set admin/security/password_lifetime 0

echo "正在为 Shopping Admin 延长 cookie 有效期..."
docker exec shopping_admin php /var/www/magento2/bin/magento config:set web/cookie/cookie_lifetime ${ONE_YEAR_IN_SECONDS}
docker exec shopping_admin php /var/www/magento2/bin/magento config:set admin/security/session_lifetime ${ONE_YEAR_IN_SECONDS}
docker exec shopping_admin sh -c "echo 'session.cookie_lifetime = ${ONE_YEAR_IN_SECONDS}' > /usr/local/etc/php/conf.d/zz-custom-session.ini"
docker exec shopping_admin sh -c "echo 'session.gc_maxlifetime = ${ONE_YEAR_IN_SECONDS}' >> /usr/local/etc/php/conf.d/zz-custom-session.ini"
docker exec shopping_admin pkill -HUP php-fpm
docker exec shopping_admin /var/www/magento2/bin/magento cache:flush

echo "Shopping refresh completed."
"""

    if VLM_EXP_DEBUG == '0':
        try:
            ssh = ssh_connect(hostname, username, key_filename, password)
            
            # Create the script on the remote server
            create_script_cmd = f"cat << 'EOF' > refresh_shopping_only.sh\n{script_content}\nEOF"
            logger.info(f"Creating refresh_shopping_only.sh on {hostname}")
            stdin, stdout, stderr = ssh.exec_command(create_script_cmd)
            exit_status = stdout.channel.recv_exit_status()
            if exit_status != 0:
                logger.error(f"Failed to create script: {stderr.read().decode('utf-8')}")
                raise Exception("Failed to create remote script")
            
            # Execute the script
            logger.info(f"Executing refresh_shopping_only.sh on {hostname}")
            stdin, stdout, stderr = ssh.exec_command("bash refresh_shopping_only.sh")
            
            # Stream output
            while not stdout.channel.exit_status_ready():
                 if stdout.channel.recv_ready():
                     line = stdout.channel.recv(1024).decode('utf-8')
                     # print(line, end="") # Optional: print to stdout
            
            exit_status = stdout.channel.recv_exit_status()
            error = stderr.read().decode('utf-8').strip()
            
            if error:
                logger.error(f"--- [STDERR] ---\n{error}\n------------------")
            
            if exit_status != 0:
                logger.warning(f"⚠️ Warning: Command exited with a non-zero status ({exit_status}).")

        except Exception as e:
            logger.error(f"❌ FATAL ERROR during connection or execution: {str(e)}")
            return (hostname, False, f"❌ FATAL ERROR during connection or execution: {str(e)}")
        finally:
            if ssh and ssh.get_transport() and ssh.get_transport().is_active():
                logger.info("\nClosing SSH connection.")
                ssh.close()
        
        logger.info("--- SSH task finished, waiting for 60 seconds... ---")
        await asyncio.sleep(60)

    # Cookie generation for Shopping sites only
    try:
        start_time = time.time()
        
        # Combinations relevant to Shopping
        shopping_combos = [
            ["shopping"], 
            ["shopping_admin"],
            ["shopping", "shopping_admin"],
            ["reddit"],
        ]
        
        tasks = []
        for combo in shopping_combos:
            logger.info(f"Queueing cookie generation for: {combo}")
            tasks.append(
                asyncio.create_task(
                    async_renew_comb(
                        combo, 
                        auth_folder=f"{webarena_auth_path}/.auth", 
                        REPLACE_WITH_YOUR_HOST=hostname, 
                        owner_actor=owner_actor
                    )
                )
            )
        
        await asyncio.gather(*tasks)
        
        end_time = time.time()
        duration_msg = f"Shopping cookie generation completed in {end_time - start_time} seconds"
        logger.info(duration_msg)
        return (hostname, True, duration_msg)
        
    except Exception as e:
        error_msg = f"❌ FATAL ERROR during cookie generation: {e}"
        logger.error(error_msg, exc_info=True)
        return (hostname, False, str(e))


def _worker_refresh_sites(hostname: str, sites_to_refresh: List[str]):
    """
    (Worker函数) 在单个ECS上刷新一组特定的网站。
    """
    sites_set = set(sites_to_refresh)
    
    sorted_sites = sorted(list(sites_set))
    logger.info(f"Host {hostname}: Starting refresh for combo: {sorted_sites}")

    public_ip = hostname
    your_actual_hostname = f"http://{public_ip}"
    commands = []
    # 清理和启动容器
    for site in sites_to_refresh:
        commands.append(f"docker stop {site} || true && docker rm {site} || true")
    
    for site in sites_to_refresh:
        if site == "shopping":
            commands.append(f"docker run --name shopping -p 7770:80 -d shopping_final_0712")
        elif site == "shopping_admin":
            commands.append(f"docker run --name shopping_admin -p 7780:80 -d shopping_admin_final_0719")
        elif site == "forum":
            commands.append(f"docker run --name forum -p 9999:80 -d postmill-populated-exposed-withimg")
        elif site == "gitlab":
            commands.append(f"docker run --name gitlab --shm-size='10g' -d -p 8023:8023 gitlab-populated-final-port8023 /opt/gitlab/embedded/bin/runsvdir-start")

    # 等待容器初始化
    sleep_time = 60 if "gitlab" in sites_to_refresh else 30
    commands.append(f"echo 'Waiting {sleep_time}s for containers...' && sleep {sleep_time}")

    # --- 配置命令生成 ---
    if "shopping" in sites_to_refresh:
        commands.extend([
            f"docker exec shopping /var/www/magento2/bin/magento setup:store-config:set --base-url='{your_actual_hostname}:7770'",
            f"docker exec shopping mysql -u magentouser -pMyPassword magentodb -e \"UPDATE core_config_data SET value='{your_actual_hostname}:7770/' WHERE path = 'web/secure/base_url';\"",
            f"docker exec shopping php /var/www/magento2/bin/magento config:set web/cookie/cookie_lifetime {ONE_YEAR_IN_SECONDS}",
            f"docker exec shopping php /var/www/magento2/bin/magento config:set admin/security/session_lifetime {ONE_YEAR_IN_SECONDS}",
            f"docker exec shopping sh -c \"echo 'session.cookie_lifetime = {ONE_YEAR_IN_SECONDS}' > /usr/local/etc/php/conf.d/zz-custom-session.ini\"",
            f"docker exec shopping sh -c \"echo 'session.gc_maxlifetime = {ONE_YEAR_IN_SECONDS}' >> /usr/local/etc/php/conf.d/zz-custom-session.ini\"",
            "docker exec shopping pkill -HUP php-fpm",
            "docker exec shopping /var/www/magento2/bin/magento cache:flush",
            "docker exec shopping /var/www/magento2/bin/magento indexer:set-mode schedule catalogrule_product",
            "docker exec shopping /var/www/magento2/bin/magento indexer:set-mode schedule catalogrule_rule",
            "docker exec shopping /var/www/magento2/bin/magento indexer:set-mode schedule catalogsearch_fulltext",
            "docker exec shopping /var/www/magento2/bin/magento indexer:set-mode schedule catalog_category_product",
            "docker exec shopping /var/www/magento2/bin/magento indexer:set-mode schedule customer_grid",
            "docker exec shopping /var/www/magento2/bin/magento indexer:set-mode schedule design_config_grid",
            "docker exec shopping /var/www/magento2/bin/magento indexer:set-mode schedule inventory",
            "docker exec shopping /var/www/magento2/bin/magento indexer:set-mode schedule catalog_product_category",
            "docker exec shopping /var/www/magento2/bin/magento indexer:set-mode schedule catalog_product_attribute",
            "docker exec shopping /var/www/magento2/bin/magento indexer:set-mode schedule catalog_product_price",
            "docker exec shopping /var/www/magento2/bin/magento indexer:set-mode schedule cataloginventory_stock"
        ])
    
    if "shopping_admin" in sites_to_refresh:
        commands.extend([
            f"docker exec shopping_admin /var/www/magento2/bin/magento setup:store-config:set --base-url='{your_actual_hostname}:7780'",
            f"docker exec shopping_admin mysql -u magentouser -pMyPassword magentodb -e \"UPDATE core_config_data SET value='{your_actual_hostname}:7780/' WHERE path = 'web/secure/base_url';\"",
            "docker exec shopping_admin php /var/www/magento2/bin/magento config:set admin/security/password_is_forced 0",
            "docker exec shopping_admin php /var/www/magento2/bin/magento config:set admin/security/password_lifetime 0",
            f"docker exec shopping_admin php /var/www/magento2/bin/magento config:set web/cookie/cookie_lifetime {ONE_YEAR_IN_SECONDS}",
            f"docker exec shopping_admin php /var/www/magento2/bin/magento config:set admin/security/session_lifetime {ONE_YEAR_IN_SECONDS}",
            f"docker exec shopping_admin sh -c \"echo 'session.cookie_lifetime = {ONE_YEAR_IN_SECONDS}' > /usr/local/etc/php/conf.d/zz-custom-session.ini\"",
            f"docker exec shopping_admin sh -c \"echo 'session.gc_maxlifetime = {ONE_YEAR_IN_SECONDS}' >> /usr/local/etc/php/conf.d/zz-custom-session.ini\"",
            "docker exec shopping_admin pkill -HUP php-fpm",
            "docker exec shopping_admin /var/www/magento2/bin/magento cache:flush"
        ])
    
    if "reddit" in sites_to_refresh:
        commands.extend([
            "docker exec forum sed -i '/@RateLimit/,/)/d' /var/www/html/src/DataObject/CommentData.php",
            "docker exec forum sed -i '/@RateLimit/,/)/d' /var/www/html/src/DataObject/SubmissionData.php",
            "docker exec forum sed -i '/@RateLimit/,/)/d' /var/www/html/src/DataObject/UserData.php",
            f"docker exec forum sed -i \"/session:/a \\        cookie_lifetime: {ONE_YEAR_IN_SECONDS}\" /var/www/html/config/packages/framework.yaml",
            f"docker exec forum sh -c \"echo 'session.cookie_lifetime = {ONE_YEAR_IN_SECONDS}' > /usr/local/etc/php/conf.d/zz-custom-session.ini\"",
            f"docker exec forum sh -c \"echo 'session.gc_maxlifetime = {ONE_YEAR_IN_SECONDS}' >> /usr/local/etc/php/conf.d/zz-custom-session.ini\"",
            "docker exec forum pkill -HUP php-fpm",
            "docker exec forum bin/console cache:clear --env=prod"
        ])
        
    if "gitlab" in sites_to_refresh:
        commands.extend([
            "echo 'Waiting extra for GitLab before reconfigure...' && sleep 60",
            f"docker exec gitlab sed -i \"s|^external_url.*|external_url '{your_actual_hostname}:8023'|\" /etc/gitlab/gitlab.rb",
            "docker exec gitlab sed -i \"s/.*postgresql\\['max_connections'.*/postgresql\\['max_connections'\\] = 2000/g\" /etc/gitlab/gitlab.rb",
            f"docker exec gitlab sh -c \"echo \\\"gitlab_rails['session_expire_delay'] = {ONE_YEAR_IN_MINUTES}\\\" >> /etc/gitlab/gitlab.rb\"",
            "docker exec gitlab gitlab-ctl reconfigure",
            "docker exec gitlab gitlab-ctl restart"
        ])

    # 3. SSH连接并执行命令
    ssh = None
    try:
        ssh = ssh_connect(hostname)
        
        for command in commands:
            logger.info(f"Host {hostname}: Executing command: '{command[:120]}...'")
            stdin, stdout, stderr = ssh.exec_command(command) # 15分钟超时以应对gitlab reconfigure
            
            exit_status = stdout.channel.recv_exit_status()
            if exit_status != 0:
                error_output = stderr.read().decode('utf-8', errors='ignore').strip()
                logger.warning(f"Host {hostname}: Command failed with status {exit_status}. CMD: '{command[:120]}...'. Stderr: {error_output}")
        return hostname, sorted_sites, True, "Refresh successful"
    except Exception as e:
        logger.error(f"❌ FATAL ERROR during SSH execution on {hostname}: {e}")
        return (hostname, False, f"SSH execution error: {e}")
    finally:
        if ssh and ssh.get_transport() and ssh.get_transport().is_active():
            ssh.close()


def _worker_generate_cookies(hostname: str, sites_to_gen: List[str], auth_path: str):
    """
    (Worker函数) 在已刷新的ECS上为特定网站组合生成Cookie。
    """
    logger.info(f"Host {hostname}: Starting cookie generation for combo: {sites_to_gen}")
    try:
        # GitLab需要更长的启动时间
        sleep_time = 60 if "gitlab" in sites_to_gen else 30
        logger.info(f"Host {hostname}: Waiting {sleep_time}s before cookie generation...")
        time.sleep(sleep_time)

        asyncio.run(async_renew_comb(sites_to_gen, auth_folder=auth_path, REPLACE_WITH_YOUR_HOST=hostname))
        return hostname, sites_to_gen, True, "Cookie generation successful"
    except Exception as e:
        logger.error(f"❌ FATAL ERROR during cookie generation for {hostname}, combo {sites_to_gen}: {e}")
        return hostname, sites_to_gen, False, str(e)

def _worker_verify_cookies(hostname: str, auth_path: str):
    """
    (Worker函数) 在指定路径验证所有生成的Cookie。
    """
    logger.info(f"Host {hostname}: Starting cookie verification in path: {auth_path}")
    try:
        # async_verif_web 已经包含了打印详细日志的逻辑
        asyncio.run(async_verif_web(auth_folder=auth_path, REPLACE_WITH_YOUR_HOST=hostname))
        return hostname, None, True, "Verification finished"
    except Exception as e:
        logger.error(f"❌ FATAL ERROR during cookie verification for {hostname}: {e}")
        return hostname, None, False, str(e)

@retry(max_tries=3, delay=5)
def ssh_connect_and_refreshweb_single_web(
    hostname: str, 
    start_urls: str, 
    webarena_auth_path=WEBARENA_AUTH_PATH,
    username="root", 
    key_filename=None, 
    password=None
) -> Tuple[str, bool, str]:

    """
    通过SSH连接，根据start_urls智能地只刷新和验证相关的网站，并执行所有详细配置。
    """
    logger.info(f"--- Starting single-web refresh for host {hostname} based on URLs ---")
    
    # 1. 解析URL确定站点
    sites_to_refresh: Set[str] = set()
    urls = [url.strip() for url in start_urls.split("|AND|") ]
    for url in urls:
        for port, site_name in SITE_PORT_MAP.items():
            if f":{port}" in url:
                sites_to_refresh.add(site_name)
    
    if "shopping" in sites_to_refresh or "shopping_admin" in sites_to_refresh:
        sites_to_refresh.update(["shopping", "shopping_admin"])

    if not sites_to_refresh:
        logger.warning(f"Host {hostname}: No known site in '{start_urls}'. No refresh needed.")
        return (hostname, True, "No known site to refresh, success.")

    logger.info(f"Host {hostname}: Identified sites to refresh: {sorted(list(sites_to_refresh))}")

    # 2. 动态生成命令列表
    public_ip = hostname
    your_actual_hostname = f"http://{public_ip}"
    
    commands = []
    


    # 4. 生成和验证相关网站的Cookie
    # 为GitLab留出更多重启时间
    final_sleep = 60 if "gitlab" in sites_to_refresh else 30
    logger.info(f"Host {hostname}: Refresh done, waiting {final_sleep}s before cookie generation...")
    time.sleep(final_sleep)
    
    try:
        start_time = time.time()
        # 将set转换为list给async_renew_comb
        sites_list = list(sites_to_refresh)
        asyncio.run(async_renew_comb(sites_list, auth_folder=webarena_auth_path, REPLACE_WITH_YOUR_HOST=hostname))
        
        # 验证生成的cookie
        url_list = [URLS[SITES.index(site)] for site in sites_to_refresh]
        asyncio.run(async_verif_web(auth_folder=webarena_auth_path, REPLACE_WITH_YOUR_HOST=hostname, url_list=url_list))

        end_time = time.time()
        msg = f"Cookie generation for {sites_list} completed in {end_time - start_time:.2f} seconds"
        logger.info(f"Host {hostname}: {msg}")
        return (hostname, True, msg)
    except Exception as e:
        logger.error(f"❌ FATAL ERROR during cookie generation for {hostname}: {e}")
        return (hostname, False, f"Cookie generation error: {e}")


# ================================================================
#             并发测试功能代码 (START)
# ================================================================

def get_REPLACE_WITH_YOUR_HOSTs_from_file(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return []
    ips = []
    except_ips = [] #['REPLACE_WITH_YOUR_HOST']
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            f.seek(0)
            if f.read(1) == '\ufeff':
                f.seek(1)
            else:
                f.seek(0)
            reader = csv.reader(f)
            header = next(reader)
            try:
                cleaned_header = [h.strip() for h in header]
                ip_column_index = cleaned_header.index("公网 IP")
            except ValueError:
                print(f"Warning: Column '公网 IP' not found. Using column 2 (index 1) as fallback.")
                ip_column_index = 1
            
            for row in reader:
                if len(row) > ip_column_index:
                    ip = row[ip_column_index].strip()
                    if ip:
                        ips.append(ip)
    except Exception as e:
        print(f"Error reading or parsing CSV file '{file_path}': {e}")
    return [ip for ip in ips if ip not in except_ips]

def test_single_host(ip):
    """
    测试单个主机的函数，返回一个元组 (ip, is_successful, message)。
    这个函数将被并发调用。
    """
    try:
        # ssh_connect 自带重试机制
        ssh_client = ssh_connect(hostname=ip, username="root")
        if ssh_client and ssh_client.get_transport() and ssh_client.get_transport().is_active():
            ssh_client.close()
            return (ip, True, "Connection successful")
        else:
            # 理论上不会到这里，因为失败会抛异常
            return (ip, False, "Connection failed (unknown reason)")
    except Exception as e:
        # 捕获 ssh_connect 最终抛出的异常
        error_message = str(e).replace('\n', ' ') # 将错误信息压缩到一行
        return (ip, False, f"Connection failed: {error_message}")

def test_all_ecs_connectivity_concurrently(file_path, max_workers=10):
    """
    主函数：并发测试ECS列表的SSH连接性。
    max_workers: 并发线程数，可以根据你的网络和CPU情况调整。
    """
    REPLACE_WITH_YOUR_HOSTs = get_REPLACE_WITH_YOUR_HOSTs_from_file(file_path)
    if not REPLACE_WITH_YOUR_HOSTs:
        print("No IPs to test. Exiting.")
        return

    print(f"\n--- Starting Concurrent SSH Test for {len(REPLACE_WITH_YOUR_HOSTs)} hosts (max_workers={max_workers}) from '{os.path.basename(file_path)}' ---")
    
    successful_hosts = []
    failed_hosts = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_ip = {executor.submit(test_single_host, ip): ip for ip in REPLACE_WITH_YOUR_HOSTs}
        
        # 实时获取已完成任务的结果
        for future in as_completed(future_to_ip):
            ip = future_to_ip[future]
            try:
                ip_result, is_successful, message = future.result()
                if is_successful:
                    print(f"✅ [SUCCESS] {ip_result}")
                    successful_hosts.append(ip_result)
                else:
                    print(f"❌ [FAILURE] {ip_result} - {message}")
                    failed_hosts.append((ip_result, message))
            except Exception as exc:
                # 捕获 test_single_host 本身可能出现的意外错误
                print(f"❌ [ERROR]   {ip} generated an exception: {exc}")
                failed_hosts.append((ip, str(exc)))

    print("\n" + "="*50)
    print("           CONCURRENT SSH TEST SUMMARY")
    print("="*50)
    print(f"Total hosts tested: {len(REPLACE_WITH_YOUR_HOSTs)}")
    print(f"✅ Successful connections: {len(successful_hosts)}")
    print(f"❌ Failed connections: {len(failed_hosts)}")
    
    if failed_hosts:
        print("\n--- Details of Failed Hosts ---")
        for host, reason in failed_hosts:
            print(f"- Host: {host}\n  Reason: {reason}")
    print("="*50)

def test_all_ecs_connectivity_concurrently_refreshweb(file_path, max_workers=10):
    """
    主函数：并发测试ECS列表的SSH连接性。
    max_workers: 并发线程数，可以根据你的网络和CPU情况调整。
    """
    REPLACE_WITH_YOUR_HOSTs = get_REPLACE_WITH_YOUR_HOSTs_from_file(file_path)
    if not REPLACE_WITH_YOUR_HOSTs:
        print("No IPs to test. Exiting.")
        return

    print(f"\n--- Starting Concurrent SSH Test for {len(REPLACE_WITH_YOUR_HOSTs)} hosts (max_workers={max_workers}) from '{os.path.basename(file_path)}' ---")
    
    successful_hosts = []
    failed_hosts = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_ip = {executor.submit(ssh_connect_and_refreshweb, ip, webarena_auth_path=WEBARENA_AUTH_PATH+f"_{ip}"): ip for ip in REPLACE_WITH_YOUR_HOSTs}
        
        # 实时获取已完成任务的结果
        for future in as_completed(future_to_ip):
            ip = future_to_ip[future]
            try:
                ip_result, is_successful, message = future.result()
                if is_successful:
                    print(f"✅ [SUCCESS] {ip_result}")
                    successful_hosts.append(ip_result)
                else:
                    print(f"❌ [FAILURE] {ip_result} - {message}")
                    failed_hosts.append((ip_result, message))
            except Exception as exc:
                # 捕获 test_single_host 本身可能出现的意外错误
                print(f"❌ [ERROR]   {ip} generated an exception: {exc}")
                failed_hosts.append((ip, str(exc)))

    print("\n" + "="*50)
    print("           CONCURRENT SSH TEST SUMMARY")
    print("="*50)
    print(f"Total hosts tested: {len(REPLACE_WITH_YOUR_HOSTs)}")
    print(f"✅ Successful connections: {len(successful_hosts)}")
    print(f"❌ Failed connections: {len(failed_hosts)}")
    
    if failed_hosts:
        print("\n--- Details of Failed Hosts ---")
        for host, reason in failed_hosts:
            print(f"- Host: {host}\n  Reason: {reason}")
    print("="*50)

def get_all_valid_site_single(input_site) -> List[List[str]]:
    combs_to_generate = []
    output_site = []
    pairs = list(combinations(SITES, 2))
    for pair in pairs:
        if "reddit" in pair and ("shopping" in pair or "shopping_admin" in pair):
            continue
        combs_to_generate.append(list(sorted(pair)))
    for combs_i in combs_to_generate:
        if input_site in combs_i:
            for combs_j in combs_i:
                output_site.append(combs_j)
    return set(output_site)


def get_all_valid_site_combinations() -> List[List[str]]:
    combs_to_generate = []
    pairs = list(combinations(SITES, 2))
    for pair in pairs:
        if "reddit" in pair and ("shopping" in pair or "shopping_admin" in pair):
            continue
        combs_to_generate.append(list(sorted(pair)))
    
    for site in SITES:
        combs_to_generate.append([site])
    # 去重
    unique_combs = []
    for comb in combs_to_generate:
        if comb not in unique_combs:
            unique_combs.append(comb)
    return unique_combs

def test_and_prepare_all_ecs_in_stages(file_path: str, max_workers: int = 10):
    """
    主函数：按阶段并发地准备所有ECS上的所有网站组合。
    """
    REPLACE_WITH_YOUR_HOSTs = get_REPLACE_WITH_YOUR_HOSTs_from_file(file_path)
    if not REPLACE_WITH_YOUR_HOSTs:
        logger.error("No IPs found. Exiting.")
        return

    site_combos = SITES
    # --- 构建任务列表 ---
    all_tasks = []
    for ip in REPLACE_WITH_YOUR_HOSTs:
        for combo in site_combos:
            all_tasks.append({
                "ip": ip,
                "combo": [combo],
            })
    logger.info(f"Total tasks 1: {len(all_tasks)}")

    # --- 阶段一：并发刷新 ---
    logger.info("\n" + "="*20 + " STAGE 1: CONCURRENTLY REFRESHING WEBSITES " + "="*20)
    failed_in_stage1 = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(_worker_refresh_sites, task["ip"], task["combo"]): task 
            for task in all_tasks
        }
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                _, _, is_successful, message = future.result()
                if is_successful:
                    logger.info(f"✅ Refresh SUCCESS: Host {task['ip']}, Combo: {task['combo']}")
                else:
                    logger.error(f"❌ Refresh FAILURE: Host {task['ip']}, Combo: {task['combo']} - {message}")
                    failed_in_stage1.append(task)
            except Exception as e:
                logger.critical(f"💥 Refresh CRITICAL: Host {task['ip']}, Combo: {task['combo']} - {e}")
                failed_in_stage1.append(task)
    
    if failed_in_stage1:
        logger.error(f"{len(failed_in_stage1)} tasks failed in Stage 1. Aborting further stages for these tasks.")
    
    site_combos = get_all_valid_site_combinations()
    # --- 构建任务列表 ---
    all_tasks = []
    for ip in REPLACE_WITH_YOUR_HOSTs:
        for combo in site_combos:
            all_tasks.append({
                "ip": ip,
                "combo": combo,
            })
    logger.info(f"Total tasks 2: {len(all_tasks)}")

    # 只为刷新成功的任务进行下一阶段
    tasks_for_stage2 = [task for task in all_tasks if task not in failed_in_stage1]

    # --- 阶段二：并发生成Cookie ---
    logger.info("\n" + "="*20 + " STAGE 2: CONCURRENTLY GENERATING COOKIES " + "="*20)
    failed_in_stage2 = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(_worker_generate_cookies, task["ip"], task["combo"], WEBARENA_AUTH_PATH+f"_{task['ip']}"): task
            for task in tasks_for_stage2
        }
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                _, _, is_successful, message = future.result()
                if is_successful:
                    logger.info(f"✅ Cookie Gen SUCCESS: Host {task['ip']}, Combo: {task['combo']}")
                else:
                    logger.error(f"❌ Cookie Gen FAILURE: Host {task['ip']}, Combo: {task['combo']} - {message}")
                    failed_in_stage2.append(task)
            except Exception as e:
                logger.critical(f"💥 Cookie Gen CRITICAL: Host {task['ip']}, Combo: {task['combo']} - {e}")
                failed_in_stage2.append(task)

    if failed_in_stage2:
        logger.error(f"{len(failed_in_stage2)} tasks failed in Stage 2. Aborting verification for these tasks.")

    # 只为Cookie生成成功的任务进行验证
    tasks_for_stage3 = [task for task in tasks_for_stage2 if task not in failed_in_stage2]

    # --- 阶段三：并发验证Cookie ---
    logger.info("\n" + "="*20 + " STAGE 3: CONCURRENTLY VERIFYING COOKIES " + "="*20)
    # 验证是按路径来的，我们可以为每个ECS的所有组合路径只验证一次
    unique_auth_paths_by_ip = {ip: set() for ip in REPLACE_WITH_YOUR_HOSTs}
    for task in tasks_for_stage3:
        unique_auth_paths_by_ip[task['ip']].add(task['auth_path'])

    failed_in_stage3 = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        verification_tasks = []
        for ip, auth_paths in unique_auth_paths_by_ip.items():
            for auth_path in auth_paths:
                verification_tasks.append({"ip": ip, "auth_path": auth_path})
        
        future_to_task = {
            executor.submit(_worker_verify_cookies, task["ip"], task["auth_path"]): task
            for task in verification_tasks
        }
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                _, _, is_successful, message = future.result()
                if not is_successful:
                    logger.error(f"❌ Verification FAILURE: Host {task['ip']}, Path: {task['auth_path']} - {message}")
                    failed_in_stage3.append(task)
            except Exception as e:
                logger.critical(f"💥 Verification CRITICAL: Host {task['ip']}, Path: {task['auth_path']} - {e}")
                failed_in_stage3.append(task)

    # --- 最终总结 ---
    print("\n" + "=" * 60)
    print("                 MULTI-STAGE PREPARATION SUMMARY")
    print("=" * 60)
    print(f"Total tasks attempted: {len(all_tasks)}")
    print(f"Failed in Stage 1 (Refresh): {len(failed_in_stage1)}")
    print(f"Failed in Stage 2 (Cookie Gen): {len(failed_in_stage2)}")
    print(f"Failed in Stage 3 (Verification): {len(failed_in_stage3)}")
    total_success = len(all_tasks) - len(failed_in_stage1) - len(failed_in_stage2)
    # Verification failures don't count against task success, but are noted
    print(f"✅ Overall successful preparations: {total_success}")
    print("=" * 60)

# ================================================================
#             并发测试功能代码 (END)
# ================================================================

if __name__ == "__main__":


    # --- 运行并发测试函数 ---
    # 你可以调整 max_workers 的值，例如 20 或 50，取决于你的机器性能和网络状况
    #test_and_prepare_all_ecs_in_stages(full_file_path, max_workers=100)
    test_all_ecs_connectivity_concurrently_refreshweb(full_file_path, max_workers=20)
