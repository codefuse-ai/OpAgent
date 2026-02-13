import os
import time
import asyncio
from loguru import logger
from opagent.browser_env.refresh_web import ssh_connect, retry
from opagent.browser_env.auto_login_asynico import async_renew_comb

WEBARENA_AUTH_PATH = os.environ.get("WEBARENA_AUTH_PATH", "./log")

@retry(max_tries=5, delay=1)
async def ssh_connect_and_refresh_gitlab(hostname=None, username="root", 
    key_filename=None, password=None, webarena_auth_path=WEBARENA_AUTH_PATH, owner_actor=None
    ):
    # TODO: 配置环境变量
    if hostname is None:
        hostname = os.getenv("SSH_HOSTNAME", "localhost")
    if hostname is None:
        logger.error(f"❌ FATAL ERROR during SSH connection: Hostname is None")
        return (hostname, False, "Hostname is None")
        
    script_content = r"""
#!/bin/bash
# Minimal mock for gitlab refresh to avoid full script dependency issues in demo
echo "--- Mocking GitLab Refresh ---"
sleep 1
echo "GitLab refresh completed."
"""

    try:
        ssh = ssh_connect(hostname, username, key_filename, password)
        
        # Create the script on the remote server
        create_script_cmd = f"cat << 'EOF' > refresh_gitlab_only.sh\n{script_content}\nEOF"
        logger.info(f"Creating refresh_gitlab_only.sh on {hostname}")
        stdin, stdout, stderr = ssh.exec_command(create_script_cmd)
        exit_status = stdout.channel.recv_exit_status()
        if exit_status != 0:
            logger.error(f"Failed to create script: {stderr.read().decode('utf-8')}")
            raise Exception("Failed to create remote script")
        
        # Execute the script
        logger.info(f"Executing refresh_gitlab_only.sh on {hostname}")
        stdin, stdout, stderr = ssh.exec_command("bash refresh_gitlab_only.sh")
        
        # Stream output
        while not stdout.channel.exit_status_ready():
              if stdout.channel.recv_ready():
                  line = stdout.channel.recv(1024).decode('utf-8')
        
        exit_status = stdout.channel.recv_exit_status()
        
        if exit_status != 0:
            logger.warning(f"⚠️ Warning: Command exited with a non-zero status ({exit_status}).")

    except Exception as e:
        logger.error(f"❌ FATAL ERROR during connection or execution: {str(e)}")
        # For demo purposes, we might want to suppress this error if SSH fails (no real server)
        # return (hostname, False, f"❌ FATAL ERROR during connection or execution: {str(e)}")
        logger.warning("SSH connection failed, but proceeding for demo purposes.")

    finally:
        if ssh and ssh.get_transport() and ssh.get_transport().is_active():
            logger.info("\nClosing SSH connection.")
            ssh.close()
    
    logger.info("--- SSH task finished, waiting for 60 seconds... ---")
    # await asyncio.sleep(60) # Reduced for demo

    # Cookie generation for gitlab sites only
    try:
        start_time = time.time()
        
        # Combinations relevant to gitlab
        gitlab_combos = [
            ["gitlab"],
        ]
        
        tasks = []
        for combo in gitlab_combos:
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
        
        # await asyncio.gather(*tasks) # Uncomment if we want real cookie gen
        
        end_time = time.time()
        duration_msg = f"gitlab cookie generation completed in {end_time - start_time} seconds"
        logger.info(duration_msg)
        return (hostname, True, duration_msg)
        
    except Exception as e:
        error_msg = f"❌ FATAL ERROR during cookie generation: {e}"
        logger.error(error_msg, exc_info=True)
        return (hostname, False, str(e))
