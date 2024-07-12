#!/bin/bash

# 启动 ssh-agent 并添加密钥
eval "$(ssh-agent -s)"
ssh-add /home/scheng1/.ssh/wl4023

# 提示输入提交消息
read -p "Enter commit message: " commit_message

# 执行 git 命令
git add .
git commit -m "$commit_message"
git push origin main
