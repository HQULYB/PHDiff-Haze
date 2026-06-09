# GitHub 上传与更新代码流程

本文档记录本项目上传到 GitHub 以及后续更新代码的常用流程。

## 1. 第一次上传项目

进入项目目录：

```bash
cd /data/users/gaoyin/2024_LYB/PhysHazeDiffusion
```

如果当前目录还没有初始化 Git：

```bash
git init
git branch -M main
```

## 2. 配置 .gitignore

建议不要上传模型权重、训练输出、日志和缓存文件。

项目根目录下的 `.gitignore` 建议包含：

```gitignore
__pycache__/
*.pyc
.DS_Store

weights/
experiment/
outputs/
logs/

*.pt
*.pth
*.ckpt
*.safetensors
```

## 3. 添加文件并提交

查看当前改动：

```bash
git status
```

添加所有文件：

```bash
git add .
```

提交：

```bash
git commit -m "Initial commit"
```

如果提示需要配置用户名和邮箱：

```bash
git config --global user.name "HQULYB"
git config --global user.email "你的邮箱"
```

然后重新提交。

## 4. 绑定 GitHub 仓库

当前仓库地址：

```bash
https://github.com/HQULYB/PHDiff-Haze.git
```

添加远程仓库：

```bash
git remote add origin https://github.com/HQULYB/PHDiff-Haze.git
```

如果提示 `remote origin already exists`，说明已经绑定过远程仓库，改用：

```bash
git remote set-url origin https://github.com/HQULYB/PHDiff-Haze.git
```

检查远程仓库：

```bash
git remote -v
```

## 5. 第一次推送到 GitHub

```bash
git branch -M main
git push -u origin main
```

如果 GitHub 要求登录，注意现在不能使用账号密码直接推送，需要使用 Personal Access Token。

## 6. 后续更新代码

每次修改代码后，按下面流程更新 GitHub：

```bash
cd /data/users/gaoyin/2024_LYB/PhysHazeDiffusion
git status
git add .
git commit -m "Update code"
git push
```

提交信息可以按实际内容改得更具体，例如：

```bash
git commit -m "Update inference script"
git commit -m "Fix training dataset loader"
git commit -m "Add physical haze inference guide"
```

## 7. 常用 Git 命令

查看当前状态：

```bash
git status
```

查看远程仓库：

```bash
git remote -v
```

查看最近 5 次提交：

```bash
git log --oneline --max-count=5
```

查看某个文件的改动：

```bash
git diff 文件名
```

撤销还没有 `git add` 的某个文件改动：

```bash
git restore 文件名
```

取消已经 `git add` 但还没 commit 的文件：

```bash
git restore --staged 文件名
```

## 8. 注意事项

不要把下面这些大文件或生成文件上传到普通 Git 仓库：

```text
weights/
experiment/
outputs/
logs/
*.pt
*.pth
*.ckpt
*.safetensors
```

模型权重、训练结果和大规模输出建议单独保存，例如服务器路径、网盘，或者使用 Git LFS 管理。
