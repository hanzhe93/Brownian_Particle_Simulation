# python自带虚拟环境管理
管理Python虚拟环境是一种良好的实践，特别是当您在项目之间使用不同的依赖关系时。下面是使用`venv`（Python 3内置的虚拟环境工具）的基本步骤：

### 创建虚拟环境：

在项目文件夹中打开终端，并运行以下命令：

```bash
# 在当前目录下创建一个名为"venv"的虚拟环境
python3 -m venv .venv
```

如果您使用的是Python 3.3以上的版本，`venv`模块通常已经包含在Python中，无需额外安装。

### 激活虚拟环境：
安装好虚拟环境好以后，下次激活仍然使用下面的命令，注意要在创建虚拟环境的同一个路径下，通常情况是在 `~/` 目录下

在Linux/macOS上：

```bash
source .venv/bin/activate
```

在Windows上：

```bash
.venv\Scripts\activate
```

激活虚拟环境后，终端提示符的前缀将显示虚拟环境的名称。

### 安装依赖：

在激活的虚拟环境中，使用`pip`安装项目所需的依赖项：

```bash
pip3 install package_name
```

### 退出虚拟环境：

在虚拟环境中工作完成后，可以通过以下方式退出：

```bash
deactivate
```
### 管理安装好的虚拟环境
##### 使用 `pipenv`：

如果您使用 `pipenv` 作为虚拟环境管理器，可以运行以下命令查看所有虚拟环境：

```bash
# 注意需要在虚拟环境安装的路径下，通常是 ‘~/’
pipenv --venv
```

这将列出所有通过 `pipenv` 创建的虚拟环境。

##### 使用 `conda`：

如果您使用 `conda` 作为虚拟环境管理器，可以运行以下命令查看所有虚拟环境：

```bash
conda info --envs
```

这将列出所有通过 `conda` 创建的环境。

### 导出依赖项列表：

为了共享项目的依赖项，您可以导出依赖项列表到一个文本文件，通常命名为`requirements.txt`：

```bash
pip3 freeze > requirements.txt
```

### 在另一个环境中安装依赖项：

在新的环境中，可以使用以下命令安装依赖项：

```bash
pip3 install -r requirements.txt
```

这些步骤可以帮助您在不同项目中轻松管理虚拟环境和依赖项。请注意，虚拟环境是与项目关联的，每个项目都应该有自己的虚拟环境。
