# DeepSeek-V4-Flash 转换与运行指南

本指南针对 Windows PowerShell，假设官方 Hugging Face 快照已经下载完成，目标是
在约 15GB RAM、x86-64 AVX2、NVMe SSD 的普通笔记本上使用纯 CPU 运行
`deepseek-ai/DeepSeek-V4-Flash-0731`。

当前 DeepSeek-V4 集成是 CLI-only：使用 `coli run` 进行单次生成，不支持
`coli chat` 或 `coli serve`。运行时不需要 Python、PyTorch 或 safetensors；这些依赖
只用于转换权重。

## 1. 路径和磁盘空间

以下命令假设：

```powershell
$Root = "D:\models"
$Source = "$Root\DeepSeek-V4-Flash-0731-hf"
$Model = "$Root\DeepSeek-V4-Flash-0731-colibri"
```

`$Source` 是已经下载的官方快照，`$Model` 是转换后供 Colibri 使用的目录。两者应
放在 NVMe SSD 上；运行时只需要保留 `$Model`，但转换完成并验证之前不要删除
`$Source`。

官方快照约 167GB，转换结果约 160--170GB。转换期间源文件和结果同时存在，建议
至少准备约 350GB 可用空间。

先检查输入目录：

```powershell
Test-Path "$Source\config.json"
Test-Path "$Source\tokenizer.json"
(Get-ChildItem -LiteralPath $Source -Filter "model-*.safetensors" -File).Count
```

前两个命令应返回 `True`。分片数量应接近官方快照的数量；如果输入目录没有
`config.json`，说明路径指向了错误的目录。

## 2. 安装转换依赖

建议单独创建转换环境，不污染项目运行环境：

```powershell
$Venv = "$Root\v4-convert-venv"

py -m venv $Venv
& "$Venv\Scripts\python.exe" -m pip install --upgrade pip
& "$Venv\Scripts\python.exe" -m pip install torch numpy safetensors
```

如果系统没有 `py` 命令，可以将上面的 `py` 替换成 `python`。

## 3. 转换官方权重

在仓库根目录 `D:\allens\code\gitroot\colibri` 执行：

```powershell
& "$Venv\Scripts\python.exe" c\tools\convert_deepseek_v4.py `
  --indir $Source `
  --outdir $Model `
  --group 64 `
  --max-gb 170
```

转换器会逐个处理 safetensors 分片：

- routed experts 保留官方 MXFP4 E2M1 + UE8M0/g32 格式；
- 非 expert 矩阵转换为 group-int4；
- 输入/输出相关矩阵使用 row-int8；
- 写入 `colibri_v4_manifest.json` 和 `model.safetensors.index.json`；
- 默认不写入可选的 DSpark/MTP sidecar，以降低存储和运行时开销。

转换过程可以中断后重跑。已经完整写入的分片会被跳过，正在写入的分片使用临时
文件，避免留下半个 safetensors 文件。

转换完成后检查输出：

```powershell
Test-Path "$Model\config.json"
Test-Path "$Model\tokenizer.json"
Test-Path "$Model\colibri_v4_manifest.json"
Test-Path "$Model\model.safetensors.index.json"
Get-Content "$Model\colibri_v4_manifest.json"
```

四个 `Test-Path` 都应返回 `True`。

## 4. 编译纯 CPU 引擎

使用 MinGW-w64 在仓库根目录执行：

```powershell
mingw32-make.exe -C c deepseek_v4 ARCH=x86-64-v3
```

`x86-64-v3` 是包含 AVX2 的可移植基线。如果编译和运行始终在同一台电脑上，
也可以使用：

```powershell
mingw32-make.exe -C c deepseek_v4 ARCH=native
```

生成文件为：

```text
c\deepseek_v4.exe
```

该目标是纯 C CPU 引擎，不需要 CUDA、GPU 或 Python runtime。

## 5. 运行模型

推荐先生成 64 个 token 做冒烟测试：

```powershell
python c\coli run `
  --model $Model `
  --ctx 50000 `
  --ngen 64 `
  "请用一句话解释 RAM 和 VRAM 的区别。"
```

也可以设置环境变量后省略 `--model`：

```powershell
$env:COLI_MODEL = $Model
python c\coli run --ctx 50000 --ngen 64 "请解释什么是 SSD streaming。"
```

程序结束时会在 stderr 输出类似：

```text
[DSV4] generated 64 token(s) in ...s (... tok/s)
```

这条消息中的 `tok/s` 是生成阶段速度。首次运行还需要加载 dense 层、建立 KV
缓存，启动时间会明显长于后续单个 token 的生成时间。

当前实现默认使用 greedy argmax，以便固定 prompt 的结果可以重复比较；每次运行
都是独立的单次生成。

## 6. 绕过 Python launcher 进行底层测试

如果需要直接测试 C 引擎，可以执行：

```powershell
$env:SNAP = $Model
$env:PROMPT = "<｜User｜>请用一句话解释 RAM 和 VRAM 的区别。<｜Assistant｜>"
$env:CTX = "50000"
$env:NGEN = "64"
& .\c\deepseek_v4.exe 0
```

正常使用仍建议使用 `python c\coli run`，因为它会自动设置 DeepSeek-V4 所需的
prompt role markers。

## 7. 测量速度

使用仓库提供的重复测试工具：

```powershell
$Engine = (Resolve-Path .\c\deepseek_v4.exe).Path

python c\tools\bench_deepseek_v4.py `
  --engine $Engine `
  --model $Model `
  --prompt '<｜User｜>请用一句话解释 RAM 和 VRAM 的区别。<｜Assistant｜>' `
  --tokens 32 `
  --ctx 50000 `
  --repeats 3 `
  --cold-expert
```

输出是 JSON，重点查看 `mean_tok_s`。`--cold-expert` 会要求引擎在读取 streamed
expert 后释放文件页缓存；Windows 上这是非强制提示，不等同于清空整个系统页缓存。

为了比较质量，应固定以下条件：

- 相同的转换结果和 `colibri_v4_manifest.json`；
- 相同的 prompt；
- 相同的 `--ctx` 和 `--tokens`；
- 相同的 greedy 解码方式。

## 8. 观察内存

当前实现通过把 routed experts 按 token 从 SSD 流式读取来控制常驻内存，但还没有
设置 Windows 级别的硬 RSS 上限。运行时可以在另一个 PowerShell 窗口查看：

```powershell
Get-Process deepseek_v4 -ErrorAction SilentlyContinue |
  Select-Object Id, ProcessName,
    @{Name="RSS_GB"; Expression={[math]::Round($_.WorkingSet64 / 1GB, 2)}}
```

如果出现内存不足：

1. 关闭其他占用内存的程序；
2. 确认 `$Model` 在 NVMe SSD，而不是网络盘或慢速 USB 盘；
3. 先用较小的 `--ctx` 诊断，例如 `8192`；
4. 通过 `--ngen` 限制输出长度。

目标配置仍然是 `--ctx 50000`；降低 context 只是排错或资源不足时的回退方案。

## 9. 常见错误

`missing ... config.json`

: `--indir` 指向了错误目录，或者下载尚未完成。

`expected model_type=deepseek_v4`

: 下载的不是 DeepSeek-V4-Flash-0731 官方快照。

`tokenizer.json is missing`

: 运行时使用了原始或不完整的目录；`--model` 必须指向转换后的 `$Model`。

`deepseek_v4 engine is not built`

: 先执行第 4 节的 `mingw32-make.exe` 命令。

生成速度显著低于 0.1--0.2 tok/s

: 先确认模型目录在 NVMe、CPU 支持 AVX2，并分别比较普通运行和
  `--cold-expert` 测试。15GB RAM、0.1--0.2 tok/s 是目标配置，必须在实际笔记本和
  完整权重上实测，不能仅由转换阶段验证。

