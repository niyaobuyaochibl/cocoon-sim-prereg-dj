# J 轨：D 轨的期刊扩展（投 Information Sciences）

日期：2026-11-09（初稿），2026-11-10（登记前按两份独立盲审修订）  
**写在任何 D 轨或 J 轨 dump 之前。** 与 `prereg_d.md` 一起登记（公开 GitHub 仓库的 release，由 Zenodo 存档并分配 DOI），登记后才开跑。

## 0. 与 D 轨的关系

投稿场地由 DASFAA 2027 改为 Information Sciences（期刊，无页数限制）。本文件只加臂和判据，不改 `prereg_d.md` 的任何判据。

- D0–D4 只在 G1–G3 上按 `prereg_d.md` 原文判。新增的 G4 用同一规则单列一行，不并入 D 轨判决，D 轨判据不因多一个生成器而变严或变松。
- 统计口径（估计对象、区间、门槛失效）和技术回退规则全部沿用 `prereg_d.md` 第 4、5 节。
- 不改 `prereg.md`、`prereg_a.md`、`prereg_r.md`、`prereg_r_loops.md`。10% 线、0.05 线不动。A2、R3 的「不支持」判决照登。

## 1. 生成器

| 代号 | 模型 | 说明 |
|------|------|------|
| G1–G3 | 同 `prereg_d.md` | 不变 |
| G4 | Llama-3.1-8B-Instruct，本地，贪心，与 G2 同环境（torch 2.4.1+cu121，transformers 4.46.3，`LOCAL_ATTN=eager`） | 第三个模型家族。记录权重来源和各文件哈希。用模型自带聊天模板，不改系统提示 |

## 2. 新增的臂

### J1 第二个数据集：EB-NeRD

EB-NeRD 同时改变了语言（丹麦语）、类别体系和曝光结构。J1 检验的是「结论在另一个语言、另一个平台的日志上是否仍成立」，不能单独归因于其中任何一项；论文按此限定措辞。

**转换**（`data/ebnerd_prepare.py`，只依据字段结构，不看任何模拟结果）：

- 只用 EB-NeRD small 的 train 和 validation 划分及 articles 表。
- `news.parquet`：每篇文章一行。`category` = `category_str`；`subcategory` = `subcategory` 列表的第一个元素，列表为空记为缺失；`title` 为丹麦语原文，不翻译；`popularity` = 该文章在 train 划分里被点击的次数。`category_str` 缺失的文章不进入任何候选、历史或点击。
- `impressions_dev.parquet`：validation 的每条曝光一行。`history` = 该用户 validation 历史文件里的文章，按阅读时间升序；`candidates` = `article_ids_inview` 原顺序，去重保留首次出现，去掉 articles 表里没有或缺类别的文章；`clicked` = `article_ids_clicked` 与 `candidates` 的交集。
- `sessions_train.parquet`：train 的每条曝光一行，按同样规则；`session_idx` 为该用户曝光按时间升序的序号（从 1 起，同时刻按曝光 ID 升序）；`history` 为该用户 train 历史文件。
- `users.parquet`：每个 validation 用户一行，`history` 同 `impressions_dev`。
- ID 一律转成字符串。
- **曝光筛选**：`impressions_dev` 只留同时满足「点击 \(\ge 1\)、历史 \(\ge 5\) 条、候选 5–50 条」的曝光，报每一步筛掉的数量。

**臂**（统计口径同 `prereg_d.md` 第 4 节）：

- 日志列表上的人类 ER、出历史供给 \(S\)、比值 \(ER/S\)。
- E1 retry：G2、G3、G4。E1 cal（oracle 目标）：G2、G4。均为筛选后的曝光表上 `sample(n=1500, random_state=0)`，所有生成器共用同一 1500 条。
- 随机列表闭环：G2、G3、G4 未校准臂，种子 0，200 用户 × 10 轮。用户用 `sample_users(users, n=200, seed=0, min_hist=10)` 从 `users.parquet` 抽，与 MIND 相同；随机列表从 `news.parquet` 全部文章抽。同一列表上跑 D3 的均匀和拟合参照点击者，\(\pi\) 在 EB-NeRD 的 `sessions_train` 上按 `prereg_d.md` D3 的细节重新拟合。
- NRMS 闭环：G2、G4，种子 0，未校准和校准两臂，描述性（不对应判据）。NRMS 只训练一次，配置与 MIND 上无 GloVe 的 NRMS 相同（`recsys/train_nrms.py`：词向量随机初始化，维度 256，2 个 epoch，batch 256，Adam lr \(10^{-4}\)，每个正例 2 个负例，`random`、`numpy`、`torch` 种子都设 0），取验证 AUC 最高的 epoch；AUC 为 validation 上 `random_state=0` 抽样曝光的条目合并 AUC，与 MIND 的计算方式相同。AUC \(\ge 0.55\) 才跑闭环；低于、训练失败或 AUC 缺失都不跑，报原因。
- 不做回放、不做多种子。

### J2 第三个模型家族（MIND 上）

G4 跑以下臂，路径在 `runs/j_track/g4/`：

- E1：retry、cal、one、empty、shuffled。不跑 temp1。
- 离线：D1-top1、D2-paired、D2-train（E1 规则）、D2-book、D8，规则同 `prereg_d.md`。
- 随机列表闭环种子 0–2 两臂；NRMS、NAML、DKN 闭环种子 0–2 两臂。
- 种子 0 回放三臂（未校准、oracle 目标校准、训练目标校准）。

G4 的 E1 retry 空点击率超过 10% 时，闭环和回放都不跑。

### J3 子类别指标（离线）

在所有有条目记录的运行上补算：子类别 ER（点击条目的子类别不在历史子类别集合里的占比）、Subcat@20（Top-20 的不同子类别数）。子类别缺失的条目不进点击分母，也不进历史子类别集合；某曝光的点击全无子类别时该曝光不计。MIND 270 个子类别；EB-NeRD 用上面转换出的子类别。旧 dump 中没有逐轮列表的运行只算 E1。

### J4 提示词敏感性（MIND E1）

同一 1500 条曝光，retry 臂，G2 和 G4。三个因子，2×2×2 共 8 个变体：

- **人设摘要**：有 / 无（`sim/prompts.py` 的 `Reader profile` 段整段保留或去掉）。
- **点击率提示句**：有 / 无（只去掉「Real readers click about … of the headlines they see.」这一句，其后的返回格式句不变）。
- **历史长度** \(h\)：30 / 10。提示里的历史列表和人设摘要都只用最近 \(h\) 条。ER 的参照历史始终是该曝光的完整历史，不随 \(h\) 变。

「有、有、30」就是主实验的 retry 臂（G2 用 `runs/rep7b/`，G4 用 J2 的 retry），不重跑，每个生成器新跑 7 个变体。空解析重试的文本与主实验相同。8 份完整模板在开跑前由代码生成并存档到 `runs/j_track/prompt/templates/`，不另加措辞、人设或示例。

## 3. 判据

| ID | 预测 | 支持 | 不支持 |
|----|------|------|--------|
| **J0** | G4 的每个格子、EB-NeRD 上的每个 LLM 格子、J4 的每个变体，空点击 \(\le 10\%\)（G4 的 D1-empty 豁免） | 每格单独算，口径同 D0 | \(>10\%\) 的格子标为「仪器失效」，只进敏感性 |
| **J1a** | EB-NeRD 上模拟器比真人少探索 | G2、G3、G4 的 E1 retry 臂各自满足 \(\Delta_{ER}\) 点估计 \(\le -0.05\) 且 95% CI 上界 \(<0\) | 任一生成器不满足：报哪个、差多少。有生成器仪器失效：判「不可判定」，另报其余生成器的估计 |
| **J1b** | EB-NeRD 随机列表第 1 轮，LLM 点击重复高于拟合点击者 | 同 D3 的规则，G2、G3、G4 各自成立 | 同 D3：另看（LLM − 均匀），CI 下界 \(>0\) 才写「高于均匀参照」 |
| **J2** | G4 在 MIND 上复现 D1–D4 | 按 D1–D4 原规则单列 G4 一行（D1、D2 只看 G4 自身；D4 用 G4 的种子 0–2 两臂） | 哪条不成立就报哪条 |
| **J3** | 子类别层面模拟器同样少探索 | MIND 的 E1 retry 臂上，子类别 \(\Delta_{ER}\) 点估计 \(\le -0.05\) 且 95% CI 上界 \(<0\)，G1–G4 各自成立 | 任一生成器不成立：类别层面的结论不外推到子类别 |
| **J4** | 少探索的方向不依赖提示词措辞 | G2、G4 各自的 8 个变体全部通过 J0 且全部满足 \(\Delta_{ER}\le -0.05\)、95% CI 上界 \(<0\) | 报满足的变体数、不满足和仪器失效的变体；论文不写「对提示词稳健」 |

J4 另报描述量：8 个变体的 \(\Delta_{ER}\) 极差，以及每个因子两水平的平均差。

## 4. 统计

同 `prereg_d.md` 第 4 节。各判据分别给判决，不做多重比较校正，论文不把多个判据合成一个总判断。

## 5. 停止

- 第二个数据集只用 EB-NeRD small，不加第三个。
- 提示词只跑第 2 节 J4 的 8 个变体，不因结果加变体、换措辞。
- 模型家族只加 G4，不因结果换模型或加模型。
- 看到结果后不改 EB-NeRD 的转换规则、曝光筛选、类别字段、语言处理。
- 若下载后发现 EB-NeRD 的字段与第 2 节描述不符，在跑任何 LLM 臂之前，把只依据字段结构的改动作为带日期的修订追加到本文件末尾，发布新的 release 让 Zenodo 存为新版本，再开跑。
- NRMS 在 EB-NeRD 上只训练一次，不调参、不换种子重训。
- `prereg_d.md` 第 5 节的规则照旧。

## 6. 输出与跑法

输出：`runs/j_track/ebnerd/`、`runs/j_track/g4/`、`runs/j_track/prompt/`、`runs/j_track/analysis/`。不覆盖 `runs/a_track/`、`runs/rep7b/`、`runs/d_track/`。EB-NeRD 的 NRMS 检查点放 `checkpoints/ebnerd/nrms/`，不与 MIND 检查点混用。

```text
# 本地 CPU
python data/ebnerd_prepare.py                     # 转换 + 各步筛选数量
python scripts/d_track.py ref --data ebnerd --chooser uniform|fitted
python analysis/j_subcat.py                       # J3
python analysis/j_verdicts.py                     # J0–J4 判决表

# G4（AutoDL），MIND
python scripts/d_track.py e1 --gen g4 --arm retry|cal|one|empty|shuffled
python scripts/d_track.py placebo --gen g4 --seed {0,1,2} [--calibrate]
python scripts/d_track.py e3 --gen g4 --rec {nrms,naml,dkn} --seed {0,1,2} [--calibrate]
python scripts/d_track.py replay --gen g4 --seed 0 [--calibrate [--target-mode train]]

# 提示词变体（G2、G4；跳过 on/on/30）
python scripts/d_track.py e1 --gen {g2,g4} --arm retry --profile {on,off} --rate-hint {on,off} --hist {30,10}

# EB-NeRD
python recsys/train_nrms.py --data ebnerd --seed 0
python scripts/d_track.py e1 --data ebnerd --gen {g2,g3,g4} --arm retry
python scripts/d_track.py e1 --data ebnerd --gen {g2,g4} --arm cal
python scripts/d_track.py placebo --data ebnerd --gen {g2,g3,g4} --seed 0
python scripts/d_track.py e3 --data ebnerd --gen {g2,g4} --rec nrms --seed 0 [--calibrate]
```
