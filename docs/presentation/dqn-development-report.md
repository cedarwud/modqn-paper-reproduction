# MODQN/DQN 專案開發流程簡報報告

## 1. 報告定位

這份報告用來說明 `modqn-paper-reproduction` 這個專案是如何從論文規格一步一步被開發成可以訓練、評估與輸出實驗結果的 DQN/MODQN 系統。

重點不是展示程式碼，而是說明：

1. 環境是根據什麼建立的
2. DQN 的資料如何由 agent 與 environment 互動產生
3. state、action、reward 如何定義
4. Q value 是什麼，network 如何輸出 Q value
5. 經驗回放是什麼，為什麼能穩定訓練
6. 目標網路是什麼，為什麼 DQN 需要它
7. epsilon-greedy、action mask、checkpoint 如何讓訓練與評估更穩定
8. MODQN 如何把多目標 reward 寫成可訓練的系統

一句話主軸：

> 這個專案不是先下載一份靜態資料集來訓練 DQN，而是先根據論文建立 LEO satellite handover 的 MDP 環境。Agent 在環境中選 beam，environment 回傳 throughput、handover cost、load-balance reward，這些互動資料形成 transition，再透過 replay buffer、target network 和 Q-learning update 訓練出每個 beam 的 Q value。

## 2. 從頭到尾的開發順序

### Step 1: 先讀論文，建立權威來源

第一步不是寫 neural network，也不是搜集 dataset，而是先確認論文中的系統設定：

1. 衛星數量
2. 每顆衛星的 beam 數量
3. user 數量
4. 衛星高度
5. user 移動速度
6. satellite 移動速度
7. carrier frequency
8. bandwidth
9. transmit power
10. noise model
11. episode 與 slot 長度
12. reward 定義
13. DQN 訓練參數

本專案的 authority order 來自 repo-local paper source、layout text、catalog、phase SDD 與 assumption register。這樣做的目的是避免把沒有被論文固定的值偷偷寫死在程式裡。

### Step 2: 把論文問題轉成 MDP

DQN 必須先有 MDP，也就是：

```text
MDP = State + Action + Reward + Transition
```

在這個專案中，MDP 對應如下：

| MDP 元素 | 本專案定義 |
|---|---|
| State | user 目前接入 beam、channel quality、beam location、beam load |
| Action | 為每個 user 選擇一個 serving beam |
| Reward | throughput、handover penalty、load-balance penalty |
| Transition | user 移動、衛星位置變化、beam load 改變、下一個 state 產生 |

這一步是整個專案最重要的設計基礎。沒有先定義 MDP，就沒有辦法談 DQN training。

### Step 3: 建立 simulator environment

這個專案的 environment 是根據論文中的 LEO multi-beam satellite network 建立的。

主要 baseline 參數：

| 項目 | 值 |
|---|---:|
| satellites | 4 |
| beams per satellite | 7 |
| total beams/actions | 28 |
| users | 100 |
| altitude | 780 km |
| user speed | 30 km/h |
| satellite speed | 7.4 km/s |
| carrier frequency | 20 GHz |
| bandwidth | 500 MHz |
| slot duration | 1 s |
| episode duration | 10 s |

Environment 負責：

1. reset 初始 user、satellite、beam 狀態
2. 產生每個 user 的 state
3. 根據目前衛星可視性產生 action mask
4. 接收 agent 選出的 beam action
5. 更新時間、user 位置、beam assignment
6. 計算 throughput、handover、load-balance reward
7. 回傳 next state 和 done flag

### Step 4: 決定資料不是事先搜集，而是互動產生

這是 DQN 與一般 supervised learning 很大的差異。

Supervised learning 通常是：

```text
事先有 dataset: (x, y)
模型學 x -> y
```

DQN 是：

```text
agent 跟 environment 互動
每一步產生 transition
transition 再成為訓練資料
```

本專案每一步產生的資料格式是：

```text
(state, action, reward, next_state, done)
```

以 baseline 為例：

```text
100 users
10 slots per episode
每個 episode 最多約產生 100 × 10 = 1000 筆 transition
```

所以這裡的「資料搜集」不是人工標註資料，而是：

1. environment reset
2. agent 根據 state 選 beam
3. environment 執行 action
4. environment 回傳 reward 和 next state
5. transition 存進 replay buffer
6. replay buffer 成為 DQN 的訓練資料池

### Step 5: 把 state 編碼成 neural network input

論文中的 state 是符號式定義，程式中必須轉成固定長度向量。

本專案 state 由四個部分組成：

```text
state = [access vector, channel quality, beam offsets, beam loads]
```

其中：

1. access vector: 目前 user 接在哪個 beam
2. channel quality: user 到各 beam 的 channel/SNR
3. beam offsets: beam 相對於 user 的位置
4. beam loads: 每個 beam 目前服務多少 users

baseline 有：

```text
4 satellites × 7 beams = 28 beams
state dimension = 5 × 28 = 140
action dimension = 28
```

也就是每個 user 的 state 會被轉成長度 140 的向量，Q network 輸出 28 個 action 的 Q value。

### Step 6: 定義 Q value

DQN 學的不是直接預測 action label，而是學：

```text
Q(s, a) = 在 state s 選 action a 之後，未來累積 reward 的估計值
```

在這個專案中：

```text
state s = 某個 user 目前看到的衛星、beam、channel、load 狀態
action a = 選擇其中一個 beam
Q(s, a) = 選這個 beam 之後，未來整體 reward 會有多好
```

因為 baseline 有 28 個 beams，所以 Q network 對每個 state 會輸出：

```text
[Q(s, beam_0), Q(s, beam_1), ..., Q(s, beam_27)]
```

決策時選 Q value 最大的 beam，但會先套用 action mask，避免選到不可視或不合法的 beam。

### Step 7: MODQN 如何處理多目標 reward

一般 DQN 通常只有一個 reward：

```text
r = scalar reward
```

但這篇論文是 multi-objective reinforcement learning，因此 reward 有三個 objective：

| reward | 意義 |
|---|---|
| r1 | throughput |
| r2 | handover penalty |
| r3 | load-balance penalty |

本專案採用三個 parallel DQNs：

```text
Q1(s,a): throughput objective
Q2(s,a): handover objective
Q3(s,a): load-balance objective
```

決策時再用 objective weights 做 scalarization：

```text
Q_scalar(s,a) = 0.5 Q1(s,a) + 0.3 Q2(s,a) + 0.2 Q3(s,a)
```

然後選擇合法 action 中 `Q_scalar` 最大的 beam。

這樣可以把 multi-objective 問題轉成可執行的 DQN decision rule，同時保留三個 reward component 的可分析性。

### Step 8: 用 epsilon-greedy 進行探索與利用

DQN 不能一開始就永遠選目前 Q value 最大的 action，因為初期 Q network 還沒有學好。

所以使用 epsilon-greedy：

```text
以 epsilon 的機率隨機選合法 action
以 1 - epsilon 的機率選 Q value 最大的合法 action
```

本專案參數：

| 項目 | 值 |
|---|---:|
| epsilon start | 1.0 |
| epsilon end | 0.01 |
| decay episodes | 7000 |

意思是：

1. 前期多探索不同 beam 選擇
2. 隨著訓練進行，逐漸改成利用目前學到的 Q value
3. 後期仍保留少量探索，避免完全卡死在某個策略

### Step 9: 經驗回放 Experience Replay

如果 DQN 只用最新一步資料訓練，資料會高度連續且相關。例如同一個 user 在連續時間 slot 的 state 很接近，這會讓神經網路訓練不穩。

Experience replay 的做法是：

```text
把過去互動產生的 transitions 存進 replay buffer
每次訓練時隨機抽一批 mini-batch
用這批資料更新 Q network
```

它的作用：

1. 降低連續資料之間的相關性
2. 重複利用過去經驗
3. 讓 mini-batch 更接近隨機樣本
4. 提升 DQN 訓練穩定性

本專案參數：

| 項目 | 值 |
|---|---:|
| replay capacity | 50,000 transitions |
| batch size | 128 |
| sampling | uniform random |

在程式概念上，replay buffer 需要支援兩件事：

1. `push`: 存入新的 transition
2. `sample`: 隨機抽出 batch 供 Q network update

### Step 10: 目標網路 Target Network

DQN 的訓練目標來自 Bellman equation：

```text
target = r + γ max_a' Q(s', a')
```

但如果 `Q(s', a')` 也使用正在被更新的同一個 network，target 會一直變動，容易造成訓練不穩。

因此 DQN 使用兩套 network：

| network | 功能 |
|---|---|
| online Q network | 每次 gradient update 都更新 |
| target Q network | 固定一段時間，只定期從 online network 複製權重 |

Bellman target 使用 target network：

```text
target = r + γ max_a' Q_target(s', a')
```

本專案參數：

| 項目 | 值 |
|---|---:|
| discount factor γ | 0.9 |
| target update | hard copy every 50 episodes |

這樣做可以讓 target 在一段時間內保持穩定，使 Q-learning update 不會追著一個快速移動的目標跑。

### Step 11: Q network update

每次從 replay buffer 抽出 batch 後，DQN 做的事情是：

```text
1. 取出 state, action, reward, next_state, done
2. 用 online network 算 Q_current(s, action)
3. 用 target network 算 max Q_target(next_state, next_action)
4. 建立 Bellman target
5. 用 MSE loss 比較 Q_current 和 target
6. 反向傳播更新 online Q network
```

概念公式：

```text
target = reward + γ max_a' Q_target(next_state, a')
loss = (Q_online(state, action) - target)^2
```

MODQN 中這件事會對三個 objective 分別做：

```text
Q1 update for r1
Q2 update for r2
Q3 update for r3
```

### Step 12: Action Mask

在衛星 handover 問題中，不是所有 beams 都可以選。

有些 beam 可能因為 satellite 不可視，或不符合 eligibility rule，因此不能成為合法 action。

所以本專案在決策時使用 action mask：

```text
合法 action: 保留原本 Q value
不合法 action: Q value 視為 -∞
```

最後只會在合法 action 中選擇最大 Q value。

這是把通訊/衛星約束放進 DQN decision process 的關鍵步驟。

### Step 13: Checkpoint 與 Evaluation

訓練不是只看最後一個 episode。

本專案保留：

1. final checkpoint
2. best-eval checkpoint
3. training log
4. run metadata
5. evaluation summary

這是因為 DQN 長時間訓練可能出現 late-training collapse。也就是說，最後的 policy 不一定是訓練過程中最好的 policy。

因此專案會保留 final policy，也會另外保存 evaluation seed 上表現最好的 checkpoint，讓結果分析更可靠。

### Step 14: Comparator 與實驗輸出

為了確認 MODQN 是否真的比其他方法好，本專案不只跑 MODQN，也跑 comparator baselines：

1. `MODQN`
2. `DQN_throughput`
3. `DQN_scalar`
4. `RSS_max`

輸出實驗包含：

1. Table II weight comparison
2. Fig. 3 user count sweep
3. Fig. 4 satellite count sweep
4. Fig. 5 user speed sweep
5. Fig. 6 satellite speed sweep

這些實驗讓專案不只是「能訓練」，也能比較不同方法在不同條件下的表現。

## 3. 完整訓練流程總覽

可以把整個訓練流程放成一張投影片：

```text
load resolved-run config
        ↓
build satellite handover environment
        ↓
initialize Q1, Q2, Q3 and target networks
        ↓
for each episode:
    reset environment
    for each time slot:
        encode state
        compute Q values
        apply action mask
        choose action by epsilon-greedy
        environment step
        receive reward vector and next state
        store transition in replay buffer
        sample mini-batch from replay buffer
        update online Q networks
    periodically sync target networks
    periodically evaluate policy
        ↓
save training log, final checkpoint, best-eval checkpoint, metadata
```

## 4. DQN 關鍵機制與本專案參數總表

| DQN 機制 | 概念 | 本專案設定 |
|---|---|---|
| State | agent 看到的環境資訊 | access, channel quality, beam offsets, beam loads |
| Action | agent 可執行的選擇 | choose one serving beam |
| Reward | action 好壞的回饋 | r1 throughput, r2 handover, r3 load balance |
| Q value | action 的未來累積 reward 估計 | 每個 beam 一個 Q value |
| MODQN | 多目標 Q-learning | Q1, Q2, Q3 三個 DQN |
| Scalarization | 多目標加權決策 | `[0.5, 0.3, 0.2]` |
| Replay buffer | 儲存過去 transition | capacity `50,000` |
| Mini-batch | 每次 update 的樣本數 | batch size `128` |
| Target network | 穩定 Bellman target | every `50` episodes hard update |
| Discount factor | 未來 reward 折扣 | `0.9` |
| Epsilon-greedy | 探索與利用 | `1.0` to `0.01` over `7000` episodes |
| Network | Q function approximator | hidden layers `[100, 50, 50]`, `tanh` |
| Optimizer | gradient update | Adam, learning rate `0.01` |
| Training length | baseline paper protocol | `9000` episodes |

## 5. 概念式程式模組對應

這一段可以用來說明「怎麼寫進程式裡運行」，但不用展示程式碼。

| 概念 | 程式中對應的責任 |
|---|---|
| Environment | reset、step、state/reward/action mask 產生 |
| State encoding | 把論文 state 轉成 fixed-length vector |
| Replay buffer | 存 transition，隨機抽 mini-batch |
| Q network | 輸入 state，輸出每個 beam 的 Q value |
| MODQN trainer | 建立 Q1/Q2/Q3、選 action、update networks |
| Target network sync | 每隔固定 episode 複製 online network 權重 |
| Training orchestration | 載入 config、啟動訓練、寫 checkpoint 和 metadata |
| Sweep/evaluation | 跑 Table II、Fig. 3 到 Fig. 6，比較不同方法 |

## 6. 建議投影片架構

### Slide 1: Project Goal

說明這個專案是在重現 LEO multi-beam satellite handover 的 MODQN 論文，目標是建立可訓練、可評估、可比較的 baseline reproduction。

### Slide 2: Why This Is Not a Static Dataset Problem

說明 DQN 的資料不是先下載 dataset，而是 agent 與 environment 互動產生 transition。

### Slide 3: Paper Specification to Simulator

列出主要環境參數：satellites、beams、users、altitude、speed、bandwidth、slot、episode。

### Slide 4: MDP Definition

用表格呈現 state、action、reward、transition。

### Slide 5: How Data Is Generated

用流程圖呈現：

```text
state → action → environment step → reward + next_state → replay buffer
```

### Slide 6: What Q Value Means

說明 Q(s,a) 是未來累積 reward 的估計，在此專案中每個 action 對應一個 beam。

### Slide 7: MODQN Design

說明三個 Q networks 對應三個 objective，最後用 weights scalarize。

### Slide 8: Training Loop

展示完整 training loop：reset、select action、step、store transition、sample、update、sync target。

### Slide 9: Experience Replay

說明 replay buffer 的目的、運作方式與參數。

### Slide 10: Target Network

說明 online network 與 target network 的差異，以及 Bellman target。

### Slide 11: Exploration and Action Mask

說明 epsilon-greedy 與 invalid beam masking。

### Slide 12: Key Parameters

放完整參數表。

### Slide 13: Evaluation and Checkpointing

說明 final checkpoint、best-eval checkpoint、training log、sweeps。

### Slide 14: Result Interpretation

說明目前專案完成的是 disclosed comparison baseline，但尚未完全證明 paper-faithful method separation。

### Slide 15: Summary

用三句話總結：

1. 先從論文建立 MDP 與 simulator
2. DQN 資料由 interaction 產生，透過 replay buffer 訓練
3. MODQN 用三個 Q networks 處理 throughput、handover、load-balance 三目標

## 7. 可以直接放進簡報的濃縮說法

### 開場說法

這個專案的核心不是先搜集一份靜態資料集，而是先把論文中的 LEO 多波束衛星 handover 問題轉成一個可互動的 MDP environment。DQN agent 在每個 time slot 根據 user state 選擇 serving beam，environment 回傳 throughput、handover penalty 和 load-balance reward，這些互動產生的 transition 才是訓練資料。

### 資料產生說法

每一步互動都會產生一筆 DQN transition：

```text
(state, action, reward, next_state, done)
```

這些 transitions 被存進 replay buffer。訓練時不是照時間順序直接使用最新資料，而是從 buffer 隨機抽 mini-batch，降低資料相關性並提升訓練穩定性。

### Q value 說法

Q value 表示在某個 state 下選擇某個 action 後，未來可以得到多少累積 reward。在這個專案中，action 就是選擇 beam，因此 Q network 對每個 user state 輸出 28 個 beam 的 Q value，再從合法 beams 中選擇 Q value 最高的 action。

### MODQN 說法

一般 DQN 只學一個 scalar reward，但這個問題有 throughput、handover、load-balance 三個目標。因此本專案使用三個 parallel DQNs，分別估計三個 objective 的 Q value，決策時用 `[0.5, 0.3, 0.2]` 權重做 scalarization，選出綜合 Q value 最高的 beam。

### 穩定訓練說法

DQN 訓練穩定性主要靠三個機制：experience replay、target network 和 epsilon-greedy。Replay buffer 讓訓練資料隨機化；target network 讓 Bellman target 在一段時間內保持穩定；epsilon-greedy 則讓 agent 前期探索、後期利用已學到的 Q value。本專案另外使用 action mask，確保不可視或不合法的 beam 不會被選到。

## 8. 誠實結論

目前這個 repo 可以被說明為：

1. 已建立完整可執行的 DQN/MODQN baseline reproduction surface
2. 已完成 training、checkpoint、replay buffer、target network、sweep、export 與 comparator evaluation
3. 已能產生 Table II 與 Fig. 3 到 Fig. 6 的機器可讀 artifact
4. 但尚未完全重現論文中明顯的 method separation
5. 因此目前最準確的定位是 disclosed comparison baseline，而不是完整 paper-faithful reproduction claim

