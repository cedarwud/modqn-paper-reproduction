# MODQN 論文重現專案技術報告

## 詳細技術報告版

### 核心判讀

這個 repo 的核心不是「把一篇論文翻成幾個 Python 檔」，而是把論文中的多目標 LEO 衛星 handover 問題，先嚴格落成一個可執行的 MOMDP／MDP 環境，再讓 DQN/MODQN 透過互動自己產生訓練資料、學 Q value、做 checkpoint 與 sweep，最後輸出可比較的 Table II 與 Fig. 3–6 artifact。你提供的初步報告把主軸放在「不是先找 supervised dataset，而是先建 environment」，這個方向和論文本身、repo 的 phase 文件、以及真實程式結構是完全一致的；下面我補上的，是每一步在程式裡究竟落在哪些模組，以及哪些地方其實是 repo 明確揭露的 reproduction assumption。fileciteturn0file1 fileciteturn11file0 fileciteturn30file0

如果用一句話總結，我會說：**這個專案已經是一個 training-ready、可重跑、可比較、可匯出的 disclosed comparison baseline，但它仍不能誠實地自稱為完整 paper-faithful reproduction。** README、baseline/closeout 文件與後續 reopen note 都反覆強調：repo 已完成 baseline 訓練、比較器、sweeps、checkpoint、bundle export 與診斷，但 method separation 仍不夠明顯，長程訓練也仍有 late-run collapse 與 reward-geometry 問題，因此最強的誠實說法是「paper-consistent reproduction baseline」，而不是「已完整重現原論文」。fileciteturn7file0 fileciteturn44file0 fileciteturn15file0 fileciteturn16file0

從 DQN 觀點看，這個專案真正重要的理解點有三個。第一，**資料不是先存在的標註集**；它是 agent 在 simulator 中每一步互動產生的 transition。第二，**Q network 的輸出不是分類 label，而是對每個 beam action 的長期報酬估計**。第三，**訓練穩定性不是來自單一技巧，而是 replay buffer、target network、epsilon-greedy、action mask、checkpoint selection 與 evaluation seed protocol 一起構成的工程化穩定面**。這些機制既符合原始 DQN 的經典作法，也和論文的 MODQN 演算法描述相呼應。citeturn0search0 fileciteturn0file0 fileciteturn32file0

### 從論文走到可執行系統

repo 在設計上先把「權威來源」和「可執行配置」分開。README 與 Phase 01 SDD 明確規定 authority order：先看 paper-source／論文，再看 assumption register，再看 phase SDD；而 config 也故意分成 `paper-envelope` 與 `resolved-run` 兩種角色。前者只記錄論文明載參數、可恢復的 weight rows 與 assumption references；後者才把 epsilon schedule、target update、replay capacity、seed、checkpoint rule 等真正會影響訓練執行的值凍結下來。`config_loader.require_training_config()` 甚至會直接拒絕 `paper-envelope` config，避免程式偷偷用 hidden default 把「論文外沒寫的值」補上。這是整個專案最重要的工程邏輯：**先把 paper 與 assumption 的邊界說清楚，再允許訓練。**fileciteturn7file0 fileciteturn11file0 fileciteturn13file0 fileciteturn14file0 fileciteturn30file0

也因此，這個專案的開發順序不是「先收集資料、再丟進神經網路」，而是相反：**先把論文問題轉成 MDP/MOMDP，再讓資料從互動中長出來。** 論文本身先把 handover 問題建模成 multi-objective optimization，再轉成 MOMDP；Algorithm 1 也明確描述了 agent 在每個 step 用 epsilon-greedy 選 action、拿到 reward vector、把經驗放進 replay memory、再以 batch 方式更新 DQN。這也是為什麼不能先搜集一份 supervised dataset：因為這裡沒有預先存在的正確 beam label，只有在當前狀態下採取某個 beam action 後，經由環境動態與後續回報才定義的長期價值。DQN 的學習目標本質上是 Bellman-style bootstrapping，不是靜態標註映射。fileciteturn0file0 citeturn0search0 fileciteturn32file0

如果用 repo 的 phase 演進來看，Phase 01 先建立 standalone Python baseline reproduction；Phase 01B 再針對 paper scenario mismatch 做顯式 follow-on；Phase 01C 檢查 comparator protocol 是否是 near-tie 的來源；Phase 01E/01F/01G 則分別處理 beam semantics、beam-aware eligibility follow-on 與 atmospheric-sign counterfactual。換句話說，專案並不是「一次到位重現論文」，而是先落地 baseline，再用 phase 文件把每一個不確定性與修正方向拆成明確的 reopening slice，這也是它能誠實宣告 claim boundary 的原因。fileciteturn11file0 fileciteturn43file0 fileciteturn15file0 fileciteturn17file0 fileciteturn18file0 fileciteturn19file0

### 系統模型如何變成環境與資料

論文與 baseline config 共同固定了 default scenario 的骨架：4 顆衛星、每顆 7 個 beam、100 個 users、780 km 高度、30 km/h 使用者速度、7.4 km/s 衛星速度、20 GHz、500 MHz、2 W、-174 dBm/Hz、Rician K=20 dB、1 秒 slot、10 秒 episode，以及 MODQN 的隱藏層、activation、optimizer、learning rate、discount factor、batch size 與預設權重 `[0.5, 0.3, 0.2]`。這些值在論文與 repo config 被當成 baseline surface；真正進到 runtime 時，`config_loader` 再把它們轉成 `StepConfig`、`OrbitConfig`、`BeamConfig`、`ChannelConfig` 與 `TrainerConfig`。fileciteturn0file0 fileciteturn11file0 fileciteturn13file0 fileciteturn14file0 fileciteturn30file0 fileciteturn46file0

真正的 environment 實作在 `src/modqn_paper_reproduction/env/step.py` 的 `StepEnvironment`。它把論文的狀態  
\( s_i(t) = (u_i(t), G_i(t), \Gamma(t), N(t)) \)  
落成 `UserState` 四個欄位：`access_vector`、`channel_quality`、`beam_offsets`、`beam_loads`。其中 `u_i(t)` 是目前接入 beam 的 one-hot；`G_i(t)` 是 user 對所有 beams 的 SNR；`\Gamma(t)` 是每個 beam center 相對 user 的 local-tangent `(east_km, north_km)`；`N(t)` 是各 beam 的 pre-decision load。到了 runtime/state encoding，再把這四塊依序攤平為 `[access, log1p(snr), offsets/scale, loads/num_users]`，因此每個 beam 對 state 貢獻 1 + 1 + 2 + 1 = 5 個特徵，所以 `state_dim_for(LK) = 5 * L * K`。在 baseline 的 4×7 拓樸下，action 維度是 28，state 維度是 140。fileciteturn0file0 fileciteturn37file0 fileciteturn36file0 fileciteturn32file0

Action 在論文裡是 access vector；在 repo 裡則是「對每個 user，選一個 global beam index」。這個 index 採用固定的 `satellite-major, beam-minor` 排序，因此 `action_dim = num_satellites × beams_per_satellite`。`StepEnvironment` 會先以 visibility 生成 `ActionMask`，baseline 預設是 `satellite-visible-all-beams`：只要某顆衛星可視，該衛星底下所有 beams 都先被視為合法 action；follow-on config 才會把它改成 `nearest-beam-per-visible-satellite`。這個排序與 mask semantics 都是 repo 有意識揭露的 reproduction assumption，而不是論文明文固定。fileciteturn11file0 fileciteturn12file0 fileciteturn14file0 fileciteturn37file0 fileciteturn45file0

Reward 在 code 裡完全對應論文的三目標。`r1` 是 throughput，直接用 Eq. (3) 的 \( B/N_b \cdot \log_2(1+\gamma) \) 計算；`r2` 是 handover penalty，若 beam 不變則為 0，同衛星換 beam 為 `-phi1`，跨衛星 handover 為 `-phi2`；`r3` 是 load-balance 項，repo 預設採用 `-(max_beam_thr - min_beam_thr)/U`，而且 gap 的範圍不是只看 occupied beams，而是 Phase 01 已明講的 `all-reachable-beams`，也就是所有至少對某位 user 可達的 beams，空 beam throughput 以 0 計。這裡的 `phi1=0.5`、`phi2=1.0`、reachable-beam semantics 都是 repo 的 disclosed assumption，不是論文明講。fileciteturn0file0 fileciteturn11file0 fileciteturn12file0 fileciteturn14file0 fileciteturn37file0

Transition dynamics 也很具體。`reset()` 會先生成 user positions、user headings、初始 assignments，並把每個 user 先接到「最近的可視 beam」；`step()` 則在每個 slot 先推進時間與 user mobility，再套用 action 更新 assignment，之後重建 next state、next masks、beam throughput 與 rewards，最後回傳 `done`。換言之，這個專案的 transition 絕對不是抽象概念，而是清楚地由 **orbit 幾何、beam 幾何、channel 模型、mobility 模型與 assignment 更新** 一起決定。orbit 由 `OrbitProxy` 提供單平面 circular shell proxy；beam 由 `BeamPattern` 提供 hex-7 幾何與 `nearest_beam()`；channel 由 `compute_channel()` 計算 FSPL、atmospheric factor、Rician fading 與 SNR；mobility 則在 baseline 用 deterministic heading stride，在 follow-on 才引入 `random-wandering`。fileciteturn38file0 fileciteturn39file0 fileciteturn40file0 fileciteturn37file0 fileciteturn43file0

最重要的是：**訓練資料就是在這個 environment 裡長出來的。** baseline 一個 episode 有 10 個 slots，每個 slot 100 個 users，所以每個 episode 最多產生約 1000 筆 per-user transition；9000 個 episodes 的完整訓練過程會生成大量互動經驗，但 replay buffer 只保留最近的 50,000 筆。論文的 Algorithm 1 寫的是「每個 objective 各有 replay memory」，但 repo 的真實 MODQN 實作不是三個 replay memories，而是**一個共享的 `ReplayBuffer`**，每筆經驗存的是 `(state, action, reward_3, next_state, mask, next_mask, done)`；更新時再從 `reward_3[:, obj_idx]` 切出對應 objective 的 reward column。這是你做口頭報告時很值得強調的一點：**repo 沒有盲目照抄 pseudocode，而是做了等價但更工程化的共享 buffer 實作。**fileciteturn0file0 fileciteturn32file0 fileciteturn34file0 fileciteturn37file0

### Q 值學習與穩定訓練機制

Q value 的本質不是「哪個 beam 比較好」的即時分數，而是  
**在 state \(s\) 下選擇 action \(a\) 後，未來折扣累積報酬的估計值 \(Q(s,a)\)**。  
在這個專案裡，state 是某個 user 當下看到的整個 beam surface；action 是從 28 個 global beam actions 中選一個；因此 Q network 的輸出就是長度 28 的向量，每一格對應一個 beam action 的 Q value。`runtime/q_network.py` 的 `DQNNetwork` 只做一件事：把長度 140 的 state 向量送進 `[100, 50, 50]` 的 MLP，輸出 `action_dim` 個 Q values。決策時如果不用 mask，網路可能會選到不可視 beam；所以 code 在 greedy 決策前，會把 invalid actions 的 Q 設為 `-inf`，只在合法 beam 中做 argmax。fileciteturn35file0 fileciteturn36file0 fileciteturn32file0 fileciteturn37file0

Experience replay 在這個專案裡不是附屬品，而是 DQN 可以穩定學習的必要條件。原始 DQN 之所以引入 replay memory，是為了把高度時間相關的序列經驗打散成近似 i.i.d. 的 mini-batches，減少梯度更新對最近幾步局部樣本的過度擬合；repo 這裡也照這個精神做。`ReplayBuffer` 採 FIFO、容量 50,000、uniform random sampling；當 buffer 內樣本數還小於 batch size 128 時不更新，滿足後才在每個 slot 後抽 128 筆來做一次整批更新。因為每個 slot 都先推入 100 位 users 的 transition，再呼叫一次 `update()`，所以 replay 與 optimizer step 的節奏，是「**per-slot 蒐樣、per-slot 更新**」，而不是「每個 user 個別更新一次」。citeturn0search0 fileciteturn12file0 fileciteturn34file0 fileciteturn32file0

Target network 的角色也非常清楚。DQN 若直接用正在更新的 online network 自己產生 bootstrap target，target 會跟著參數一起飄，學習就容易震盪。原始 DQN 因此用兩套網路：online network 負責被更新，target network 負責在一段時間內固定，提供較穩定的 Bellman target；repo 的 `MODQNTrainer` 與 `ScalarDQNTrainer` 都保留這個設計。以 MODQN 為例，每個 objective 都有自己的 `q_nets[i]` 與 `target_nets[i]`，target 預設每 50 episodes 做一次 hard copy。由於 environment 一個 episode 恰好是 10 個 slots，50 episodes 也等於 500 個 environment steps；而每個 episode 產生約 1000 筆 per-user transitions，因此這個 cadence 和 50,000 的 replay capacity 大致落在「一輪 target sync 約剛好灌滿一個 buffer 規模」的量級上。citeturn0search0 fileciteturn12file0 fileciteturn32file0 fileciteturn33file0

Bellman update 在實作上長這樣：先算目前 online network 的 `Q(s,a)`，再用 target network 計算下個狀態所有合法 actions 的 `Q_target(s',a')`，取最大值組成 target，最後用 MSE loss 更新 online network。repo 實作其實比概念式公式多做了一件很關鍵的事：**replay 裡把 `next_mask` 一起存下來**，更新時先把 invalid next-actions 的 Q 壓成大負值，再做 `max`。也就是說，這個專案不是只在 decision time 做 action mask，而是把「next-state 的合法 action 集合」一起帶進 Bellman backup，避免 target 去看一個實際永遠不能執行的 beam。fileciteturn34file0 fileciteturn32file0

Epsilon-greedy 也是訓練穩定性的核心之一。resolved config 把 schedule 明確凍結為 `1.0 → 0.01`、線性 decay 7000 episodes；意思是前七千個 episodes 逐步從大量探索過渡到大量利用，最後兩千個 episodes 維持低探索尾端。這和原始 DQN 的精神一致：一開始先讓 agent 多看不同狀態–動作結果，避免因為初始 Q 還不準就過早卡死在某個 beam 選擇；等 replay 裡有足夠 diverse transition 後，再逐漸把 policy 拉回 exploitation。repo 還把 evaluation 明確設成 `eps=0.0` 的 greedy rollout，避免 training 與 evaluation 混在一起。citeturn0search0 fileciteturn12file0 fileciteturn14file0 fileciteturn32file0

整個訓練流程，在 `run_train_command()` 與 `MODQNTrainer.train()` 裡可以被濃縮成下面這條 pseudo-flow：

```text
load resolved config
→ build environment + trainer config + seeds
→ initialize Q1/Q2/Q3, target nets, optimizers, replay
→ for each episode:
    reset environment
    encode per-user states
    for each slot:
        epsilon-greedy masked action selection
        env.step(actions)
        encode next states
        push per-user transitions into replay
        sample mini-batch
        Bellman update for each objective network
    every 50 episodes: sync target nets
    every eval cadence and final episode: greedy evaluation on eval seeds
→ save training_log.json
→ save final checkpoint
→ save best-eval checkpoint
→ write run_metadata.json
```

這個流程不是文件敘述而已，而是 train entrypoint 的真實執行骨架。特別是 best-eval checkpoint 的選擇規則：repo 用 evaluation seed set 的**平均 weighted reward** 來挑 secondary checkpoint，而不是只信最後一個 episode。這也是為什麼它能對 late-training collapse 做比較誠實的揭露。fileciteturn31file0 fileciteturn32file0 fileciteturn14file0

### MODQN 多目標設計

一般 DQN 只有一個 scalar reward，因此只需要一個 Q network；MODQN 的核心差異，是它把 throughput、handover cost、load balance 這三個目標拆開來學。論文把這三個 reward 分量記成 \(r_1, r_2, r_3\)，並透過多個 DQNs 各自逼近對應的 action-value function；repo 的 `MODQNTrainer` 也忠實實作成三個 parallel DQNs：`q_nets = [Q1, Q2, Q3]`、`target_nets = [T1, T2, T3]`、`optimizers = [Adam1, Adam2, Adam3]`。這個設計的好處是，網路不需要在訓練時把所有目標先壓成一個 scalar loss surface；每個 objective 先各自形成自己的 Q landscape，最後才在決策時做 scalarization。fileciteturn0file0 fileciteturn32file0

在這個專案裡，三個 objectives 的實際含義非常具體。`Q1(s,a)` 對應 throughput objective，學的是「選這個 beam 之後，未來 throughput 積分大概多好」；`Q2(s,a)` 對應 handover objective，學的是「這個 beam policy 將來在 handover penalty 上大概多痛」；`Q3(s,a)` 對應 load-balance objective，學的是「它會把 beam throughput gap 拉大還是拉小」。決策時，`_scalarize_q_values()` 直接算  
`w1 * Q1 + w2 * Q2 + w3 * Q3`，baseline 預設權重就是 `[0.5, 0.3, 0.2]`。因此你在報告時可以明確地說：**repo 不是在 reward 端把三目標硬加總再學一個 Q，而是在 value 端先分網路，再於 action selection 做線性標量化。**fileciteturn0file0 fileciteturn13file0 fileciteturn32file0 fileciteturn47file0

不過 repo 也保留了單目標與單標量 comparator，這正好能拿來對照一般 DQN 與 MODQN 的差異。`DQN_scalar` 的真實實作在 `algorithms/dqn_scalar.py`，它只維持一個 `q_net` / `target_net`，reward 在存入 replay 前就先用一組權重壓成 scalar；`DQN_throughput` 則**沒有獨立的 `dqn_throughput.py` 檔案**，而是 `sweeps.py` 用 `ScalarDQNPolicyConfig(name="DQN_throughput", scalar_reward_weights=(1.0, 0.0, 0.0))` 動態組出來的 special case。這一點非常值得口頭說清楚，因為它直接證明你真的讀過 repo，而不是只憑名字想像模組結構。fileciteturn33file0 fileciteturn41file0

權重 protocol 也不是隨意寫的。Phase 01 SDD 已規定：`MODQN` 在同一 topology family 下訓練一次，評估時再套不同 Table II weight rows 重新 scalarize；`DQN_scalar` 因為 reward 本身就依賴權重，所以必須 per-weight-row retrain；`DQN_throughput` 與 `RSS_max` 則不 need per-row retrain。這個 protocol 很重要，因為它決定了「MODQN 的多目標學習能力」和「單一 scalar DQN 的權重敏感性」到底怎麼被公平比較。fileciteturn11file0 fileciteturn12file0 fileciteturn41file0

這裡還有一個和論文、程式、結果三者都連在一起的關鍵細節：repo 的 Phase 1 channel model **不使用 off-axis beam gain**。`beam.py` 明確寫出 off-axis angle 只提供 state／未來擴充用，Phase 1 channel quality 主要由 user–satellite slant range、atmospheric factor 與 fading 決定；`step.py` 甚至在可視衛星上把同一顆衛星底下所有 beams 的 `snr_arr` 都設成相同值。這件事與 baseline 的 `satellite-visible-all-beams` eligibility 一組合，就很容易形成 beam-level collapse：同衛星 beams 的 channel 幾乎不分家，Q 排序更多是由 offsets、loads 與 tie-breaking 決定，而不是 beam-specific channel gain。後面 Phase 01E/01F 的 beam semantics reopen，正是圍繞這個問題展開。fileciteturn39file0 fileciteturn37file0 fileciteturn17file0 fileciteturn18file0

### 參數總表與程式模組對應

| 項目 | baseline / 規則 | 性質 | 主要落點 |
|---|---:|---|---|
| satellites | 4 | paper-backed baseline | `configs/modqn-paper-baseline.yaml` / `build_orbit_config()` |
| beams per satellite | 7 | paper-backed baseline | `configs/modqn-paper-baseline.yaml` / `BeamConfig` |
| users | 100 | paper-backed baseline | `configs/modqn-paper-baseline.yaml` / `StepConfig` |
| total actions | 28 | 由 4×7 推得 | `MODQNTrainer.action_dim` |
| state dimension | 140 | 由 `5 × L × K` 推得 | `state_dim_for()` |
| hidden layers | `[100, 50, 50]` | paper-backed baseline | `TrainerConfig.hidden_layers` |
| activation | `tanh` | paper-backed baseline | `DQNNetwork` |
| optimizer | `Adam` | paper-backed baseline | `MODQNTrainer.optimizers` / `ScalarDQNTrainer.optimizer` |
| learning rate | `0.01` | paper-backed baseline | `TrainerConfig.learning_rate` |
| discount factor | `0.9` | paper-backed baseline | `TrainerConfig.discount_factor` |
| batch size | `128` | paper-backed baseline | `TrainerConfig.batch_size` |
| replay capacity | `50,000` | reproduction assumption | `ReplayBuffer(capacity)` |
| epsilon schedule | `1.0 → 0.01`, decay `7000` episodes | reproduction assumption | `TrainerConfig` / `epsilon()` |
| target update period | every `50` episodes, hard copy | reproduction assumption | `sync_targets()` |
| objective weights | `[0.5, 0.3, 0.2]` | paper-backed baseline / Table II default | config + scalarization |
| slot duration | `1 s` | paper-backed baseline | `StepConfig.slot_duration_s` |
| episode duration | `10 s` | paper-backed baseline | `StepConfig.episode_duration_s` |
| steps per episode | `10` | 由 `10 / 1` 推得 | `StepConfig.steps_per_episode` |
| training episodes | `9000` | paper-backed baseline | `TrainerConfig.episodes` |
| evaluation seeds | `[100, 200, 300, 400, 500]` | reproduction assumption | `seed_and_rng_policy` |
| checkpoint rule | primary=`final-episode-policy`; secondary=`best-weighted-reward-on-eval` | reproduction assumption | `checkpoint_selection_rule` |
| topology handling | `per-topology-retrain` | reproduction assumption | `ASSUME-MODQN-REP-011` |
| action catalog order | `satellite-major, beam-minor` | reproduction assumption | `action_masking_semantics` |
| baseline mask mode | `satellite-visible-all-beams` | reproduction assumption | `ActionMask` / `StepConfig` |
| baseline ground point | `(0°, 0°)` | reproduction assumption | `ground_point` |
| baseline user scatter | `uniform-circular`, radius `50 km` | reproduction assumption | `user_scatter_radius` |
| baseline mobility | deterministic heading stride | reproduction assumption | `user_heading_stride` / `_move_users()` |

表中的 baseline 值與規則整理自 paper-envelope config、resolved-run template、assumption register、trainer spec 與 state/environment code；其中 `state dimension = 5 × L × K`、`total actions = L × K` 是由實作直接推得，而不是文件另外寫死。fileciteturn13file0 fileciteturn14file0 fileciteturn12file0 fileciteturn46file0 fileciteturn36file0 fileciteturn32file0 fileciteturn37file0

| 概念 | 檔案路徑 | 主要 class / function | 實際責任 |
|---|---|---|---|
| environment 主體 | `src/modqn_paper_reproduction/env/step.py` | `StepEnvironment`, `StepConfig`, `_build_states_and_masks()`, `_compute_rewards()` | reset、step、state、mask、reward、done |
| orbit / visibility | `src/modqn_paper_reproduction/env/orbit.py` | `OrbitConfig`, `OrbitProxy`, `all_visibility()`, `visible_satellites()` | 衛星位置、slant range、elevation 可視性 |
| beam 幾何 | `src/modqn_paper_reproduction/env/beam.py` | `BeamConfig`, `BeamPattern`, `nearest_beam()`, `beam_centers_ground()` | 7-beam hex layout、beam center、最近 beam |
| channel 模型 | `src/modqn_paper_reproduction/env/channel.py` | `ChannelConfig`, `compute_channel()`, `compute_path_loss()` | FSPL、atmospheric factor、Rician fading、SNR |
| state encoding | `src/modqn_paper_reproduction/runtime/state_encoding.py` | `encode_state()`, `state_dim_for()` | 把符號 state 轉成固定長度向量 |
| Q network | `src/modqn_paper_reproduction/runtime/q_network.py` | `DQNNetwork` | 輸入 state、輸出各 action 的 Q values |
| replay buffer | `src/modqn_paper_reproduction/runtime/replay_buffer.py` | `ReplayBuffer.push()`, `ReplayBuffer.sample()` | FIFO 經驗儲存與 uniform sampling |
| trainer 規格 | `src/modqn_paper_reproduction/runtime/trainer_spec.py` | `TrainerConfig`, `EpisodeLog`, `EvalSummary` | 所有 training hyperparameters 與 log 型別 |
| objective 數學 | `src/modqn_paper_reproduction/runtime/objective_math.py` | `scalarize_objectives()`, `apply_reward_calibration()` | 三目標標量化與可選 reward calibration |
| MODQN trainer | `src/modqn_paper_reproduction/algorithms/modqn.py` | `MODQNTrainer` | 三個 Q nets、action selection、Bellman update、eval、checkpoint |
| scalar DQN comparator | `src/modqn_paper_reproduction/algorithms/dqn_scalar.py` | `ScalarDQNTrainer`, `ScalarDQNPolicyConfig` | `DQN_scalar` 與 `DQN_throughput` 的單網路 comparator |
| RSS comparator | `src/modqn_paper_reproduction/baselines/rss_max.py` | `evaluate_rss_max()` | 不訓練、直接選最大 channel quality |
| config 載入 | `src/modqn_paper_reproduction/config_loader.py` | `load_training_yaml()`, `build_environment()`, `build_trainer_config()` | 驗證 config 角色、建立 env 與 trainer |
| train orchestration | `src/modqn_paper_reproduction/cli.py`、`src/modqn_paper_reproduction/orchestration/train_main.py`、`scripts/train_modqn.py` | `train_main()`, `run_train_command()` | CLI 入口、訓練執行、log/checkpoint/metadata 輸出 |
| sweeps / evaluation | `src/modqn_paper_reproduction/sweeps.py` | `run_table_ii()`, `run_figure_suite()` | Table II 與 Fig. 3–6 比較流程 |
| artifact export | `src/modqn_paper_reproduction/export/replay_bundle.py` | `export_replay_bundle()` | replay-complete bundle 與 timeline 匯出 |

這張表也揭露兩個很重要的「不要腦補 repo 裡不存在的東西」的事實。第一，repo 並沒有獨立的 `dqn_throughput.py`；它是 `ScalarDQNTrainer + (1,0,0)` policy config 的 comparator 組態。第二，repo 的 MODQN 並沒有 paper pseudocode 那種三個獨立 replay memories，而是共享一個 `ReplayBuffer` 存三維 reward 向量。這兩點都直接影響你在簡報時要怎麼描述「論文 → 程式」之間的落差與等價實作。fileciteturn33file0 fileciteturn41file0 fileciteturn32file0 fileciteturn34file0 fileciteturn29file0 fileciteturn31file0 fileciteturn48file0

### 結果邊界與研究限制

repo 目前已完成的東西其實很多。README 與 reproduction-status 都確認：baseline 環境、MODQN trainer、DQN comparators、RSS baseline、checkpointing、resume、best-eval checkpoint、Table II、Fig. 3–6、CSV/JSON/PNG artifact、以及 Phase 03A/03B 的 replay bundle 匯出全都已經存在。從工程可執行性與 downstream comparison 的角度看，它完全有資格被稱作一個 working disclosed comparison baseline。fileciteturn7file0 fileciteturn44file0 fileciteturn48file0

但它之所以只能叫 disclosed comparison baseline，而不能叫完整 paper-faithful reproduction，原因也非常清楚。第一，assumption register 仍保留若干重要 assumption，例如 STK trace/import path 仍是 open，orbit shell layout、beam geometry、phi1/phi2、action masking、state encoding、seed protocol 等等都是顯式 reproduction assumption。第二，repo 自己承認 baseline scenario 其實與論文描述的 `200 km × 90 km`、`(40°N, 116°E)`、`random wandering` 並不相同；這些是到 Phase 01B 才被當成 follow-on correction，而不是 baseline 既有 faithful surface。第三，即便做了 scenario correction、comparator protocol probe、beam-aware follow-on 與 atmospheric-sign audit，也都還沒形成足夠強的新 claim，反而多數都以 negative result 或 bounded follow-on 的形式收束。fileciteturn12file0 fileciteturn43file0 fileciteturn15file0 fileciteturn16file0

限制大致可以分成四層。第一層是**result surface 的 near-tie**。closeout/status 文件顯示 Table II 與 Fig. 3–6 大多仍近乎 tie，差異主要集中在 `r2` handover，而 `r1` 和 `r3` 的跨方法分離度不夠；這使得「MODQN 明顯優於 comparators」的論文式結論，在 reproduction baseline 上還站不太穩。第二層是**long-run collapse**。`run-9000` 被明確記錄出現 late-training collapse：中段 checkpoint 比 final checkpoint 更好，吞吐與 load-balance 在後段訓練惡化，而 handover 似乎獨自改善。第三層是**reward dominance**。`StepEnvironment` 會在 reset 時輸出 reward scale diagnostics，而且 atmospheric-sign 審查也指出，修正 atmospheric sign 後 `r1` 對 `r2` 的優勢仍在數百倍量級，reward geometry 並沒有真正被修好。第四層是**beam semantics 壓縮**：baseline 裡同衛星 beams 共用同一組 slant-range-based SNR，再加上 `satellite-visible-all-beams`，導致 Phase 01E 審核出 pervasive 的 valid-mask collapse 與 channel-value collapse。fileciteturn44file0 fileciteturn15file0 fileciteturn37file0 fileciteturn17file0 fileciteturn19file0

Phase 01F 的 beam-aware follow-on 很有價值，但它也沒有授權 repo 把 baseline silently 替換掉。01F 的 bounded pilot 顯示 `nearest-beam-per-visible-satellite` 的確能移除 beam collapse，並在 bounded held-out surface 上大幅改善 throughput 與 load-balance；可是 20 episodes 和 200 episodes 在 beam-aware branch 裡得到幾乎相同的 held-out best-eval surface，表示**語義修正本身有用，但單純延長訓練還沒有被證明值得**。而 01G 也進一步說明 corrected lossy atmospheric sign 主要只是把 throughput-related magnitudes 稍微縮小，沒有在審核過的 replay traces 上改變動作。因此 repo 現在最合理的結論不是「再跑久一點就會重現論文」，而是 README 與 current-direction memo 的那個版本：**先停在這裡、保留 frozen baseline 作為 disclosure authority，只在有新科學問題時再開新的實驗 surface。**fileciteturn18file0 fileciteturn19file0 fileciteturn16file0 fileciteturn7file0

如果你要把這段變成口頭報告的最後一頁，我會建議你用下面這個說法收尾：  
**這個專案最成功的地方，不是已經完美重現論文數值，而是把「論文規格、缺失資訊、工程假設、訓練流程、比較 protocol、負面結果、claim boundary」全部拆開來公開化。** 從研究誠實性來看，這其實比只展示一組漂亮最終曲線更有價值。fileciteturn7file0 fileciteturn15file0 fileciteturn16file0

## 簡報濃縮版

**投影片：專案要回答的真正問題**

- 不是只問「MODQN 能不能跑」
- 而是問「論文如何被拆成可執行的 environment、trainer、sweeps、artifact」
- 核心主軸是從 paper spec 走到可重跑 reproduction system
- 最終成果是 disclosed comparison baseline，而非完整 faithful reproduction

**講者說明**：這份專案的價值，在於它把論文問題拆成一個完整可執行 surface：先定 authority order，再把 config、environment、training、evaluation、export 逐步落地。repo 自己也很誠實地把目前 claim boundary 定義為 comparison baseline，而不是宣稱數值已完整對齊原論文。fileciteturn7file0 fileciteturn11file0 fileciteturn44file0

**投影片：為什麼不是先蒐集 supervised dataset**

- DQN 沒有現成標註好的 beam label
- 真正的監督訊號來自環境互動後的 reward 與 next state
- handover 問題先被建模成 MOMDP
- 沒有 environment，就沒有 transition，也沒有 Bellman target

**講者說明**：這不是 image classification 類型的任務，不能先下載一包 `(x, y)` 資料集。論文先把問題轉成 MOMDP；演算法再透過「state → action → reward → next_state」的互動來產生訓練樣本。也因此，環境一定要先存在，DQN 才有東西可以學。fileciteturn0file0 citeturn0search0 fileciteturn32file0

**投影片：從論文到 repo 的開發順序**

- 先建立 authority order 與 assumption register
- 再區分 paper-envelope config 與 resolved-run config
- 接著實作 environment、trainer、checkpoint、sweeps
- 之後才開 scenario correction、beam semantics、atmospheric-sign 等 follow-on

**講者說明**：repo 不是先寫網路、再補文件，而是先把哪些是 paper-backed、哪些是 reproduction assumption 分清楚。這也是為什麼 `paper-envelope` 不能直接拿來訓練，必須先進 resolved-run surface。後續的 01B、01C、01E、01F、01G 都是顯式 follow-on，而不是偷偷改 baseline。fileciteturn11file0 fileciteturn12file0 fileciteturn30file0 fileciteturn15file0 fileciteturn16file0

**投影片：論文系統模型如何變成 environment**

- state = access vector + channel quality + beam offsets + beam loads
- action = 選一個 global beam index
- reward = throughput、handover、load balance 三項
- transition 由 orbit、beam、channel、mobility 與 assignment 更新共同形成

**講者說明**：`StepEnvironment` 是整個專案的核心。它把論文中的符號式 state 落成 `UserState`，把可行 action 落成 `ActionMask`，把三個 reward 分量落成 `RewardComponents`。每一個 step 都會更新時間、移動 user、套用 beam action、重建 state 與 rewards，這就是 DQN 真正學習的世界模型。fileciteturn37file0 fileciteturn38file0 fileciteturn39file0 fileciteturn40file0

**投影片：資料如何由互動產生**

- 每個 episode 有 10 個 slots
- 每個 slot 100 位 users 各產生一筆 per-user transition
- baseline 每個 episode 最多約 1000 筆樣本
- replay buffer 把這些經驗變成可抽樣的訓練資料池

**講者說明**：在這個 baseline 裡，資料不是先存成 csv 再訓練，而是由 agent-environment interaction 現場產出。repo 的實際 transition 比教科書版多兩個欄位：除了 `(state, action, reward, next_state, done)`，還額外存 `mask` 與 `next_mask`，因為 Bellman update 需要知道下個狀態哪些 beams 合法。fileciteturn37file0 fileciteturn34file0 fileciteturn32file0

**投影片：Q value 與 Q network 在這個專案中的意義**

- Q(s,a) = 選某個 beam 後的長期累積報酬估計
- baseline 有 28 個 actions，所以 Q network 輸出 28 個 values
- state dimension 是 140，來自 `5 × 28`
- invalid actions 會先被 mask 掉，再做 argmax

**講者說明**：這裡的 Q network 不是做分類，它是對每個 beam action 估計長期價值。因為 state encoding 對每個 beam 提供 5 個特徵，4 顆衛星乘 7 beams 就得到 state_dim=140；而 action_dim 就是 28。decision time 的關鍵在 action mask，不然網路可能會挑到不可視 beam。fileciteturn36file0 fileciteturn35file0 fileciteturn32file0 fileciteturn37file0

**投影片：MODQN 與一般 DQN 的差異**

- 一般 DQN：一個 scalar reward、一個 Q network
- MODQN：三個 parallel DQNs，分別學 r1、r2、r3
- 決策時再用 weights 做 scalarization
- baseline 預設權重是 `[0.5, 0.3, 0.2]`

**講者說明**：repo 的 `MODQNTrainer` 真的有三套 online/target nets 和三個 Adam optimizers。它不是先把 reward 壓成一個 scalar 再學，而是先保留 throughput、handover、load-balance 三個 value surfaces，最後在 action selection 才做線性加權。這就是 MODQN 與 `DQN_scalar` 最本質的差別。fileciteturn0file0 fileciteturn32file0 fileciteturn33file0 fileciteturn47file0

**投影片：Experience Replay 與 Target Network**

- replay buffer 容量 50,000，uniform random sampling
- batch size 128
- target network 每 50 episodes hard copy 一次
- 這兩個機制一起降低資料相關性與 target 漂移

**講者說明**：Experience replay 解的是「連續樣本太相關」；target network 解的是「Bellman target 跟著 online network 一起飄」。原始 DQN 就用這兩招來穩定學習，repo 也保存了這個標準結構，只是把 Bellman max 再和 action mask 結合起來，避免 target 落到非法 beam。citeturn0search0 fileciteturn12file0 fileciteturn34file0 fileciteturn32file0

**投影片：完整訓練流程**

- reset environment
- encode states → epsilon-greedy masked action selection
- env.step → rewards + next states
- push replay → sample mini-batch → Bellman update
- sync targets、evaluate、save final/best-eval checkpoints

**講者說明**：`run_train_command()` 先組 environment 與 trainer，再在 `MODQNTrainer.train()` 內反覆做 episode/slot 迴圈。每個 slot 都會把 100 個 users 的 transition 推進 replay，然後做一次 batch update；每到 evaluation cadence 與最後一個 episode，還會用 eval seed set 做 greedy evaluation，挑出 best-eval checkpoint。fileciteturn31file0 fileciteturn32file0 fileciteturn14file0

**投影片：關鍵參數與評估 protocol**

- hidden layers `[100, 50, 50]`，activation `tanh`
- optimizer `Adam`，lr `0.01`，discount `0.9`
- epsilon `1.0 → 0.01`，decay `7000` episodes
- evaluation seeds `[100, 200, 300, 400, 500]`
- checkpoint rule = final policy + best-eval policy

**講者說明**：這些值不是散落在程式裡的魔法常數，而是被 `resolved-run` config 與 `TrainerConfig` 收斂成顯式欄位。這點很重要，因為它讓你可以在口頭報告中清楚區分：哪些是論文明載，哪些是 repo 為了可執行性而做的 disclosed assumption。fileciteturn14file0 fileciteturn46file0 fileciteturn30file0

**投影片：結果與限制要怎麼誠實表達**

- repo 已完成可訓練 baseline、comparators、sweeps、export
- 但 method separation 仍弱，很多 surface 近乎 tie
- long-run collapse 與 reward dominance 仍存在
- beam-aware follow-on 有用，但不能拿來偷偷替換 frozen baseline

**講者說明**：最好的收尾不是說「我們完全重現了論文」，而是說「我們已經把論文變成一個可比較、可診斷、可重跑的 baseline system，並且把失敗邊界與 assumption 公開化」。這樣的說法更符合 repo 目前的證據，也更能展現你真的理解 DQN/MODQN 與 reproduction research 的差異。fileciteturn44file0 fileciteturn15file0 fileciteturn17file0 fileciteturn18file0 fileciteturn16file0