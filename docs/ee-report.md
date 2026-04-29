# `modqn-paper-reproduction` EE 路線下一步評估報告

## 2026-04-29 RA-EE-04 implementation update

RA-EE-04 已實作為 **fixed-association centralized power-allocation bounded
pilot**，不是 old EE-MODQN continuation。新的 config 是
`configs/ra-ee-04-bounded-power-allocator-control.resolved.yaml` 與
`configs/ra-ee-04-bounded-power-allocator-candidate.resolved.yaml`；artifact
寫在 `artifacts/ra-ee-04-bounded-power-allocator-control-pilot/`、
`artifacts/ra-ee-04-bounded-power-allocator-candidate-pilot/` 與
`artifacts/ra-ee-04-bounded-power-allocator-candidate-pilot/paired-comparison-vs-control/`。

這輪結果是 **PASS as bounded fixed-association power-allocation pilot only**。
candidate `safe-greedy-power-allocator` 在 `hold-current`、`random-valid`、
`spread-valid` 三個 non-collapsed fixed trajectories 上都讓
`denominator_varies_in_eval=true`，且 `EE_system` 對 matched fixed 1 W
control 分別提升 `+3.170352486871593`、`+1.9471815289625738`、
`+0.04925456347802992`，同時 p05 throughput ratio 分別為
`0.9762855249221921`、`0.9645767924959346`、`0.9932220296454757`，通過
QoS / budget / inactive-power guardrails。這只能支持「固定 association 的
集中式 power allocation pilot gate 已通過」；仍不能 claim learned
association、完整 RA-EE-MODQN effectiveness、HOBS optimizer、physical energy
saving、Catfish-EE，或 full paper-faithful reproduction。

## 2026-04-28 implementation update

Phase 03C-C 已把 Phase 03C-B 的 static / counterfactual power-MDP precondition
推到 bounded paired runtime pilot，但結果是 **BLOCKED**，不是 promote。新的
control / candidate config 分別是
`configs/ee-modqn-phase-03c-c-power-mdp-control.resolved.yaml` 與
`configs/ee-modqn-phase-03c-c-power-mdp-candidate.resolved.yaml`；candidate
啟用了 `runtime-ee-selector`，comparison artifact 寫在
`artifacts/ee-modqn-phase-03c-c-power-mdp-candidate-pilot/paired-comparison-vs-control/`。

最重要的結果是：candidate best-eval 仍然每一步都是 one active beam，
`selected_power_profile` 也完全 collapse 成 `fixed-low`，所以
`denominator_varies_in_eval=false`、active power 是單點 `0.5 W` 分布。
雖然 `EE_system` aggregate 對 control 有極小正 delta
(`+0.000270185470526485`)，但 p05 throughput 掉約 `50%`，且
throughput-vs-EE Pearson / Spearman 仍接近 `1.0`、same-policy rescore ranking
沒有改變。因此這輪實作強化了本報告原本的結論：不能 claim
EE-MODQN effectiveness，不能把 scalar reward 或 per-user EE credit 當主證據；
下一步若要重開，必須先解 one-beam collapse 和真正 denominator-sensitive
action / power coupling，而不是加長同一路線的 training。

## Short verdict

我的短結論是：**應暫停目前這條 EE-MODQN training 路線的擴大訓練與任何 effectiveness claim，先把問題重定義成「顯式 power-coupling 的新方法」再往下走。** 不是因為 EE 指標本身錯，而是因為目前專案裡的學習行為仍然是 **beam-only handover policy**，而評估時 learned policy 幾乎每一步都收斂成 **one active beam / total active power = 2.0 W**，導致 `EE_system` 在實際 evaluation 裡沒有被 policy 真正驅動；Phase 03A/03B 已經明確顯示 `denominator_varies_in_eval=false`、`EE_system` 與 control 打平、throughput–EE correlation 幾乎等於 1，且 repo 內部 gate 也已寫明「更多 episode 不是下一個 gate」。在這種狀態下，繼續做 reward normalization、r3 calibration、或直接把 Catfish 疊上去，都不太可能把「EE = throughput / 幾乎固定常數」這個核心問題變成可辯護的 EE learning。fileciteturn11file0L1-L1 fileciteturn12file0L1-L1 fileciteturn15file0L1-L1

更精準地說，**beam-only action 對 handover optimization 是足夠的，但對 HOBS-style energy efficiency optimization 通常不夠。** HOBS 論文把 EE 明確寫成總吞吐除以總 beam transmit power，且把優化變數直接定義成 power、association indicator 與 beam-training set，之後再把問題拆成「beam training / handover」與「power allocation」兩個子問題，並另外給出 dynamic power control。相對地，MODQN 論文可確認支持的是 throughput / handover / load-balance 的多目標 handover MDP，而不是顯式 power-control MDP。這表示如果你要在這個專案裡追求真正的 EE，不該再把它當成「只換 r1」的 Phase 03 延伸，而要把它視為 **new extension / method-design surface**。citeturn4view1turn4view2turn5search0turn5search1 fileciteturn9file0L1-L1 fileciteturn13file0L1-L1

## HOBS / MODQN papers actually support what

**HOBS paper 實際支持的，是「EE 必須與顯式 beam transmit power 綁在一起」。** 在 HOBS 的系統模型裡，SINR 式子直接包含 `P_{n,m}(t)`，文中也明確說 `P_{n,m}(t)` 是 beam `m` 在 satellite `n` 的 transmit power；其 EE 定義則是 `E_eff(t) = R_tot(t) / Σ_n Σ_m P_{n,m}(t)`。更重要的是，它的總優化問題把決策變數直接寫成 `P, a, Φ`，並施加 per-beam power constraint 與 per-satellite total power constraint；進入方法設計後，又把原問題拆成「beam training association」與「power allocation」兩個子問題，接著在第二個子問題裡給了 dynamic power control 更新規則。也就是說，HOBS 不是在做「只靠 beam selection 間接碰到 EE」，而是在做 **beam / handover + explicit power allocation** 的聯合設計。citeturn1view0turn4view1turn4view2

**MODQN paper 實際支持的，則是「multi-objective beam handover」，不是 energy-aware power control。** 可取得的 paper metadata / abstract 明確表示，它把 multi-beam LEO handover 問題建成 multi-objective optimization，再轉成 MOMDP，用 MODQN 來 jointly maximize throughput、minimize handover frequency、keep load balanced。repo 內被 freeze 的 baseline surface 也完全對齊這個邏輯：`r1 = throughput`、`r2 = handover penalty`、`r3 = load balance`，action 是 across all beams 的 one-hot beam selection。repo guardrails 也明說 original MODQN baseline 只能作為 disclosed comparison baseline，不可被靜默改寫成 EE or Catfish follow-on。換句話說，**MODQN paper 支持 beam-handover MORL；它不支持你把 HOBS 的 EE 直接視為原始 MODQN 的內生 objective，也不支持把 power-control 說成原 paper action space 的一部分。**citeturn5search0turn5search1 fileciteturn8file0L1-L1 fileciteturn9file0L1-L1 fileciteturn13file0L1-L1

放到你的八個問題裡，第一題的答案就很清楚了：**在 LEO / NTN / multi-beam literature 裡，handover 本身常可用 beam-only action 建模；但一旦目標是 EE 或更完整的 resource optimization，常見做法就會把 power、bandwidth、time-slot、beam profile 等資源顯式納入決策。** 這不只在 HOBS 如此；其他 LEO / multibeam 資源配置文獻也把 joint power/channel allocation、joint power/bandwidth allocation、time/power/frequency 三維資源配置，視為主要設計面。citeturn8view0turn10view0turn10view1turn6search0turn9view0

## Current implementation gap

目前專案的核心 gap，不是在「EE 公式有沒有 audit 到」，而是在 **policy 與 denominator 沒有真正耦合**。repo 的 environment 目前仍以 one-hot beam action 為主：state 由 access vector、channel quality、beam offsets、beam loads 構成；action 是 across all beams 的 one-hot beam selection；baseline reward surface 仍是 throughput / handover / load balance。雖然 `step.py` 已額外實作 `beam_transmit_power_w`、`active_beam_mask`，也在 reward component 裡加入 `r1_energy_efficiency_credit` 與 `r1_beam_power_efficiency_credit`，但那是 follow-on credit surface，不等於 paper-backed 的 system EE MDP。fileciteturn13file0L1-L1

Phase 02B 的確把 `EE_system(t) = sum_i R_i(t) / sum_active_beams P_b(t)` 做成了可 audit 的 runtime surface，而且這個 surface 在 repo 規格裡被明確標示為 `active-load-concave`、inactive beam `0 W`、units = linear W；它讓 `beam_transmit_power_w[b]` 也能進入 SINR numerator path。可是同一份 execution report 與 config 都反覆強調：這只是 **Phase 02B synthesized allocation proxy**，目的是讓 EE denominator 可被 defensibly audit，**不是 HOBS paper-backed optimizer，也不是已被論文支持的 power-control policy**。fileciteturn10file0L1-L1 fileciteturn16file0L1-L1

真正讓 EE route 卡死的，是 Phase 03A/03B 的 learned behavior。repo 的 Phase 03A diagnostic review 已經指出：environment 裡 `denominator exists in environment = True`，但 `learned policies exercise denominator variability = False`；active beam count distinct values 只有 `[1.0]`，total active beam power distinct values 只有 `[2.0]`，而且 action mask 並沒有強迫 single-beam，主要診斷反而是 `r3 load balance insufficient to prevent collapse` 與 `per-user EE degenerates under collapse = True`。Phase 03B 再加入 reward normalization、r3 calibration、以及 `per-user-beam-ee-credit = R_i(t)/P_b(t)` 之後，best-eval 仍然在 `EE_system`、throughput、served ratio、handover count、active beam count mean、total active beam power mean 全數與 control 打平，並保持 `denominator_varies_in_eval = false`；同時計算出的 throughput-vs-EE correlation 幾乎是 1，同 policy rescoring 也不改排名。這代表目前的 EE-MODQN 不像是在學「energy-aware tradeoff」，而像是在學「throughput 的重標度版本」。fileciteturn12file0L1-L1 fileciteturn11file0L1-L1

所以第三題「如何避免 EE 退化成 throughput / constant？」在這個專案脈絡下，答案非常具體：**不是先改 reward，而是先讓 evaluation 時的 denominator 成為 policy-sensitive quantity。** 如果 learned policy 在 greedy eval 裡永遠只使用單一 active beam、總 active power 永遠 2 W，那你把 r1 換成 `R_i/P_b`、`R_i/(P_b/N_b)`、甚至再做 normalization，最後都只是在同一個 collapse surface 上重寫 scoring。repo 自己的 validation gate 已經把這件事形式化：若 `ee_denominator_varies_in_eval` 不成立、one-beam collapse 未解除、`EE_system` 未改善、throughput–EE ranking 未分離，就不能 promote。fileciteturn15file0L1-L1 fileciteturn11file0L1-L1

## Candidate design routes

如果你現在要重設方法，我認為有四條路，但只有其中兩條值得當下一個正式 gate。

**第一條路是 hierarchical handover + power-control design，我認為這是最可辯護的主路線。** 這條路與 HOBS 最相容，因為 HOBS 本來就把問題拆成 association / beam training 與 power allocation 兩個子問題；對你這個 repo 來說，則可以保留 MODQN-style handover backbone，同時額外加入一個顯式 beam-power controller，在 frame-level 或 slot-level 更新 `P_b(t)`。這樣做的好處是：一方面保留 MODQN 的可比較性，另一方面又能讓 `EE_system` 的 denominator 真的由 controller / policy 控制；壞處是它已經超出原始 MODQN paper，必須明確標成 **new extension / method-design surface**。citeturn4view0turn4view2 fileciteturn9file0L1-L1

**第二條路是離散化 joint action，例如 `(beam_id, power_level)`、`(beam_id, Δpower sign)`、或 `(beam_id, power-codebook index)`。** 這條路的優點是可以維持 DQN / MODQN 家族的離散 action machinery，不必直接跳進 continuous control；缺點是 action space 會放大，而且 per-user action 與 per-beam power 的耦合要很小心設計，否則容易造成多 user 同步動作對同一 beam power 的 conflict。這類 joint discrete resource action 在 LEO / multibeam RL 文獻裡並不罕見，例如 joint channel-power allocation 的 DQN 類方法直接把 action 定義成 `{m, p}`，其中 `m` 是 channel、`p` 是 power，state 則包含 beam / traffic / resource occupancy。若你堅持留在 MODQN 家族，這是最實務的 power-control MDP 版本，但同樣必須標成 **new extension**，不能說是 HOBS 或 MODQN 原論文已支持的原生 action。citeturn10view0turn10view1turn10view2

**第三條路是 profile / codebook action。** 也就是不讓 agent 直接輸出所有 beam 的連續 power，而是從一小組預先定義的 beam-power profile、active-beam budget、spread / concentrate 模式、或 time–power–frequency template 中擇一。這條路在樣本效率上通常比 fully continuous 好，而且與 multibeam / beam-hopping literature 裡的三維資源配置思維一致；但它抽象層更高，跟原始 MODQN 的距離也更大，因此更適合作為後續 method-design surface，而不是你現在的最小下一步。citeturn6search0turn9view0

**第四條路是維持 beam-only action，只重做 reward 或 replay。這條路我不建議當主路線。** 因為 repo 已經做過最接近這個方向的 Phase 03A/03B：reward normalization、r3 calibration、per-user beam-power EE credit 都進過場，但 one-beam collapse 仍然存在。這表示問題不是簡單的 reward scale mismatch，而是 action / state / denominator coupling 缺位。就第五題而言，**Catfish replay / intervention 單獨不太可能解這個 collapse**。Catfish 是 training strategy，不是 environment / action semantics；它也許可以改 exploration 或 sample efficiency，但不能憑空創造 `P_b(t)` 的可控性。repo 的 master plan 已明講：Phase 04 Catfish feasibility 可以作為 original MODQN reward branch 的獨立驗證，但不是 EE route 的 default next step；而 Phase 03B 也已明講，未來 EE route 需要的是 denominator-sensitive action/power design 或更強的 credit-assignment review，不是更多 episode，更不是直接把 Catfish 疊上去。fileciteturn11file0L1-L1 fileciteturn9file0L1-L1 fileciteturn12file0L1-L1

## Recommended next gate

我建議的下一個 gate 很明確：**暫停 EE-MODQN training promotion，先寫一份新的 power-coupled MDP / controller spec，通過 design review 後再做新的 bounded pilot。** 這其實就是第四題的答案：是，應先停，不要再把資源投入同一條 beam-only EE training 線上。repo 的 guardrails 與 master plan 本身就要求：baseline frozen、follow-on 用新 config / method family / artifact namespace，並且當 Phase 03 已證明「更多 episodes alone 不是 next gate」時，不能把持續訓練當成方法進展。fileciteturn8file0L1-L1 fileciteturn9file0L1-L1

這個設計 gate 至少要先把四件事情寫死。第一，**agent granularity**：你到底要做 per-user handover agent + frame-level power controller，還是 central controller；我建議前者，因為它最接近 HOBS 的分解。第二，**state contract**：不只要看 beam quality，還要顯式看 `beam_loads`、`active_beam_mask`、`beam_transmit_power_w`、剩餘 power budget、以及 QoS slack。第三，**action contract**：若你走 hierarchy，power controller 至少要能選 `ΔP_b` 或 codebook profile；若你走 joint discrete，則每個 handover action 都要帶 power token。第四，**evaluation contract**：主指標只能是 `EE_system = sum_i R_i / sum_active_beams P_b`；per-user EE credit 只能當 training signal 或 diagnostic，不能當 claim 主體。這四項沒有明寫好，就不應該重新進入 training。citeturn4view1turn4view2turn10view0turn10view1turn9view0 fileciteturn13file0L1-L1

對 Catfish 的定位也應一起寫清楚。我的建議是：**把 Catfish 暫時從 EE 修復路線中拿掉，僅保留為原始 MODQN objective 的獨立 Phase 04 feasibility branch。** 等到 power-coupled EE extension 已證明 `denominator_varies_in_eval=true`、且 `EE_system` 與 throughput ranking 已分離之後，再考慮 Catfish 對新 EE extension 是否有增益。否則 Catfish 的任何正向結果，都很容易只是對 throughput-like objective surface 的 optimization 改善，而不是對 energy-aware learning 的改善。fileciteturn9file0L1-L1 fileciteturn11file0L1-L1

## Minimal experiment

最小可辯護的下一個實驗，我不建議一開始就做「完整 joint RL」。我建議做一個**兩段式最小實驗**，先證明 denominator 真的可被控制，再證明學習是必要的。

第一段是 **power-sensitivity proof**。做法是先把 beam policy 固定住，用同一組 beam decisions 分別配上三種 power semantics：`fixed-2W baseline`、`Phase 02B active-load-concave proxy`、以及一個**明確可控的 power controller**。這個第三個 controller 可以非常小：例如每個 active beam 從 `{0.5, 1.0, 1.5, 2.0} W` 四個 level 中選一個，並遵守 per-satellite total power budget；或者仿照 HOBS 的 sign-flip power update 做一個 rule-based DPC。這一步的目標不是證明 RL，而是先看：**在 beam decisions 相同時，system EE ranking 會不會因 power control 而改變？活躍 beam 數 / total active power 會不會在 eval 中出現多樣性？throughput–EE correlation 會不會下降？** 如果連這一步都無法讓 EE 與 throughput 分離，那就不應該進入第二段。citeturn4view2turn9view0turn10view0 fileciteturn10file0L1-L1 fileciteturn11file0L1-L1

第二段才是 **bounded RL pilot**。若第一段通過，再用一個最小的 power-coupled agent 做 paired control vs candidate：兩邊用相同 seed set、相同 environment surface、相同 checkpoint rule、相同 eval cadence；差別只在 candidate 是否擁有 power action。若你想最小變動、又不離開 Q-learning family，我建議先做 **discrete joint action** 或 **hierarchical DQN + rule-based DPC**，不要馬上跳 continuous actor-critic。因為你眼前最需要回答的不是「哪種 RL 最強」，而是「一旦 action 對 power 有控制權，EE 是否不再退化成 throughput 的重標度？」citeturn10view0turn10view1turn9view0

這個最小實驗的觀測指標也要先固定。除了 scalar reward，你至少要記錄：`EE_system aggregate`、`EE_system step mean`、`active_beam_count_distribution`、`total_active_beam_power_w_distribution`、`denominator_varies_in_eval`、`all_evaluated_steps_one_active_beam`、`raw_throughput_mean`、`raw_throughput_low_p05`、`served_ratio`、`handover_count`、以及 throughput-vs-EE correlation 與 same-policy rescore ranking 是否改變。repo 現在的 phase03 validator 已經幫你把這些檢查項大致定義好了；下一個實驗不需要再發明新的成功條件，只要把它們移到新的 power-coupled design 上即可。fileciteturn15file0L1-L1

## Acceptance criteria

在任何新設計下，要 claim「EE-MODQN effective」之前，我認為至少要同時滿足以下 acceptance criteria，而且主體必須是 **system EE**，不是 per-user EE credit。

第一，**learned best-eval policy 必須真的讓 denominator 動起來。** 最低要求是 `denominator_varies_in_eval=true`，且 `all_evaluated_steps_one_active_beam=false`；更實際地說，你要看到 active beam count 與 total active beam power 在 evaluation seeds 上都不是單點分布。這是目前 Phase 03B 最明確沒過去的一條，也是所有後續 claim 的前置條件。fileciteturn11file0L1-L1 fileciteturn15file0L1-L1

第二，**`EE_system` 必須在 matched control 上有正向提升，而且不是以明顯 QoS 崩壞換來的。** repo 現有 gate 對 throughput / service / handover 已有 guardrail：low-p05 throughput 不應明顯惡化、served ratio 不應顯著下降、handover count 不應無理由暴增。若 EE 變好但 throughput tail、served ratio 或 handover 成本明顯變壞，那只代表你找到了一個差的能耗 tradeoff，不代表方法有效。fileciteturn15file0L1-L1

第三，**throughput 與 EE 的 ranking 必須產生可解釋的分離。** 也就是說，至少要滿足以下其中之一：throughput-vs-EE correlation 不再幾乎等於 1；或 same-policy rescoring 會改變方法排名；或在一些 seed / scenario 下，control 與 candidate 在 throughput 差不多時 EE_system 會分出高下。若排名永遠不變，那你頂多證明新 reward 是舊 reward 的 rescaling。fileciteturn11file0L1-L1 fileciteturn15file0L1-L1

第四，**成功依據不能只靠 scalar reward。** 這不只是你的限制，也是 repo guardrails 的硬規則。即便 candidate 在 scalarized reward 上贏，只要 `EE_system` 沒贏、或 denominator 沒變、或 per-user EE 增加但 system EE 沒變，就不能 promote。fileciteturn8file0L1-L1 fileciteturn15file0L1-L1

第五，**所有 power semantics 都必須有 provenance label。** 若仍沿用 Phase 02B power surface，claim 必須限制為「在 disclosed synthesized proxy 下成立」；若換成新的 power-control MDP 或 controller，則必須標為 **new extension / method-design surface**。只有當設計真的被 HOBS 或 MODQN 原論文明確支持時，才可以把它說成 paper-backed；目前這一點尚未成立。fileciteturn10file0L1-L1 fileciteturn16file0L1-L1

## Stop conditions and forbidden claims

接下來我會把 stop conditions 與 forbidden claims 一起寫，因為它們本質上是在回答「什麼情況下必須停、哪些話現在絕對不能說」。

應當立刻 stop 的情況有幾個。**只要新設計跑完後，best-eval 仍然是 `denominator_varies_in_eval=false`、仍然所有 evaluated steps 只有一個 active beam、或 throughput-vs-EE correlation 仍然近乎 1、或排名在 throughput-rescore / EE-rescore 下完全不變，就應停止 EE effectiveness 敘事，而不是再加 episode。** 同樣地，若看見的只是 scalar reward 上升、per-user EE credit 上升、或 Catfish 讓 sample efficiency 變好，但 `EE_system` 沒改善，那也應停。這些都不是 energy-aware learning 的證據。fileciteturn11file0L1-L1 fileciteturn12file0L1-L1 fileciteturn15file0L1-L1

目前仍必須禁止的 claims，我建議你明文列成專案 guardrail。**不能 claim full paper-faithful reproduction；不能把 per-user EE credit 當成 system EE；不能用 scalar reward alone 當成功依據；不能把 Phase 02B synthetic `P_b(t)` 說成 HOBS paper-backed optimizer；不能 claim current Phase 03 / 03A / 03B 已證明 EE-MODQN effective；也不能 claim Catfish replay / intervention 單獨解掉了 EE collapse。** 若你接下來採用任何顯式 power-control MDP、hierarchical controller、joint discrete action、或 profile codebook，都應標成 **new extension / method-design surface**，除非未來能從原始 HOBS / MODQN text 找到更直接的支持。fileciteturn8file0L1-L1 fileciteturn9file0L1-L1 fileciteturn10file0L1-L1 fileciteturn17file0L1-L1

最後補一個必要的限制說明：**我對 HOBS 的 EE formula、`P_b(t)` 與 dynamic power control 有直接文本依據；但對 MODQN 論文的 state / action / reward 細節，我能高信心確認的是 abstract-level 的三目標定義，而更細的 MDP surface 主要依賴 repo 內的 paper-linked implementation / SDD mapping。** 這不影響本報告的主結論，因為你眼前要解的不是「原 MODQN 到底有沒有 power action」，而是「目前 repo 的 EE route 是否已足夠可辯護」；答案仍然是否定的，且下一步最合理的是先做顯式 power-coupled redesign。citeturn5search0turn5search1 fileciteturn13file0L1-L1
