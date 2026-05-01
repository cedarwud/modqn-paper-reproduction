  直接讀過的本地檔案：

  - system-model-refs/system-model-formulas.md（§2.12-2.18 SINR/throughput、§3.1-3.16 power/energy/EE）
  - system-model-refs/system-model-derivation.md（§2 SINR derivation、§5 power model finalization）
  - system-model-refs/simulator-parameter-spec.md（§4 power、§6 energy、§9 GAP-1~9）
  - system-model-refs/simulator-parameter-provenance-inventory.md（assumption-backed 清單、Round 4/5 修正）
  - paper-catalog/paper-audit-report.md（C3/C4/C5 power/EE/HO 證據表）
  - paper-catalog/catalog/PAP-2024-HOBS.json、PAP-2017-BEAMCOORD-EE.json、PAP-2022-EESAT-RELIABLE.json、PAP-2025-EEBH-UPLINK.json
  - 補充 catalog records：PAP-2025-EAQL、PAP-2024-EAQL-DRL、PAP-2025-SMASH-MADQL、PAP-2025-BIPARTITE-HM、PAP-2024-MORL-MULTIBEAM、PAP-2025-MAAC-BHPOWER
  - 對應的 txt_all/*.txt 與 txt_layout_all/*.layout.txt（HOBS Eq.4-13、28-30；BEAMCOORD-EE §I/II/III/VII；EESAT §2.2；EEBH-UPLINK §II-C）

  四篇主檔的 catalog 摘要與我抽查的 PDF/text 內容沒有牴觸；以下所有 EE 公式陳述以原文 Eq. 編號為據。

  HOBS Signal / Throughput / EE Formula

  直接抄自 PAP-2024-HOBS PDF（VTC2024-Spring）、保留原作者記號：

  SINR — Eq.(4)
  γ_{n,m,k}(t) = P_{n,m}(t) · H_{n,m,k}(t) · G^T_{n,m,k} · G^R_{n,m,k} · a_{n,m,k}(t) / ( I^a_{n,m,k} + I^b_{n,m,k} + σ² )

  - I^a：同衛星其他 beam（Eq.5）
  - I^b：其他衛星 beam（Eq.6）
  - σ²：thermal noise

  Latency-aware throughput — Eq.(9)
  R_{n,m,k}(t) = (1 − C_n(t)/T_f) · (W_{n,m}/U_{n,m}(t)) · log₂(1 + γ_{n,m,k}(t))

  - C_n(t)：beam-training latency（Eq.7：C_n = |φ^LEO_n|·T_beam + Σ U_{n,m}·(T_fb+T_ack)）
  - W_{n,m}：beam bandwidth；U_{n,m}：服務 user 數（等分頻寬）

  Sum-rate — Eq.(11)
  R_tot(t) = Σ_n Σ_m Σ_k R_{n,m,k}(t)

  System EE — Eq.(13)
  E_eff(t) = R_tot(t) / Σ_{n∈N} Σ_{m∈M} P_{n,m}(t)

  Per-beam EE — Eq.(29)
  E^eff_{n,m}(t) = Σ_k R_{n,m,k}(t) / P_{n,m}(t)

  Power-control 變數與限制

  - DPC 增量法：P_{n,m}(t) = P_{n,m}(t−T_f) + ξ^P_{n,m}(t)（Eq.27）
  - 反向觸發（Eq.28）：若 E^eff_{n,m}(t) ≤ E^eff_{n,m}(t−T_f) → ξ^P 反號
  - SINR 守底（Eq.30）：若任一 user γ < γ_thr → ξ^P 取絕對值，強制升功率
  - (14g) 0 ≤ P_{n,m}(t) ≤ P_beam（per-beam cap）
  - (14h) 0 ≤ Σ_m P_{n,m}(t) ≤ P_max（per-satellite cap）
  - (14b) Throughput 下限 R_th；(14c-d) beam alignment accuracy ≥ δ；(14e-f) 總/個別 training latency ≤ ρ

  重要觀察

  - HOBS 分母只有 transmit power Σ P_{n,m}。沒有 circuit power、static power、active-beam overhead、handover energy。
  - Beam-training overhead 已透過 (1 − C_n/T_f) 進入 throughput numerator，不作為分母懲罰。
  - DPC 由 per-beam EE 驅動，最終報告用 system EE。
  - HOBS Table I 的 P_max=50 dBm 與 thesis 採用的 [S10] 10 dBW per beam / 13 dBW per sat 不同來源（GAP-9）。

  Other EE Power-Denominator Patterns

  ┌────────────────────────┬──────────────────────────────────────┬─────────────────────┬──────────────────────────────┬───────────────────────────────┐
  │        Pattern         │                 來源                 │      Numerator      │         Denominator          │            套用域             │
  ├────────────────────────┼──────────────────────────────────────┼─────────────────────┼──────────────────────────────┼───────────────────────────────┤
  │ Transmit-power-only    │ HOBS Eq.(13)/(29)                    │ R_tot               │ Σ_{n,m} P_{n,m}              │ LEO downlink、multi-beam、Ka  │
  ├────────────────────────┼──────────────────────────────────────┼─────────────────────┼──────────────────────────────┼───────────────────────────────┤
  │ (1/η)P_tx + P_CP +     │                                      │                     │ Σ‖w_k‖²/η + P_CP +           │                               │
  │ P_RD·δ(r)              │ BEAMCOORD-EE Eq.(5)-(10)             │ Σ_b R̃_b             │ P_RD·r^m，其中 P_CP =        │ terrestrial multicell MISO    │
  │                        │                                      │                     │ P_FIX+P_TC+P_CE+P_LP         │                               │
  ├────────────────────────┼──────────────────────────────────────┼─────────────────────┼──────────────────────────────┼───────────────────────────────┤
  │ 預期 HARQ 能量         │ EESAT-RELIABLE                       │ K_correct_bits      │ E[N_s · Σ_i P_i across L     │ 單衛星、單用戶、IR-HARQ；單   │
  │                        │ Eq.(11)/(15)/(21)/(23)               │                     │ rounds]                      │ beam 無干擾                   │
  ├────────────────────────┼──────────────────────────────────────┼─────────────────────┼──────────────────────────────┼───────────────────────────────┤
  │ (P_l + P_sys) per      │ EEBH-UPLINK Eq.(6)/(7)               │ C_{j,l,t}           │ P_l^t +                      │ uplink；P_l = TT TX；P_sys =  │
  │ terminal + EES weight  │                                      │                     │ P_sys（per-terminal）        │ TT 維護                       │
  ├────────────────────────┼──────────────────────────────────────┼─────────────────────┼──────────────────────────────┼───────────────────────────────┤
  │ 能量當 state /         │ EAQL-DRL P_c=P_a+ΣP_tt；E_max=20 Wh  │ （目標式為 weighted │                              │                               │
  │ blocking               │ / SMASH-MADQL                        │  blocking + HO −    │ 並未寫成 bits/J ratio        │ 衛星電量耗盡判 blocking       │
  │                        │ Pc=P_ant+ΣP_t；E_n=E₀−∫P_c dt        │ rate）              │                              │                               │
  ├────────────────────────┼──────────────────────────────────────┼─────────────────────┼──────────────────────────────┼───────────────────────────────┤
  │ Utility-form           │ EAQL Eq. utility/reward              │ ∫γ dt               │ 不是分母；λ·E_HO 直接相減    │ EE-aware HO；λ=0.2            │
  │ (cost-subtract)        │                                      │                     │                              │ sensitivity-optimal           │
  ├────────────────────────┼──────────────────────────────────────┼─────────────────────┼──────────────────────────────┼───────────────────────────────┤
  │ Cost form (HO penalty  │ BIPARTITE-HM C_S =                   │ （rate 間接）       │ cost 加總                    │ gateway DC scenario           │
  │ + RTT)                 │ Σ(ℓ_h+s+α₁E_h+α₂RTT_h)               │                     │                              │                               │
  └────────────────────────┴──────────────────────────────────────┴─────────────────────┴──────────────────────────────┴───────────────────────────────┘

  關鍵區辨：

  - BEAMCOORD-EE 的數值（P_RD=2.4 W/(Gbps)^m, m=1.2、η、P_FIX/TC/CE/LP）是 terrestrial MBS 值，不能轉移到 LEO。只有 結構 可借。
  - EESAT-RELIABLE 的分母是隨機期望能量（HARQ 失敗→重傳次數），跟「每秒 W」不同維度；單 beam，無干擾，與 HOBS 場景不相容。
  - EEBH-UPLINK 的 P_sys 是地面 TT 的維護功率，不是衛星電路功率；其 EE 是 uplink 場景。直接挪用會把語意錯置。
  - EAQL-DRL / SMASH-MADQL 並沒有把 EE 寫成顯式 bits/J；他們把能量當作 state / 約束。
  - 既有運行時已將 systemEeBitsPerJoule（active-TX-only）與 totalPowerW（assumption-backed broader proxy）刻意分開，這個切分必須在公式層級也保持。

  Consistency Requirements

  若 SINR 採 HOBS-style 分母 = (intra+inter+thermal noise)，分子含 p_{s,b}·h_{u←s,b}，則最低限度的 EE 一致性要求：

  1. 同一 power 變數：EE 分母必須消費 SINR numerator 用的同一個 p_{s,b}^t，不可加入和 SINR 完全脫鉤的偽功率（idle/off）。
  2. 同一活動語意：採用 §2.9 z_{s,b}^t 與 §2.11 closure；非活動 beam 透過 p_{s,b}=0 自動歸零，不需在 EE 分母再乘 z。
  3. 同一時間網格：R_u^t、p_{s,b}^t、E_HO 都用相同 Δt；EE 必為 time-summed bits / time-summed Joules。
  4. 線性域功率：所有 p_{s,b} 用 W；不在線性 EE 比值中混入 dBW（formulas.md §3.1 已固化此規則）。
  5. Power cap 一致：§3.2 SINR-inversion 的 required power 與 §3.4 z·P_beam,max 上限要在同一 EE 路徑可見；不能 EE 分母用一套上限、SINR 用另一套。
  6. 任何附加項皆需顯式宣告：P_c, 1/ρ, active-beam overhead, E_HO 都要逐項標 paper-backed / assumption / sensitivity。對應 §3.16 的 4
  列分隔（systemEeBitsPerJoule / totalPowerW / handover-aware-ee / utility-form-fallback）。

  Candidate EE Formula Versions

  四個正式候選 + 兩個明確不建議。

  ---
  V1 — HOBS-style transmit-power EE（最簡、最 paper-anchored）

  公式
  η_EE^{V1} = ( Σ_t Σ_u R_u^t Δt ) / ( Σ_t Σ_s Σ_b p_{s,b}^t Δt )

  - 分子定義：thesis 主稿 §2.18 R_u^t = W_u^t·log₂(1+γ_u^t)（served）/ 0（unserved）累加。等價於 HOBS Eq.(11) 改成 thesis (u,s,b,t) 記號後再乘 Δt。
  - 分母定義：所有衛星、所有 beam 的 transmit-power × Δt 累加。Inactive beam 因 z=0 → p=0 自動歸零，不必另寫。
  - 需要的參數：p_{s,b}^t（已是 SINR 變數）、W、N_0、SF/CL、G_T(θ)、Δt；若沿用 HOBS C_n 則需 T_beam/T_fb/T_ack/T_f。
  - paper-backed：HOBS Eq.(13) 公式骨架；P_beam,max=10 dBW、P_sat,max=13 dBW 來自 [S10]；W=100 MHz、Ka 28 GHz 來自 HOBS Table I；NF=9 dB / G_R=0 dBi 來自 TR
   38.811 Table 4.4-1；σ_SF/CL 來自 TR 38.811 Table 6.6.2-1/3。
  - assumption / sensitivity：本版本不需要；乾淨。
  - 優點：SINR numerator 與 EE denominator 用同一個 p_{s,b}，不會雙重記帳；reviewer 可直接對著 HOBS Eq.(13) 重做。
  - 風險：若 simulator policy 不真正 調整 p_{s,b}（恆設 P_beam,max），η_EE = (Σ R)/(K_active·P_beam,max·Δt) → 變成 throughput 線性比例；但這不是公式問題、是
   policy 設計問題。也不能直接懲罰「beam 雖空但仍開硬體」的場景。
  - 位置：Main metric。

  ---
  V2 — Composite simulated-system EE（structural borrowing from BEAMCOORD-EE）

  公式
  η_EE^{V2} = ( Σ_t Σ_u R_u^t Δt ) / ( Σ_t Σ_s P_tot,s^t Δt )，
  其中 P_tot,s^t = P_c,s + (1/ρ_s) · Σ_b p_{s,b}^t

  - 分子：與 V1 相同。
  - 分母：對齊 system-model-formulas.md §3.5 與 §3.13；採用 BEAMCOORD-EE Eq.(5) 的兩段式結構。
  - 需要的參數：p_{s,b}（paper-backed）、P_c,s（assumption）、ρ_s（assumption）。
  - paper-backed：分母 結構 來自 PAP-2017 Eq.(5)；BEAMCOORD-EE 主要貢獻就是這個形式。
  - assumption / sensitivity：P_c,s 與 ρ_s 沒有 LEO 來源（GAP-1 / GAP-2）；P_RD·δ(r) 故意省略（GAP-4 — terrestrial only）。
  - 優點：吻合 §3.13/§3.16 已建立的揭露框架；可進行「circuit power 多大時策略順序會翻轉」這類 sensitivity；仍與 HOBS SINR 共用同一 p_{s,b}。
  - 風險：在 GAP-1/GAP-2 關閉前，η_EE^{V2} 的 絕對值 不可與任何文獻 baseline 比；容易讓人把 runtime totalPowerW 直接套作 V2 分母（即把 assumption-backed
  數值悄悄升級成 paper-backed），這是 §9.1 governance 明確禁止的。
  - 位置：Sensitivity / Ablation only. 不上 main metric，直到 GAP-1/2 關閉。

  ---
  V3 — Literature-rich handover-aware EE（sensitivity-bracketed）

  公式
  η_EE,HO^{V3} = ( Σ_t Σ_u R_u^t Δt ) / ( Σ_t Σ_s P_tot,s^t Δt + E_HO,tot )，
  其中 E_HO,tot = Σ_t Σ_u δ_{u,HO}^t · E_{u,HO}

  - 分子：與 V1/V2 相同。
  - 分母：V2 分母 + 累積 per-event handover energy。對齊 §3.14。
  - 需要的參數：V2 全部 + E_{u,HO}。
  - paper-backed：HO event δ 來自 thesis closure + standard FSM；分母結構為 thesis-level 整合。
  - assumption / sensitivity：E_{u,HO} 沒有 corpus-wide LEO 數值；PAP-2025-EAQL 用 3 J 是 paper-specific assumption；PAP-2019-HO-OVERHEAD 是 LTE signaling
  cost，不可轉移。
  - 優點：若需重現 EAQL 風格的 EE-vs-HO trade-off，這版本最貼。
  - 風險：分母同時依賴 P_c, ρ, E_HO 三個 assumption-backed 量；若 E_HO 掃 0 與大值會讓比值大幅波動，讀者容易誤讀；最容易讓 3 J 默默升級為 universal
  default。
  - 位置：Sensitivity sweep only. 不上 main，也不應作為 ablation 唯一行；要顯式寫出 E_{u,HO} 取值範圍。

  ---
  V4 — Utility-form fallback objective（safer when denominator-sensitivity 風險高）

  公式
  J^{V4} = ( Σ_t Σ_u R_u^t Δt ) − λ_HO · E_HO,tot

  對齊 system-model-formulas.md §3.15。

  - 分子型項：總比特累積。
  - 成本項：λ_HO · E_HO,tot 直接相減，不放分母。
  - 需要的參數：E_{u,HO}、λ_HO。
  - paper-backed：λ_HO = 0.2 來自 PAP-2025-EAQL [S9] sensitivity-optimal — corpus 中唯一有「最佳值」defended 的 EE-related 權重。
  - assumption / sensitivity：E_{u,HO} 同 V3。
  - 優點：避免「用一個包含 assumption 項的和當分母」的 trap；λ_HO 是少數 paper-backed 權重；仍能懲罰 HO 事件；當 V2/V3 分母敘事不安全時，這是最安全的 main
  objective 候選。
  - 風險：嚴格來說不是 bits/J（單位是 bits / scenario）；要明說是 utility，不是 EE；跨 traffic regime 比較較困難。
  - 位置：Fallback main objective；或與 V1 並列為 secondary headline。

  ---
  Vbad — EEBH-UPLINK 風 (P_l + P_sys) WSEE — 不建議

  - 那個 (P_l + P_sys) 是 uplink、地面終端側 的功率；本研究 SINR 以 HOBS downlink 為主，主導功率是衛星 beam transmit power，不是用戶 TX。
  - 強行把 P_sys 換成「衛星 circuit」其實是 V2 的偽裝，並讓命名誤導讀者。
  - λ_l EES 加權是另一條研究軸（heterogeneous-terminal），不在 HOBS-style scope 內。

  Vbad-2 — EESAT-RELIABLE IR-HARQ 期望能量 EE — 不建議

  - 它是 HARQ-process-level EE：分母是 HARQ 重傳隨機過程的期望能量。
  - 多 beam multi-LEO HOBS 預設沒有 HARQ retransmission；強行加會是 捏造 場景語意。
  - 單 beam 單用戶、無 inter-beam 干擾，與 HOBS 骨架物理不相容。

  Recommended Ranking

  依推薦強度（可單獨辯護程度）：

  1. V1（HOBS-style transmit-power EE）— 主指標。 直接 paper anchor，分母與 SINR numerator 共用 p_{s,b}，無 assumption-backed 項。
  2. V4（utility-form J = Σ R − λ·E_HO）— 安全替代或共同 headline。 唯一 paper-backed 權重 λ=0.2 主動使用；避開分母敏感問題。
  3. V2（composite EE）— 用作 sensitivity row。 顯式聲明 P_c, ρ 為 scenario assumption；可展示電路功率對策略排序的影響。
  4. V3（handover-aware EE）— sensitivity sweep only。 E_HO ∈ {1, 3, 5, 10} J 等明確掃描，永不上 main。
  5. （不建議）Vbad / Vbad-2 — 場景錯位。

  Main Metric Recommendation

  主指標採 V1：

  η_EE = ( Σ_t Σ_u R_u^t Δt ) / ( Σ_t Σ_s Σ_b p_{s,b}^t Δt )

  採 HOBS Eq.(13) 完全對齊的解讀：分子 = 所有 served user 的 sum-rate × Δt；分母 = 所有衛星所有 beam 的 active TX power × Δt。idle / off beams 因 p_{s,b}=0
  自動進入分母為 0。這正是運行時 systemEeBitsPerJoule 已鎖定的「active-TX-power-oriented EE」。

  若 thesis 同時要前置 handover-energy 議題，把 V4（utility-form J）放為 secondary headline；V2/V3 揭露為 sensitivity 行。不要 把 V2/V3 當 main，原因見
  GAP-1/2/3。

  Ablation / Sensitivity Recommendations

  - Power-cap 敏感：P_beam,max ∈ {7, 10, 13} dBW 掃描 — 同時涵蓋 [S10] 中心、低預算、與 HOBS 50 dBm 偏向（GAP-9 對應路徑）。
  - Per-beam vs system EE 並列：HOBS Eq.(29) 與 Eq.(13) 同時報告；某些 policy 提升 per-beam EE 但拖低 aggregate。
  - Beam-training overhead 開關：(1 − C_n/T_f) 開/關，觀察 EE-claim 排序是否被 training overhead 帶動。
  - Circuit-power 敏感（V2）：P_c,s ∈ {0, 50, 100, 200} W，明確標 “scenario assumption”；P_c=0 row 即可恢復 V1。
  - PA 效率敏感（V2）：ρ_s ∈ {0.3, 0.5, 0.7, 0.9}；不可宣稱任何單一值是 paper-backed。
  - HO-energy 敏感（V3 / V4）：E_{u,HO} ∈ {1, 3, 5, 10} J；3 J 那行明標為 PAP-2025-EAQL paper-specific 假設，不是 universal default。
  - λ_HO 敏感（V4）：λ ∈ {0.1, 0.2, 0.5}；強調 0.2 為 EAQL paper-backed optimum。
  - FRF / K_active ablation：V1 在 FR1 vs FR3、K_active ∈ {3, 10, 12, 19} 下報告，使 EE 數字回掛到干擾結構而非分母調整。

  這些 ablation 都和 EE 定義本身正交 — 它們是讓 EE 指標 有資訊量 的物理/政策變數。

  Parameters: Paper-Backed vs Assumption

  EE-相關參數一覽：

  ┌───────────────────────────────────────┬───────────────────────────┬───────────────────────────────────────────────────────────────┐
  │               Parameter               │          Status           │                            Source                             │
  ├───────────────────────────────────────┼───────────────────────────┼───────────────────────────────────────────────────────────────┤
  │ p_{s,b}^t                             │ derived from SINR/policy  │ HOBS Eq.(4)/(28)                                              │
  ├───────────────────────────────────────┼───────────────────────────┼───────────────────────────────────────────────────────────────┤
  │ P_beam,max = 10 dBW                   │ paper-backed              │ PAP-2025-MAAC-BHPOWER [S10]                                   │
  ├───────────────────────────────────────┼───────────────────────────┼───────────────────────────────────────────────────────────────┤
  │ P_sat,max = 13 dBW                    │ paper-backed              │ PAP-2025-MAAC-BHPOWER [S10]                                   │
  ├───────────────────────────────────────┼───────────────────────────┼───────────────────────────────────────────────────────────────┤
  │ W (beam BW) = 100 MHz                 │ paper-backed              │ HOBS Table I                                                  │
  ├───────────────────────────────────────┼───────────────────────────┼───────────────────────────────────────────────────────────────┤
  │ σ²/N_0 (noise PSD)                    │ standard                  │ TR 38.811 + 物理                                              │
  ├───────────────────────────────────────┼───────────────────────────┼───────────────────────────────────────────────────────────────┤
  │ NF=9 dB, G_R=0 dBi                    │ standard                  │ TR 38.811 Table 4.4-1                                         │
  ├───────────────────────────────────────┼───────────────────────────┼───────────────────────────────────────────────────────────────┤
  │ L_impl=2.5 dB（feeder+pointing）      │ paper-backed              │ PAP-2022-SENSORS-BH Table 3                                   │
  ├───────────────────────────────────────┼───────────────────────────┼───────────────────────────────────────────────────────────────┤
  │ T_f, T_beam, T_fb, T_ack, C_n         │ paper-backed              │ HOBS Table I + Eq.(7)                                         │
  ├───────────────────────────────────────┼───────────────────────────┼───────────────────────────────────────────────────────────────┤
  │ γ_thr = 10 dB                         │ paper-backed              │ HOBS handover threshold                                       │
  ├───────────────────────────────────────┼───────────────────────────┼───────────────────────────────────────────────────────────────┤
  │ λ_HO = 0.2                            │ paper-backed              │ PAP-2025-EAQL [S9]                                            │
  ├───────────────────────────────────────┼───────────────────────────┼───────────────────────────────────────────────────────────────┤
  │ P_c,s（衛星 circuit）                 │ assumption                │ GAP-2；無 LEO 來源                                            │
  ├───────────────────────────────────────┼───────────────────────────┼───────────────────────────────────────────────────────────────┤
  │ ρ_s（PA 效率）                        │ assumption                │ GAP-1；無 LEO 來源                                            │
  ├───────────────────────────────────────┼───────────────────────────┼───────────────────────────────────────────────────────────────┤
  │ P_RD·δ(r)（rate-dependent）           │ structural-only           │ PAP-2017-BEAMCOORD-EE 是 terrestrial；數值不可轉 LEO（GAP-4） │
  ├───────────────────────────────────────┼───────────────────────────┼───────────────────────────────────────────────────────────────┤
  │ E_{u,HO}（per-HO energy）             │ scenario / sensitivity    │ GAP-3；EAQL 3 J 是 paper-specific                             │
  ├───────────────────────────────────────┼───────────────────────────┼───────────────────────────────────────────────────────────────┤
  │ Active/idle/off beam power 20/5/0.1 W │ assumption                │ GAP-5；SMASH-MADQL locator 未驗證                             │
  ├───────────────────────────────────────┼───────────────────────────┼───────────────────────────────────────────────────────────────┤
  │ totalPowerW runtime field             │ assumption-backed proxy   │ §3.16；不是 V1/V2 main 分母                                   │
  ├───────────────────────────────────────┼───────────────────────────┼───────────────────────────────────────────────────────────────┤
  │ HOBS 50 dBm total TX                  │ source-paper context only │ GAP-9；不採作 default cap                                     │
  └───────────────────────────────────────┴───────────────────────────┴───────────────────────────────────────────────────────────────┘

  Forbidden / Unsafe Claims

  下面這些不能寫：

  - ❌「我們的 EE 採用 PAP-2017-BEAMCOORD-EE 的 rate-dependent power model with P_RD=2.4 W/(Gbps)^m」— terrestrial 值，不能套 LEO。
  - ❌「per-handover energy 採 PAP-2025-EAQL 的 3 J 為預設」— 那是 paper-specific assumption。
  - ❌「EE 分母含衛星 circuit power，依某論文採 X W」— corpus 沒有任何 paper 給 LEO 衛星 circuit power 的數值。
  - ❌「HOBS 的 EE 分母含 idle-beam power」— 錯；HOBS Eq.(13) 只 sum active beams' TX power。
  - ❌「V2（含 P_c, ρ）是 paper-backed」— 只有 結構 從 terrestrial PAP-2017 借來。
  - ❌「runtime totalPowerW 就是 EE 分母」— 運行時已分離 systemEeBitsPerJoule（active-TX-only） 與 totalPowerW（assumption-backed proxy）。
  - ❌「採 EEBH-UPLINK 的 (P_l+P_sys) 為 EE 分母」— 那是 uplink terminal 側，不適用 downlink 衛星場景。
  - ❌「使用 HOBS 50 dBm 為 default power cap」— 與 thesis 採用的 [S10] 10/13 dBW 衝突；只能作為 source-paper 註腳或 sensitivity 路徑。
  │ Active/idle/off beam power 20/5/0.1 W │ assumption                │ GAP-5；SMASH-MADQL locator 未驗證                             │
  ├───────────────────────────────────────┼───────────────────────────┼───────────────────────────────────────────────────────────────┤
  ├───────────────────────────────────────┼───────────────────────────┼───────────────────────────────────────────────────────────────┤
  │ totalPowerW runtime field             │ assumption-backed proxy   │ §3.16；不是 V1/V2 main 分母                                   │
  ├───────────────────────────────────────┼───────────────────────────┼───────────────────────────────────────────────────────────────┤
  │ HOBS 50 dBm total TX                  │ source-paper context only │ GAP-9；不採作 default cap                                     │
  └───────────────────────────────────────┴───────────────────────────┴───────────────────────────────────────────────────────────────┘

  Forbidden / Unsafe Claims

  下面這些不能寫：

  - ❌「我們的 EE 採用 PAP-2017-BEAMCOORD-EE 的 rate-dependent power model with P_RD=2.4 W/(Gbps)^m」— terrestrial 值，不能套 LEO。
  - ❌「per-handover energy 採 PAP-2025-EAQL 的 3 J 為預設」— 那是 paper-specific assumption。
  - ❌「EE 分母含衛星 circuit power，依某論文採 X W」— corpus 沒有任何 paper 給 LEO 衛星 circuit power 的數值。
  - ❌「HOBS 的 EE 分母含 idle-beam power」— 錯；HOBS Eq.(13) 只 sum active beams' TX power。
  - ❌「V2（含 P_c, ρ）是 paper-backed」— 只有 結構 從 terrestrial PAP-2017 借來。
  - ❌「runtime totalPowerW 就是 EE 分母」— 運行時已分離 systemEeBitsPerJoule（active-TX-only） 與 totalPowerW（assumption-backed proxy）。
  - ❌「採 EEBH-UPLINK 的 (P_l+P_sys) 為 EE 分母」— 那是 uplink terminal 側，不適用 downlink 衛星場景。
  - ❌「使用 HOBS 50 dBm 為 default power cap」— 與 thesis 採用的 [S10] 10/13 dBW 衝突；只能作為 source-paper 註腳或 sensitivity 路徑。

  Questions / Assumptions

  下一輪要釐清的：

  1. 要不要保留 HOBS 的 (1 − C_n/T_f) beam-training overhead？ 留 → V1 分母單純就是 TX power；不留 → 要顯式註明已捨棄該因子，並接受跟 HOBS Eq.(13) 數值不可逐項對照。
  2. policy 是否實際做 per-beam power control？ 若 p_{s,b} 永遠頂在 P_beam,max，V1 變成 (Σ R)/(K_active·P_beam,max·Δt)，數字趨近 throughput 線性比例；公式沒問題，問題在 policy 設計。
  3. DAPS 雙活期是否雙計分母功率？ 若 UE 在 serving 與 target 兩衛星都被服務，兩個 p_{s,b}^t 都計入分母 — 這是 HOBS-style 的「直白」算法，要在 thesis 明文說明。
  4. 要不要 WSEE-style 多 UE 加權？ 若要，那是另一條研究軸（heterogeneous-terminal），不能默默地擴展 V1。
  5. EE 報告窗口為何？ per-second / per-pass / per-experiment？V1 維度上一致，但圖表必須鎖一個窗口。
  6. V2 是 per-satellite P_c 還是 fleet-wide 常數？ 兩種都 assumption-backed；per-satellite 對齊 §3.5，fleet-wide 比較簡單。

  PASS / NEEDS MORE EVIDENCE

  PASS — 對 V1/V2/V3/V4 四個候選的 結構：證據足以辯護彼此差別與選擇。V1 是 HOBS Eq.(13) 直接 anchor；V4 是 EAQL λ_HO=0.2 anchor；V2/V3 結構來自 system-model-formulas §3.13/§3.14 與 BEAMCOORD-EE §IV，可以列為
  sensitivity。

  NEEDS MORE EVIDENCE — 任何 V2/V3 的 數值 主張：

  - LEO 衛星 circuit power P_c,s（GAP-2）— 仍開
  - LEO PA 效率 ρ_s（GAP-1）— 仍開
  - LEO per-handover-event energy E_{u,HO}（GAP-3）— 只有 paper-specific 值；無轉移性 default
  - LEO rate-dependent baseband（GAP-4）— terrestrial only
  - Active/idle/off beam power 數值（GAP-5）— assumption-only

  在這些 GAP 透過獨立來源（Starlink FCC filings、ITU-R S.1528、ESA 衛星 power budget tech notes 等）關  即可恢復 V1。
  - PA 效率敏感（V2）：ρ_s ∈ {0.3, 0.5, 0.7, 0.9}；不可宣稱任何單一值是 paper-backed。
  - HO-energy 敏感（V3 / V4）：E_{u,HO} ∈ {1, 3, 5, 10} J；3 J 那行明標為 PAP-2025-EAQL paper-specific
   假設，不是 universal default。
  - λ_HO 敏感（V4）：λ ∈ {0.1, 0.2, 0.5}；強調 0.2 為 EAQL paper-backed optimum。
  - FRF / K_active ablation：V1 在 FR1 vs FR3、K_active ∈ {3, 10, 12, 19} 下報告，使 EE
  數字回掛到干擾結構而非分母調整。

  這些 ablation 都和 EE 定義本身正交 — 它們是讓 EE 指標 有資訊量 的物理/政策變數。

  Parameters: Paper-Backed vs Assumption

  EE-相關參數一覽：

  ┌────────────────────────────────┬──────────────────────┬───────────────────────────────────────┐
  │           Parameter            │        Status        │                Source                 │
  ├────────────────────────────────┼──────────────────────┼───────────────────────────────────────┤
  │ p_{s,b}^t                      │ derived from         │ HOBS Eq.(4)/(28)                      │
  │                                │ SINR/policy          │                                       │
  ├────────────────────────────────┼──────────────────────┼───────────────────────────────────────┤
  │ P_beam,max = 10 dBW            │ paper-backed         │ PAP-2025-MAAC-BHPOWER [S10]           │
  ├────────────────────────────────┼──────────────────────┼───────────────────────────────────────┤
  │ P_sat,max = 13 dBW             │ paper-backed         │ PAP-2025-MAAC-BHPOWER [S10]           │
  ├────────────────────────────────┼──────────────────────┼───────────────────────────────────────┤
  │ W (beam BW) = 100 MHz          │ paper-backed         │ HOBS Table I                          │
  ├────────────────────────────────┼──────────────────────┼───────────────────────────────────────┤
  │ σ²/N_0 (noise PSD)             │ standard             │ TR 38.811 + 物理                      │
  ├────────────────────────────────┼──────────────────────┼───────────────────────────────────────┤
  │ NF=9 dB, G_R=0 dBi             │ standard             │ TR 38.811 Table 4.4-1                 │
  ├────────────────────────────────┼──────────────────────┼───────────────────────────────────────┤
  │ L_impl=2.5                     │ paper-backed         │ PAP-2022-SENSORS-BH Table 3           │
  │ dB（feeder+pointing）          │                      │                                       │
  ├────────────────────────────────┼──────────────────────┼───────────────────────────────────────┤
  │ T_f, T_beam, T_fb, T_ack, C_n  │ paper-backed         │ HOBS Table I + Eq.(7)                 │
  ├────────────────────────────────┼──────────────────────┼───────────────────────────────────────┤
  │ γ_thr = 10 dB                  │ paper-backed         │ HOBS handover threshold               │
  ├────────────────────────────────┼──────────────────────┼───────────────────────────────────────┤
  │ λ_HO = 0.2                     │ paper-backed         │ PAP-2025-EAQL [S9]                    │
  ├────────────────────────────────┼──────────────────────┼───────────────────────────────────────┤
  │ P_c,s（衛星 circuit）          │ assumption           │ GAP-2；無 LEO 來源                    │
  ├────────────────────────────────┼──────────────────────┼───────────────────────────────────────┤
  │ ρ_s（PA 效率）                 │ assumption           │ GAP-1；無 LEO 來源                    │
  ├────────────────────────────────┼──────────────────────┼───────────────────────────────────────┤
  │ P_RD·δ(r)（rate-dependent）    │ structural-only      │ PAP-2017-BEAMCOORD-EE 是              │
  │                                │                      │ terrestrial；數值不可轉 LEO（GAP-4）  │
  ├────────────────────────────────┼──────────────────────┼───────────────────────────────────────┤
  │ E_{u,HO}（per-HO energy）      │ scenario /           │ GAP-3；EAQL 3 J 是 paper-specific     │
  │                                │ sensitivity          │                                       │
  ├────────────────────────────────┼──────────────────────┼───────────────────────────────────────┤
  │ Active/idle/off beam power     │ assumption           │ GAP-5；SMASH-MADQL locator 未驗證     │
  │ 20/5/0.1 W                     │                      │                                       │
  ├────────────────────────────────┼──────────────────────┼───────────────────────────────────────┤
  │ totalPowerW runtime field      │ assumption-backed    │ §3.16；不是 V1/V2 main 分母           │
  │                                │ proxy                │                                       │
  ├────────────────────────────────┼──────────────────────┼───────────────────────────────────────┤
  │ HOBS 50 dBm total TX           │ source-paper context │ GAP-9；不採作 default cap             │
  │                                │  only                │                                       │
  └────────────────────────────────┴──────────────────────┴───────────────────────────────────────┘

  Forbidden / Unsafe Claims

  下面這些不能寫：

  - ❌「我們的 EE 採用 PAP-2017-BEAMCOORD-EE 的 rate-dependent power model with P_RD=2.4 W/(Gbps)^m」—
   terrestrial 值，不能套 LEO。
  - ❌「per-handover energy 採 PAP-2025-EAQL 的 3 J 為預設」— 那是 paper-specific assumption。
  - ❌「EE 分母含衛星 circuit power，依某論文採 X W」— corpus 沒有任何 paper 給 LEO 衛星 circuit power
   的數值。
  - ❌「HOBS 的 EE 分母含 idle-beam power」— 錯；HOBS Eq.(13) 只 sum active beams' TX power。
  - ❌「V2（含 P_c, ρ）是 paper-backed」— 只有 結構 從 terrestrial PAP-2017 借來。
  - ❌「runtime totalPowerW 就是 EE 分母」— 運行時已分離 systemEeBitsPerJoule（active-TX-only） 與
  totalPowerW（assumption-backed proxy）。
  - ❌「採 EEBH-UPLINK 的 (P_l+P_sys) 為 EE 分母」— 那是 uplink terminal 側，不適用 downlink
  衛星場景。
  - ❌「使用 HOBS 50 dBm 為 default power cap」— 與 thesis 採用的 [S10] 10/13 dBW 衝突；只能作為
  source-paper 註腳或 sensitivity 路徑。

  Questions / Assumptions

  下一輪要釐清的：

  1. 要不要保留 HOBS 的 (1 − C_n/T_f) beam-training overhead？ 留 → V1 分母單純就是 TX power；不留 →
  要顯式註明已捨棄該因子，並接受跟 HOBS Eq.(13) 數值不可逐項對照。
  2. policy 是否實際做 per-beam power control？ 若 p_{s,b} 永遠頂在 P_beam,max，V1 變成 (Σ
  R)/(K_active·P_beam,max·Δt)，數字趨近 throughput 線性比例；公式沒問題，問題在 policy 設計。
  3. DAPS 雙活期是否雙計分母功率？ 若 UE 在 serving 與 target 兩衛星都被服務，兩個 p_{s,b}^t
  都計入分母 — 這是 HOBS-style 的「直白」算法，要在 thesis 明文說明。
  4. 要不要 WSEE-style 多 UE 加權？ 若要，那是另一條研究軸（heterogeneous-terminal），不能默默地擴展
  V1。
  5. EE 報告窗口為何？ per-second / per-pass / per-experiment？V1 維度上一致，但圖表必須鎖一個窗口。
  6. V2 是 per-satellite P_c 還是 fleet-wide 常數？ 兩種都 assumption-backed；per-satellite 對齊
  §3.5，fleet-wide 比較簡單。

  PASS / NEEDS MORE EVIDENCE

  PASS — 對 V1/V2/V3/V4 四個候選的 結構：證據足以辯護彼此差別與選擇。V1 是 HOBS Eq.(13) 直接
  anchor；V4 是 EAQL λ_HO=0.2 anchor；V2/V3 結構來自 system-model-formulas §3.13/§3.14 與 BEAMCOORD-EE
   §IV，可以列為 sensitivity。

  NEEDS MORE EVIDENCE — 任何 V2/V3 的 數值 主張：

  │ totalPowerW runtime field             │ assumption-backed proxy   │ §3.16；不是 V1/V2 main 分母                                   │
  ├───────────────────────────────────────┼───────────────────────────┼───────────────────────────────────────────────────────────────┤
  │ HOBS 50 dBm total TX                  │ source-paper context only │ GAP-9；不採作 default cap                                     │
  └───────────────────────────────────────┴───────────────────────────┴───────────────────────────────────────────────────────────────┘

  Forbidden / Unsafe Claims

  下面這些不能寫：

  - ❌「我們的 EE 採用 PAP-2017-BEAMCOORD-EE 的 rate-dependent power model with P_RD=2.4 W/(Gbps)^m」— terrestrial 值，不能套 LEO。
  - ❌「per-handover energy 採 PAP-2025-EAQL 的 3 J 為預設」— 那是 paper-specific assumption。
  - ❌「EE 分母含衛星 circuit power，依某論文採 X W」— corpus 沒有任何 paper 給 LEO 衛星 circuit power 的數值。
  - ❌「HOBS 的 EE 分母含 idle-beam power」— 錯；HOBS Eq.(13) 只 sum active beams' TX power。
  - ❌「V2（含 P_c, ρ）是 paper-backed」— 只有 結構 從 terrestrial PAP-2017 借來。
  - ❌「runtime totalPowerW 就是 EE 分母」— 運行時已分離 systemEeBitsPerJoule（active-TX-only） 與 totalPowerW（assumption-backed proxy）。
  - ❌「採 EEBH-UPLINK 的 (P_l+P_sys) 為 EE 分母」— 那是 uplink terminal 側，不適用 downlink 衛星場景。
  - ❌「使用 HOBS 50 dBm 為 default power cap」— 與 thesis 採用的 [S10] 10/13 dBW 衝突；只能作為 source-paper 註腳或 sensitivity 路徑。

  Questions / Assumptions

  下一輪要釐清的：

  1. 要不要保留 HOBS 的 (1 − C_n/T_f) beam-training overhead？ 留 → V1 分母單純就是 TX power；不留 → 要顯式註明已捨棄該因子，並接受跟 HOBS Eq.(13) 數值不可逐項對照。
  2. policy 是否實際做 per-beam power control？ 若 p_{s,b} 永遠頂在 P_beam,max，V1 變成 (Σ R)/(K_active·P_beam,max·Δt)，數字趨近 throughput 線性比例；公式沒問題，問題在 policy 設計。
  3. DAPS 雙活期是否雙計分母功率？ 若 UE 在 serving 與 target 兩衛星都被服務，兩個 p_{s,b}^t 都計入分母 — 這是 HOBS-style 的「直白」算法，要在 thesis 明文說明。
  4. 要不要 WSEE-style 多 UE 加權？ 若要，那是另一條研究軸（heterogeneous-terminal），不能默默地擴展 V1。
  5. EE 報告窗口為何？ per-second / per-pass / per-experiment？V1 維度上一致，但圖表必須鎖一個窗口。
  6. V2 是 per-satellite P_c 還是 fleet-wide 常數？ 兩種都 assumption-backed；per-satellite 對齊 §3.5，fleet-wide 比較簡單。

  PASS / NEEDS MORE EVIDENCE

  PASS — 對 V1/V2/V3/V4 四個候選的 結構：證據足以辯護彼此差別與選擇。V1 是 HOBS Eq.(13) 直接 anchor；V4 是 EAQL λ_HO=0.2 anchor；V2/V3 結構來自 system-model-formulas §3.13/§3.14 與 BEAMCOORD-EE §IV，可以列為
  sensitivity。

  NEEDS MORE EVIDENCE — 任何 V2/V3 的 數值 主張：

  - LEO 衛星 circuit power P_c,s（GAP-2）— 仍開
  - LEO PA 效率 ρ_s（GAP-1）— 仍開
  - LEO per-handover-event energy E_{u,HO}（GAP-3）— 只有 paper-specific 值；無轉移性 default
  - LEO rate-dependent baseband（GAP-4）— terrestrial only
  - Active/idle/off beam power 數值（GAP-5）— assumption-only

  在這些 GAP 透過獨立來源（Starlink FCC filings、ITU-R S.1528、ESA 衛星 power budget tech notes 等）關閉之前，V2/V3 數值必須留在 simulator-parameter-spec §9.1 governance 規定的 Sensitivity / Internal-only
  區段；V1 留在 Realistic baseline。

● 審查結束 — 沒有寫程式、沒有改文件、沒有跑實驗，只在 paper-catalog 與 system-model-refs 之上做文獻層比對。建議採 V1（HOBS-style 純 TX-power EE）作為主指標、V4（utility-form）作為安全替代或共同 headline；V2/V3
   限於 sensitivity，受 GAP-1/2/3 governance。