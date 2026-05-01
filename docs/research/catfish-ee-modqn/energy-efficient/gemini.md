根據你提供的系統文件、審查指南以及 paper-catalog 中論文的資料（包含 PAP-2024-HOBS, PAP-2017-BEAMCOORD-EE, PAP-2022-EESAT-RELIABLE, PAP-2025-EEBH-UPLINK, PAP-2019-HO-OVERHEAD, PAP-2026-D2AOF-HOPOWER
  等），以下是針對 Energy Efficiency (EE) 系統模型的文獻與公式審查報告。

  Current Source Evidence
   - Local Guideline & Specs: AGENTS.md, system-model-formulas.md, system-model-derivation.md, simulator-parameter-spec.md, simulator-parameter-provenance-inventory.md, paper-audit-report.md.
   - Target Papers:
     - PAP-2024-HOBS: 定義了 Multi-LEO 下的 SINR、Latency-Aware Throughput 與最簡 EE（純發射功率）。
     - PAP-2017-BEAMCOORD-EE: 定義了指標性的 Multi-antenna 系統 EE 模型，包含 PA 效率、rate-independent 電路功率、以及 rate-dependent 基頻處理功率。
     - PAP-2022-EESAT-RELIABLE: 定義了 IR-HARQ 下的 $EE = \text{bits} / \text{Joule}$，優化各輪 Variable Power。
     - PAP-2025-EEBH-UPLINK: 定義了 Uplink WSEE，分母包含終端發射功率與系統維持功率 ($P_{sys}$)。
     - PAP-2019-HO-OVERHEAD: 定義了 LTE 換手信令（MeasReport, HOcmd 等）的 Supply Power 消耗（包含 PA 與 RF電路耗能），提供了 HO Overhead 能耗量化的基礎。
     - PAP-2026-D2AOF-HOPOWER: 定義了將 Handover Timeout 與 Throughput 結合的 Reward，但其 Power Allocation 僅為控制變數，未提供 closed-form EE 定義。

  ---

  HOBS Signal / Throughput / EE Formula
   1. SINR 公式:
     $$ \gamma_{n,m,k}(t) = \frac{P_{n,m}(t) H_{n,m,k}(t) G^T_{n,m,k}(t) G^R_{n,m,k}(t) a_{n,m,k}(t)}{I^a_{n,m,k} + I^b_{n,m,k} + \sigma^2} $$
      - 考量了 Intra-LEO ($I^a$) 與 Inter-LEO ($I^b$) 的同頻干擾。
   2. Throughput 公式:
     $$ R_{n,m,k}(t) = \left(1 - \frac{C_n(t)}{T_f}\right) \frac{W_{n,m}}{U_{n,m}(t)} \log_2(1 + \gamma_{n,m,k}(t)) $$
      - 引入了 Beam training latency $C_n(t)$ 作為時間資源 overhead。
   3. EE 公式:
     $$ E_{eff}(t) = \frac{R_{tot}(t)}{\sum_{n} \sum_{m} P_{n,m}(t)} $$
      - 分子: 系統總吞吐量 $R_{tot}$。
      - 分母: 僅為所有啟動波束的「純發射功率 ($P_{n,m}$)」總和。
   4. Power-Control 限制:
      - 每個波束具有最大功率限制 $P_{beam}$，單顆衛星具有總功率限制 $P_{max}$。
      - 使用啟發式的動態功率控制 (DPC)：根據上個 Frame 的 EE 增減來決定發射功率的 step 增減。

  ---

  Other EE Power-Denominator Patterns
   1. PAP-2017-BEAMCOORD-EE (Composite Power Model):
      - 分母：$P_{tot} = \frac{1}{\eta} P_{tx} + P_{CP} + P_{RD}\delta(r)$
      - 將功率拆解為：PA 轉換損耗 ($\eta$)、固定靜態電路功率 ($P_{FIX}$)、波束 RF 鏈與線性處理功率 ($P_{TC} + P_{LP}$)、以及與吞吐量非線性相關的基頻處理功率 ($P_{RD}\cdot r^m$)。
   2. PAP-2025-EEBH-UPLINK (System Maintenance Power):
      - 分母：$P_{tx} + P_{sys}$。將系統維持所需的基本耗能獨立列出。
   3. PAP-2019-HO-OVERHEAD (Signaling Overhead Power):
      - 分母：$P_{sup} = P_{out}/\eta + (n_{RB}/N_{RB}) \cdot P_{RF}$。明確指出在頻繁進行 Handover 信令交換時，會佔用資源並產生額外的 RF 電路與發射耗能。
   4. PAP-2022-EESAT-RELIABLE (Energy per Bit):
      - $EE = \text{bits} / E_{total}$。利用總消耗焦耳 (Joules) 來計算，適合跨時域 (Time-window) 的傳輸與重傳耗能加總。

  ---

  Consistency Requirements
   - 若本研究的核心 SINR 採用 HOBS-style，SINR 分子中的 $P_{n,m}$ 就是純粹的 RF Transmit Power。
   - 在構建 EE 分母時，必須維持物理量的一致性：$P_{n,m}$ 在進入 EE 分母時，理應乘上 Power Amplifier 效率倒數 ($1/\eta$)，才能代表衛星系統為了發射這個 RF 訊號所實際從電源抽取的瓦數（如 PAP-2017-BEAMCOORD-EE
     所述）。
   - 若要優化 Handover 與 Beam Configuration，分母必須具備能懲罰「無謂開啟波束」與「頻繁換手」的對應項（如 $P_{circuit}$ 與 $E_{HO}$），否則 EE 的優化方向將會與 Throughput 完全一致。

  ---

  Candidate EE Formula Versions

  Version 1: Strict HOBS-style Transmit-Power EE
   - 公式: $EE = \frac{\sum R_{n,m,k}}{\sum P_{n,m}}$
   - 分子: 總吞吐量。
   - 分母: 僅包含啟動波束的 RF 發射功率總和。
   - 需要的參數: $P_{n,m}$。
   - Paper-backed: PAP-2024-HOBS。
   - Assumption: 無。
   - 優點: 最簡單，與 HOBS 完全一致，直接對齊 SINR 的變數。
   - 風險: 缺乏電路靜態功耗，EE 容易退化成 Throughput Proxy；演算法會傾向盲目開啟所有波束以衝高分子，因為多開一個波束的額外邊際功耗在分母占比不高。
   - 適合位置: Ablation Study / Baseline。

  Version 2: Composite Simulated System EE (推薦)
   - 公式: $EE = \frac{\sum R_{n,m,k}}{ P_{static} + \sum_{m \in \mathcal{M}_{active}} \left(\frac{1}{\eta} P_{n,m} + P_{act,m}\right) + P_{RD}\cdot(\sum R_{n,m,k})^C }$
   - 分子: 總吞吐量。
   - 分母: 衛星靜態維持功率 ($P_{static}$) + (發射功率 / $\eta$ + RF 鏈維持功率 $P_{act,m}$) $\times$ Active 波束數量 + 基頻資料處理功率。
   - 需要的參數: $\eta$ (PA efficiency), $P_{static}$ (衛星基礎功耗), $P_{act,m}$ (單一波束 RF/Baseband 維持功耗), $P_{RD}$ 與 $C$ (處理功耗係數)。
   - Paper-backed: PAP-2017-BEAMCOORD-EE 提供完整的模型結構與 $\eta, P_{RD}, C$ 參考值；PAP-2025-EEBH-UPLINK 支持 $P_{sys}$ 的概念。
   - Assumption: 具體的 LEO 衛星 $P_{static}$ 與 $P_{act,m}$ 瓦數需自訂 (Assumption-backed) 或依賴 simulator-parameter-spec.md。
   - 優點: 物理意義最完備。引入 $P_{act,m}$ 能迫使 RL 學習在「流量需求低時關閉波束」以提升 EE，避免全開波束。
   - 風險: 參數較多，若 $P_{static}$ 設得過大，EE 會完全被分子 (Throughput) 主導；若 $P_{act,m}$ 設得過大，RL 會傾向完全關閉波束。
   - 適合位置: Main Metric。

  Version 3: Literature-rich HO-Penalty EE
   - 公式: $EE = \frac{\sum R_{n,m,k}}{\text{Power}_{System} + \frac{1}{T_{window}} \sum_{t} E_{u,HO}}$
   - 分子: 總吞吐量。
   - 分母: Version 2 的 System Power 加上單位時間內的 Handover 信令能量消耗 ($E_{u,HO}$)。
   - 需要的參數: Version 2 所有參數 + $E_{u,HO}$。
   - Paper-backed: PAP-2019-HO-OVERHEAD 提供換手信令能量懲罰的文獻支持；system-model-formulas.md 明定 $E_{u,HO} = 3\text{ J}$。
   - Assumption: 換手事件平攤為連續功率的時間窗口 $T_{window}$ 長度。
   - 優點: 將 DAPS 減少 Handover 中斷 / 信令開銷的故事線，直接在物理 EE 公式中兌現。
   - 風險: 瞬時的焦耳 (J) 轉換為瓦特 (W) 的時序平滑處理在公式表達上容易顯得不自然，可能會被 reviewer 挑戰維度一致性。
   - 適合位置: Sensitivity Analysis / Future Work，或作為 DRL Reward 的額外 Penalty 結構。

  Version 4: QoS-Weighted EE (不建議)
   - 公式: 將 timeout penalty, EES weight, 或 latency 直接放入 EE 數學式（如 PAP-2025-EEBH-UPLINK 的 WSEE 或 PAP-2026-D2AOF-HOPOWER 的 unified reward）。
   - 風險: 混淆了 bits/Joule 的嚴謹物理定義。

  ---

  Recommended Ranking
   1. 推薦第一名: Version 2 (Composite Simulated System EE)。結構嚴謹，直接解決了純 transmit power 造成的退化問題，並有 PAP-2017-BEAMCOORD-EE 作為強力文獻支持。
   2. 推薦第二名: Version 1 (Strict HOBS-style EE)。作為對比用的 Baseline（證明若不考慮波束開啟成本，傳統演算法有多浪費電）。
   3. 推薦第三名: Version 3 (HO-Penalty EE)。放在 RL reward function 中實作，而非硬寫入 System Model 的 EE metric 公式中。

  ---

  Main Metric Recommendation
  建議採用 Version 2 作為論文的 Main EE Metric。
  將系統總能耗明確拆分為：$P_{total} = P_{static} + P_{active\_beam\_overhead} + P_{transmit\_actual} + P_{processing}$。
  這能確保模型具備：(1) 對 SINR 的敏感度 (via $P_{transmit}$)，(2) 對波束配置的敏感度 (via $P_{active\_beam}$)，(3) 對 Throughput 上限的物理制約 (via $P_{processing}$)。

  Ablation / Sensitivity Recommendations
   - Ablation: 使用 Version 1 作為消融實驗。展示如果 Agent 只針對 HOBS 的純發射功率 EE 進行優化，在真實的 Composite Power 環境中，其真實 EE 會暴跌（因為它開啟了過多不必要的波束）。
   - Sensitivity: 針對 $\eta$ (PA efficiency) 與 $P_{act,m}$ (Active beam overhead) 進行敏感度分析。這兩個參數直接決定了系統應該傾向「少波束高功率」還是「多波束低功率」。

  Parameters: Paper-Backed vs Assumption
   - Paper-Backed:
     - $P_{n,m}$, $H_{n,m,k}$, $G^T$, $G^R$, $I^a, I^b, \sigma^2$ (PAP-2024-HOBS)
     - $\eta, P_{RD}, m$ 等基頻/放大器結構與數值 (PAP-2017-BEAMCOORD-EE 提供例如 $\eta \approx 30\%$, $P_{RD}=2.4$)
     - $E_{u,HO} = 3\text{ J}$ (基於 local system-model-formulas.md)
   - Assumption / Scenario Dependent:
     - 衛星基礎維持功耗 $P_{static}$ 與單一波束 RF 維持功耗 $P_{act,m}$。在文獻中通常高度依賴硬體規格（例如 LEO 衛星通常總功耗在幾百瓦到幾千瓦不等）。必須在參數表中明確標示為 Simulation Assumption。

  Forbidden / Unsafe Claims
   - 禁止宣稱：「本研究的 EE 公式完全照抄 PAP-2024-HOBS」。(因為我們建議擴充了 Circuit/Baseband Power)。
   - 禁止宣稱：「$P_{static}$ 與 $P_{act,m}$ 是由 3GPP 或 ITU-R 明確定義的物理常數」。(它們是系統模擬假設值)。
   - 禁止將 QoS (如延遲或 Timeout) 直接以權重形式寫入 EE (bits/Joule) 的物理分母中。這只能放在 RL Reward 中。

  Questions / Assumptions
   - 是否應完全照抄 HOBS 的 EE 公式？
    否。因為它會讓 EE 失去優化「波束啟閉 (Beam Configuration)」的能力。
   - 是否應加入 circuit/static/processing power？
    強烈建議加入。PAP-2017-BEAMCOORD-EE 證明了這對系統真實 EE 有決定性影響。
   - 是否應加入 active-beam overhead？
    是。這是懲罰無效波束開啟、促使 Agent 學習 Beam Hopping/Switching 節能的數學基礎。
   - 是否可以只使用 transmit power？
    僅限於作為對比用的 Baseline (Ablation)，不應作為 proposed algorithm 的最終目標。
   - 如果要避免 EE 退化成 throughput proxy，formula 層面需要什麼條件？
    分母必須存在與發射功率 ($P_{tx}$) 無關、且數值不可忽略的**常數項或與啟動狀態綁定的階躍項** (例如 $P_{static}$ 或 $\sum_{active} P_{act,m}$)。這樣當 Throughput
  上升但開啟過多波束時，分母的膨脹速度才會大於分子的增長，從而使 EE 產生 Trade-off 的極值點。

  PASS / NEEDS MORE EVIDENCE
  PASS. 文獻證據充足，PAP-2017-BEAMCOORD-EE 與 PAP-2024-HOBS 提供了完美的結合點（HOBS 負責 Signal Model，BEAMCOORD-EE 負責 Power Denominator Model）。不需要進一步尋找其他文獻即可確立堅實的 mathematical model。