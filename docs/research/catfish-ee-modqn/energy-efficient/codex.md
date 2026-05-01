Current Source Evidence

  我只根據本地 paper-catalog、PDF text/layout extraction、system-model-refs 判斷。主要結論是：HOBS 可作為 SINR / throughput / transmit-power EE 的 anchor，但它沒有定義 total consumed power EE。

  主要 evidence：

  - HOBS 原文定義 multi-LEO multi-beam SINR，signal power 與 interference 都使用 beam transmit power P_{n,m}(t)：paper-catalog/txt_all/2024_09_Energy-
    Efficient_Joint_Handover_and_Beam_Switching_Scheme_for_Multi-LEO_Networks.txt:171
  - HOBS throughput 使用 latency-aware rate：
    R = (1 - C_n(t)/T_f) * (W_{n,m}/U_{n,m}(t)) * log2(1 + gamma)：paper-catalog/txt_all/2024_09_Energy-Efficient_Joint_Handover_and_Beam_Switching_Scheme_for_Multi-LEO_Networks.txt:286
  - HOBS EE 是 total throughput divided by sum of beam transmit powers，不含 circuit/static/processing：paper-catalog/txt_all/2024_09_Energy-Efficient_Joint_Handover_and_Beam_Switching_Scheme_for_Multi-
    LEO_Networks.txt:321
  - 系統模型目前也把 HOBS 作為 SINR/interference 來源，但 power model 已另外整理為 p, z, P_c, rho 等可選結構：system-model-refs/system-model-formulas.md:235
  - simulator-parameter-spec 明確標示：circuit power、PA efficiency、active/idle/off beam power、handover energy 都是 assumption / sensitivity，不是完整 paper-backed default：system-model-refs/simulator-
    parameter-spec.md:104

  HOBS Signal / Throughput / EE Formula

  HOBS signal model 可整理成 thesis-friendly notation：

  [
  \gamma_u^t =
  \frac{x_{u,s,b}^t p_{s,b}^t h_{u,s,b}^t}
  {
  \sum_{b' \ne b} z_{s,b'}^t p_{s,b'}^t h_{u,s,b'}^t
  +
  \sum_{s' \ne s}\sum_{b''} z_{s',b''}^t p_{s',b''}^t h_{u,s',b''}^t
  +
  N_0 W_u^t
  }
  ]

  其中 p 必須是線性 Watt；x 是 user association；z 是 beam active indicator。這和 system-model-formulas 的 integrated SINR 一致：system-model-refs/system-model-formulas.md:259

  HOBS throughput：

  [
  R_{n,m,k}(t)=
  \left(1-\frac{C_n(t)}{T_f}\right)
  \frac{W_{n,m}}{U_{n,m}(t)}
  \log_2(1+\gamma_{n,m,k}(t))
  ]

  如果主模擬器沒有 modeling C_n(t)，可以用：

  [
  R_u^t = W_u^t \log_2(1+\gamma_u^t)
  ]

  但要說清楚這是 HOBS-style SINR + Shannon throughput，不是完整照抄 HOBS latency-aware throughput。

  HOBS EE：

  # [
  E_{\mathrm{eff}}(t)

  \frac{R_{\mathrm{tot}}(t)}
  {\sum_n \sum_m P_{n,m}(t)}
  ]

  更適合本研究的 time-aggregated 版本：

  # [
  \eta_{\mathrm{TX}}

  \frac{\sum_t \sum_u R_u^t \Delta t}
  {\sum_t \sum_s \sum_b z_{s,b}^t p_{s,b}^t \Delta t}
  \quad [\mathrm{bits/J}]
  ]

  必要 constraints：

  [
  0 \le p_{s,b}^t \le z_{s,b}^t P_{\mathrm{beam,max}}
  ]

  [
  \sum_b p_{s,b}^t \le P_{\mathrm{sat,max}}
  ]

  [
  x_{u,s,b}^t \le z_{s,b}^t
  ]

  HOBS 原文有 per-beam power cap 與 per-LEO aggregate power cap：paper-catalog/txt_all/2024_09_Energy-Efficient_Joint_Handover_and_Beam_Switching_Scheme_for_Multi-LEO_Networks.txt:387

  Other EE Power-Denominator Patterns

  1. HOBS transmit-power denominator

  [
  \frac{R_{\mathrm{tot}}}{\sum P_{\mathrm{beam}}}
  ]

  只看 beam transmit power。這是最乾淨、最直接和 SINR 變數一致的 pattern。

  2. Beam coordination / rate-dependent processing power

  PAP-2017-BEAMCOORD-EE 定義 total BS power：

  # [
  P_{\mathrm{tot},b}

  \frac{1}{\eta}\sum_k |w_k|^2
  +
  P_{\mathrm{CP},b}
  +
  P_{\mathrm{RD}}\delta(r_b)
  ]

  其中包含 PA efficiency、circuit / fixed / transceiver / channel-estimation / linear-processing power、rate-dependent processing power：paper-catalog/txt_all/2017_07_Energy-
  Efficient_Beam_Coordination_Strategies_With_Rate-Dependent_Processing_Power.txt:316

  這篇提供 denominator 結構，但它是 terrestrial MISO cellular model。P_RD=2.4 W/(Gbps)^m、m=1.2 不能直接宣稱為 LEO satellite-backed value。

  3. Uplink beam-hopping terminal EE

  PAP-2025-EEBH-UPLINK 定義：

  # [
  E_{j,l,t}

  \frac{C_{j,l,t}}
  {P_l^t + P_{\mathrm{sys}}}
  ]

  其中 P_l^t 是 terminal transmit power，P_sys 是 ground user normal-operation / circuit-like power：paper-catalog/txt_all/2025_07_Energy-Efficiency-
  Based_Joint_Uplink_Resources_Allocation_for_LEO_Satellite_Beam-Hopping_System.txt:431

  這支援「TX power + system maintenance power」結構，但它是 uplink terminal-side，不應直接搬成 downlink satellite EE。

  4. Reliable satellite communication expected energy

  PAP-2022-EESAT-RELIABLE 使用：

  # [
  \mathrm{EE}

  \frac{\text{average correctly decoded bits}}
  {\text{average total consumed energy}}
  ]

  並定義 equal-power / variable-power HARQ energy expectation：paper-catalog/txt_layout_all/2022_04_Reliable_and_Energy-Efficient_LEO_Satellite_Commun.layout.txt:513

  這適合 reliability / HARQ 層，不適合直接當 HOBS multi-beam downlink 主公式。

  5. Satellite battery / circuit + transmit energy accounting

  其他 satellite handover/energy papers 有：

  [
  P_c^t = P_a^t + \sum_k P_{t,k,n}^t X_{k,n}^t
  ]

  [
  E_n(t') = E_n(0) - \int P_c^t dt
  ]

  這支持「circuit-like baseline + transmit load」的 energy accounting pattern：paper-catalog/txt_all/2024_03_Energy-Aware_Satellite_Handover_Based_on_Deep_Reinforcement_Learning.txt:277

  但其中的 scenario battery / pass energy values 不能轉成通用 fixed Watt default。

  Consistency Requirements

  如果採用 HOBS-style SINR，EE denominator 至少要滿足：

  1. Denominator 中的 transmit power 必須是 SINR 裡同一個 p_{s,b}^t，不能另設一個不影響 SINR 的 power variable。
  2. 單位必須用 linear Watt；dBm/dBW 只作輸入轉換。
  3. 如果加入 PA efficiency，應是：
     [
     \frac{1}{\rho_s}\sum_b z_{s,b}^t p_{s,b}^t
     ]
     而不是新增一個和 SINR 無關的 TX term。
  4. 如果加入 circuit/static/processing/active-beam overhead，這些要標示為 assumption 或 sensitivity variable，除非有 LEO-specific paper-backed numeric source。
  5. 分子與分母要在同一 time horizon 聚合。若分子是 sum bits，分母就應是 sum Joules。
  6. 若所有 beams 永遠 active、p 固定不變，EE 會退化成 throughput proxy。

  Candidate EE Formula Versions

  Version A: 最簡 HOBS-style transmit-power EE

  # [
  \eta_{\mathrm{TX}}

  \frac{\sum_t \sum_u R_u^t \Delta t}
  {\sum_t \sum_s \sum_b z_{s,b}^t p_{s,b}^t \Delta t}
  ]

  分子：HOBS-style SINR 產生的 user throughput，可用 Shannon rate；若有 C_n/T_f 則用 HOBS latency-aware throughput。

  分母：active beam RF transmit energy only。

  需要參數：x, z, p, channel gain, bandwidth, noise PSD, Delta t, per-beam / per-satellite power caps。

  Paper-backed：HOBS SINR、throughput、transmit-power EE 結構；S10 power caps 在 system-model 中是目前 default cap source。

  Assumption：不包含 circuit/static/PA loss/processing/beam overhead。

  優點：最忠於 HOBS；和 SINR 變數一致；claim 最安全。

  風險：不是 total consumed energy；若 power control 不變，容易接近 throughput proxy。

  位置：main metric 可以使用，但應命名為 transmission-side EE 或 active beam transmit-power EE。

  Version B: Composite simulated system EE

  # [
  P_{\mathrm{comm},s}^t

  P_{c,s}
  +
  \frac{1}{\rho_s}\sum_b z_{s,b}^t p_{s,b}^t
  +
  P_{\mathrm{beam,on}}\sum_b z_{s,b}^t
  ]

  # [
  \eta_{\mathrm{comm}}

  \frac{\sum_t \sum_u R_u^t \Delta t}
  {\sum_t \sum_s P_{\mathrm{comm},s}^t \Delta t}
  ]

  分子：同 Version A。

  分母：satellite communication power，包括 static/circuit、PA-scaled transmit power、active-beam overhead。

  需要參數：P_c, rho, P_beam,on, z, p。

  Paper-backed：結構上有 PAP-2017 total power decomposition、PAP-2025-EEBH 的 TX + P_sys pattern、satellite energy accounting papers 的 baseline + transmit load pattern。

  Assumption：P_c, rho, P_beam,on 的 LEO-specific numeric values。

  優點：比 Version A 更像 system energy；能懲罰開太多 beam。

  風險：assumption-heavy；若拿來當主 metric，容易被質疑 denominator 參數來源。

  位置：ablation / sensitivity 較合適；若當 main metric，必須完整揭露 assumption set。

  Version C: Literature-rich / sensitivity-analysis EE

  # [
  P_{\mathrm{rich},s}^t

  P_{c,s}
  +
  \frac{1}{\rho_s}\sum_b z_{s,b}^t p_{s,b}^t
  +
  P_{\mathrm{beam,on}}\sum_b z_{s,b}^t
  +
  P_{\mathrm{RD}}\left(\sum_{u \in \mathcal{U}_s^t} R_u^t\right)^m
  ]

  # [
  \eta_{\mathrm{rich}}

  \frac{\sum_t \sum_u R_u^t \Delta t}
  {\sum_t \sum_s P_{\mathrm{rich},s}^t \Delta t + E_{\mathrm{HO,total}}}
  ]

  分子：same delivered bits。

  分母：static + PA-scaled TX + active-beam overhead + rate-dependent processing + optional handover event energy。

  需要參數：Version B parameters plus P_RD, m, optional E_HO.

  Paper-backed：PAP-2017 supports rate-dependent processing structure; handover overhead literature supports signaling-energy concept only.

  Assumption：LEO-specific P_RD, m, E_HO, active-beam overhead values。

  優點：最適合檢查 EE 結論是否對 denominator modeling robust。

  風險：不是 HOBS-native；參數自由度高；不適合寫成 paper-backed main model。

  位置：sensitivity / future work。

  Version D: HARQ expected-energy EE，不建議作 HOBS main

  # [
  \eta_{\mathrm{HARQ}}

  \frac{K_{\mathrm{correct}}}
  {\mathbb{E}[E_{\mathrm{tx,total}}]}
  ]

  分子：average correctly decoded bits。

  分母：HARQ rounds 的 expected consumed energy。

  Paper-backed：PAP-2022-EESAT-RELIABLE。

  風險：它是 single-link reliability / retransmission framing，不是 HOBS multi-beam scheduling EE。

  位置：future work 或 reliability extension。

  Recommended Ranking

  1. Version A as main metric：HOBS-style active transmit-power EE。
  2. Version B as primary ablation：composite simulated system EE。
  3. Version C as sensitivity：rate-dependent / active-beam / handover-energy robustness。
  4. Version D only if the simulator explicitly adds HARQ/reliability behavior。
  5. 不建議版本：直接把 uplink terminal EE、terrestrial BS processing constants、或 assumption-based active-beam overhead 宣稱成 LEO paper-backed main EE。

  Main Metric Recommendation

  主 metric 建議用：

  # [
  \eta_{\mathrm{TX}}

  \frac{\sum_t \sum_u R_u^t \Delta t}
  {\sum_t \sum_s \sum_b z_{s,b}^t p_{s,b}^t \Delta t}
  ]

  名稱不要叫 total energy efficiency；建議叫：

  - transmission-side energy efficiency
  - active beam transmit-power EE
  - HOBS-style transmit-power EE

  這樣最符合 HOBS SINR anchor，也最不會引入 unsupported denominator parameters。

  Ablation / Sensitivity Recommendations

  建議至少做兩組 denominator sensitivity：

  1. Add PA + static circuit：
     [
     P_c + \frac{1}{\rho}\sum z p
     ]
  2. Add active-beam overhead：
     [
     P_c + \frac{1}{\rho}\sum z p + P_{\mathrm{beam,on}}\sum z
     ]
  3. Optional rate-dependent processing：
     [
      - P_{\mathrm{RD}}R^m
        ]

  但第 2、3 組都要明確標為 assumption / sensitivity，不要宣稱為 HOBS-backed。

  Parameters: Paper-Backed vs Assumption

  Paper-backed / structurally backed：

  - HOBS SINR, intra/inter-satellite interference, throughput structure, transmit-power EE denominator。
  - HOBS SINR threshold = 10 dB 可作 source-context parameter。
  - HOBS 50 dBm max transmit power 是 HOBS scenario value，不是目前 system-model default。
  - S10 per-beam 10 dBW and per-satellite 13 dBW caps 是目前 system-model default cap source。
  - PAP-2017 supports TX/eta + circuit + rate-dependent processing denominator structure。
  - PAP-2025-EEBH-UPLINK supports transmit power + P_sys pattern for uplink terminal EE。
  - PAP-2022-EESAT supports expected-energy bits/J for reliability/HARQ setting。

  Assumption / sensitivity only：

  - Satellite circuit/static power in W。
  - PA efficiency rho。
  - Active beam overhead P_beam,on。
  - Idle/off beam power。
  - LEO-specific rate-dependent processing constants。
  - Universal handover event energy in Joules。
  - Converting paper-specific Wh/pass battery accounting into fixed W defaults。

  Forbidden / Unsafe Claims

  不要寫：

  - HOBS EE includes total consumed power。
  - HOBS includes circuit/static/processing power。
  - Active-beam overhead has a paper-backed default value。
  - PAP-2017 rate-dependent processing constants are LEO satellite-backed。
  - PAP-2025-EEBH uplink terminal denominator directly applies to downlink satellite beams。
  - HOBS 50 dBm is the adopted default thesis power cap if system-model uses S10 caps。
  - EE is not a throughput proxy unless denominator actually changes with p, z, or other energy terms。

  Questions / Assumptions

  我假設主研究目標是 downlink multi-LEO multi-beam EE，而不是 uplink terminal EE。

  我也假設主 HOBS-style simulator 目前沒有 HARQ/retransmission energy layer；若未來加入 reliability layer，PAP-2022-EESAT 的 expected-energy framing 才適合作為擴充。

  還需要決定：主 throughput 是否保留 HOBS 的 latency factor (1 - C_n/T_f)。如果 simulator 沒有明確建模 training / coordination overhead，主公式應避免假裝完整採用該項。

  PASS / NEEDS MORE EVIDENCE

  PASS：採用 HOBS-style SINR + transmit-power EE 作為 main metric，但要命名為 transmission-side EE。

  NEEDS MORE EVIDENCE：把 circuit/static/processing/active-beam overhead 放進 headline total-system EE，除非另有 LEO-specific numeric source 或明確把它們標成 assumption / sensitivity。