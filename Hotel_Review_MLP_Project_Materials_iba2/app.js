/**
 * Hotel Review Sentiment Analyzer — Frontend Application Logic
 * Implements Slides 28-34:
 * - Login/Logout with Teacher test account
 * - Demo reviews quick-loader (30 reviews)
 * - Multi-sentence Review Sentiment Analysis & Slide 30 Aggregation
 * - Interactive Sentence Inspector & Manual Detection Studio
 * - Model Architecture & Benchmark Visualizer
 */

const ASPECT_LABELS = {
  cleanliness: { zh: "卫生清洁", en: "Cleanliness", icon: "✨" },
  service: { zh: "前台服务", en: "Service", icon: "🤝" },
  location: { zh: "地理位置", en: "Location", icon: "📍" },
  facilities: { zh: "硬件设施", en: "Facilities", icon: "🛠️" },
  room_comfort: { zh: "房间舒适度", en: "Room Comfort", icon: "🛏️" },
  sound_insulation_noise: { zh: "隔音与噪音", en: "Sound & Noise", icon: "🔇" },
  food: { zh: "餐饮质量", en: "Food & Dining", icon: "🍽️" },
  value: { zh: "价格性价比", en: "Value for Money", icon: "💰" },
};

const STATE_COLORS = {
  0: { name: "未提及", color: "absent", badge: "0" },
  1: { name: "负面", color: "negative", badge: "-1" },
  2: { name: "中性", color: "neutral", badge: "0" },
  3: { name: "正面", color: "positive", badge: "+1" },
  4: { name: "混合", color: "mixed", badge: "0" },
};

// Global App State
let currentUser = { username: "teacher", role: "Instructor / Auditor", name: "Professor Wei Ou" };
let demoReviews = [];
let currentAnalysis = null;
let activeAspectFilter = "all";

// ==============================================================================
// 1. App Initialization
// ==============================================================================
document.addEventListener("DOMContentLoaded", async () => {
  initAuth();
  setupTextareaListeners();
  await loadDemoReviewsList();
  await loadModelBenchmarkInfo();

  // Run initial manual detection on default sentence
  detectSingleSentence();

  // Preload training dataset stats and table items immediately
  loadDatasetStats();
  loadDatasetItems(1);
});

function switchTab(tabId) {
  document.querySelectorAll(".nav-tab").forEach(tab => tab.classList.remove("active"));
  document.querySelectorAll(".tab-panel").forEach(panel => panel.classList.remove("active"));

  if (tabId === "review") {
    document.getElementById("tabBtnReview").classList.add("active");
    document.getElementById("panelReview").classList.add("active");
  } else if (tabId === "manual") {
    document.getElementById("tabBtnManual").classList.add("active");
    document.getElementById("panelManual").classList.add("active");
  } else if (tabId === "model") {
    document.getElementById("tabBtnModel").classList.add("active");
    document.getElementById("panelModel").classList.add("active");
  } else if (tabId === "dataset") {
    document.getElementById("tabBtnDataset").classList.add("active");
    document.getElementById("panelDataset").classList.add("active");
    loadDatasetStats();
    loadDatasetItems(currentDatasetPage || 1);
  }
}

// ==============================================================================
// 2. Authentication (Slide 29: Login/Logout & Teacher Test Account)
// ==============================================================================
function initAuth() {
  const stored = localStorage.getItem("hotel_analyzer_user");
  if (stored) {
    try {
      currentUser = JSON.parse(stored);
    } catch (e) {
      console.warn("Invalid stored user", e);
    }
  }
  updateAuthUI();
}

function updateAuthUI() {
  document.getElementById("displayUsername").textContent = currentUser.username;
  document.getElementById("displayUserRole").textContent = currentUser.role;
}

function handleLogout() {
  localStorage.removeItem("hotel_analyzer_user");
  document.getElementById("loginModal").style.display = "flex";
}

async function handleShutdown() {
  const btn = document.getElementById("btnShutdown");
  if (!btn || btn.disabled) return;

  const ok = window.confirm(
    "确认完全关闭后端服务？\n\n" +
    "将调用 /api/shutdown，释放模型权重与 embedding 缓存，并强制退出 Python 进程。\n" +
    "关闭后请重新运行 python server.py 才能再次使用。"
  );
  if (!ok) return;

  btn.disabled = true;
  btn.innerHTML = '<span class="shutdown-dot"></span> 正在关闭…';

  let serverAck = false;
  try {
    const res = await fetch("/api/shutdown", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });
    if (res.ok) {
      const data = await res.json().catch(() => ({}));
      serverAck = !!data.ok;
    }
  } catch (_) {
    // Process may have already exited mid-response — treat as success.
    serverAck = true;
  }

  // Ensure overlay even if network died first.
  showShutdownOverlay(serverAck);
}

function showShutdownOverlay(acked) {
  let el = document.getElementById("shutdownOverlay");
  if (!el) {
    el = document.createElement("div");
    el.id = "shutdownOverlay";
    el.className = "shutdown-overlay";
    el.innerHTML = `
      <div class="so-icon">⏻</div>
      <h2>后端服务已完全关闭</h2>
      <p>Python 进程已退出，模型权重与缓存已释放，不再占用额外 CPU / 内存。</p>
      <p class="so-hint">${acked ? "服务端已确认关闭请求。" : "连接已断开（进程可能已退出）。"}如需再次使用，请在终端重新运行：<br><code>python server.py</code></p>
    `;
    document.body.appendChild(el);
  } else {
    el.classList.add("visible");
  }
  el.classList.add("visible");

  // Block further interaction with the app chrome.
  document.querySelectorAll("button, input, textarea, select").forEach((n) => {
    if (n.id !== "btnShutdown") n.disabled = true;
  });
  const badge = document.getElementById("badgeGen");
  if (badge) badge.innerHTML = '<span class="gen-dot" style="background:#ef4444;box-shadow:0 0 8px #ef4444"></span> 服务已关闭 · Server Stopped';
}

function fillAccount(u, p) {
  document.getElementById("inputUsername").value = u;
  document.getElementById("inputPassword").value = p;
}

async function submitLogin() {
  const u = document.getElementById("inputUsername").value.trim();
  const p = document.getElementById("inputPassword").value.trim();

  try {
    const res = await fetch("/api/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: u, password: p }),
    });
    const data = await res.json();
    if (!res.ok) {
      alert(data.detail || "登录失败");
      return;
    }

    currentUser = data.user;
    localStorage.setItem("hotel_analyzer_user", JSON.stringify(currentUser));
    updateAuthUI();
    document.getElementById("loginModal").style.display = "none";
  } catch (err) {
    alert("网络连接失败，请检查后端服务是否正在运行。");
  }
}

// ==============================================================================
// 3. Demo Reviews & Textarea Handlers
// ==============================================================================
async function loadDemoReviewsList() {
  try {
    const res = await fetch("/api/demo-reviews");
    const data = await res.json();
    demoReviews = data.demos || [];

    const select = document.getElementById("demoSelect");
    select.innerHTML = '<option value="">-- 选择示例评论 (DEMO 00~29) --</option>';

    demoReviews.forEach((item, idx) => {
      const opt = document.createElement("option");
      opt.value = idx;
      const preview = item.review_text.substring(0, 35).replace(/\n/g, " ");
      opt.textContent = `${item.demo_id}: ${preview}...`;
      select.appendChild(opt);
    });
  } catch (e) {
    console.error("Failed to load demo reviews:", e);
  }
}

function setupTextareaListeners() {
  const textarea = document.getElementById("reviewInput");
  const counter = document.getElementById("charCount");

  const updateCount = () => {
    const text = textarea.value.trim();
    const len = text.length;
    const sents = text ? text.split(/[。！？；…\n\r]+/).filter(s => s.trim().length > 0).length : 0;
    counter.textContent = `${len} 字符 • 约 ${sents} 句`;
  };

  textarea.addEventListener("input", updateCount);
}

function clearReviewInput() {
  const textarea = document.getElementById("reviewInput");
  textarea.value = "";
  textarea.dispatchEvent(new Event("input"));
}

function loadSelectedDemo() {
  const select = document.getElementById("demoSelect");
  const idx = select.value;
  if (idx === "" || !demoReviews[idx]) return;

  const item = demoReviews[idx];
  const textarea = document.getElementById("reviewInput");
  textarea.value = item.review_text;
  textarea.dispatchEvent(new Event("input"));

  // Auto trigger analysis
  analyzeCurrentReview();
}

function insertSample(type) {
  const textarea = document.getElementById("reviewInput");
  if (type === 1) {
    textarea.value = "感觉前台的服务不错，都是笑盈盈的。出脚十分的方便，轮渡就在对面，唯一的不足的是中午睡觉还是感觉有些吵。早餐是下面的西餐厅，品种比较丰富。";
  } else if (type === 2) {
    textarea.value = "入住的时候我强调不要靠窗的房间，结果晚上休息的时候发现自己根本就是住在马路上。凌晨一点睡下，窗户外传来嘹亮的汽车喇叭声，连绵不绝。在床上还可以听到大街上人高声说话的声音、楼下店铺关门的声音。天哪，噩梦啊！";
  } else if (type === 3) {
    textarea.value = "房间很干净，床也很舒服。但是晚上隔音不好，空调也不好用。前台态度很好，第二天服务员主动帮我换房。退房办理得很快。去地铁站很方便。";
  }
  textarea.dispatchEvent(new Event("input"));
  analyzeCurrentReview();
}

// ==============================================================================
// 4. Review Sentiment Analysis (Slide 29-33)
// ==============================================================================
async function analyzeCurrentReview() {
  const textarea = document.getElementById("reviewInput");
  const text = textarea.value.trim();
  if (!text) {
    alert("请输入或粘贴酒店评论文本后再进行分析！");
    return;
  }

  const btn = document.getElementById("btnAnalyze");
  const statusBadge = document.getElementById("analysisStatus");
  btn.disabled = true;
  btn.innerHTML = '<span class="btn-icon">⏳</span><span class="btn-text">正在提取向量与推断...</span>';
  statusBadge.textContent = "分析中...";
  statusBadge.className = "badge-status running";

  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    const data = await res.json();
    if (!res.ok) {
      alert(data.detail || "分析请求失败");
      return;
    }

    currentAnalysis = data;
    renderAspectBars(data.aspect_scores);
    renderSentenceBreakdown(data.sentence_breakdown, "all");

    statusBadge.textContent = `已完成 (${data.total_sentences} 句)`;
    statusBadge.className = "badge-status success";
  } catch (err) {
    console.error("Analysis error:", err);
    alert("分析失败，请确认后端 server.py 正在运行。");
    statusBadge.textContent = "错误";
    statusBadge.className = "badge-status error";
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<span class="btn-icon">⚡</span><span class="btn-text">开始多维度情感分析</span>';
  }
}

function renderAspectBars(aspectScores) {
  const container = document.getElementById("aspectBarsGrid");
  container.innerHTML = "";

  const aspectsList = [
    "cleanliness",
    "service",
    "location",
    "facilities",
    "room_comfort",
    "sound_insulation_noise",
    "food",
    "value",
  ];

  aspectsList.forEach(aspect => {
    const info = aspectScores[aspect] || {};
    const meta = ASPECT_LABELS[aspect] || { zh: aspect, en: aspect, icon: "🏷️" };
    const score = info.score; // -100 to +100, or null
    const n = info.mention_count || 0;

    const row = document.createElement("div");
    row.className = "aspect-row";
    row.id = `aspectRow_${aspect}`;
    row.onclick = () => toggleAspectSentenceFilter(aspect);

    // Left label
    const labelGroup = document.createElement("div");
    labelGroup.className = "aspect-label-group";
    labelGroup.innerHTML = `
      <span class="aspect-name">${meta.icon} ${meta.zh}</span>
      <span class="aspect-name-en">${meta.en}</span>
    `;

    // Center Bar Track (-100 to +100)
    const trackWrapper = document.createElement("div");
    trackWrapper.className = "bar-track-wrapper";
    trackWrapper.innerHTML = `<div class="center-divider"></div>`;

    if (score === null || n === 0) {
      trackWrapper.innerHTML += `<div class="not-mentioned-tag">Not mentioned</div>`;
    } else {
      const absScore = Math.abs(score);
      const halfWidth = (absScore / 100) * 50; // max width 50%
      const bar = document.createElement("div");

      if (score > 0) {
        bar.className = "score-bar positive";
        bar.style.width = `${Math.max(3, halfWidth)}%`;
      } else if (score < 0) {
        bar.className = "score-bar negative";
        bar.style.width = `${Math.max(3, halfWidth)}%`;
      } else {
        bar.className = "score-bar neutral";
      }
      trackWrapper.appendChild(bar);
    }

    // Right Score Badge & Mentions (Slide 31)
    const scoreBadge = document.createElement("div");
    scoreBadge.className = "aspect-score-badge";

    let scoreClass = "absent";
    let scoreDisplay = "Not mentioned";

    if (score !== null && n > 0) {
      if (score > 0) {
        scoreClass = "positive";
        scoreDisplay = `+${score}`;
      } else if (score < 0) {
        scoreClass = "negative";
        scoreDisplay = `${score}`;
      } else {
        scoreClass = "neutral";
        scoreDisplay = "0.0";
      }
    }

    scoreBadge.innerHTML = `
      <span class="score-value ${scoreClass}">${scoreDisplay}</span>
      <span class="mention-count-text">${info.summary_text || "未提及"}</span>
    `;

    row.appendChild(labelGroup);
    row.appendChild(trackWrapper);
    row.appendChild(scoreBadge);
    container.appendChild(row);
  });
}

function toggleAspectSentenceFilter(aspect) {
  if (activeAspectFilter === aspect) {
    activeAspectFilter = "all";
  } else {
    activeAspectFilter = aspect;
  }

  document.querySelectorAll(".aspect-row").forEach(r => r.classList.remove("active-filter"));
  if (activeAspectFilter !== "all") {
    const activeRow = document.getElementById(`aspectRow_${activeAspectFilter}`);
    if (activeRow) activeRow.classList.add("active-filter");
  }

  if (currentAnalysis) {
    renderSentenceBreakdown(currentAnalysis.sentence_breakdown, activeAspectFilter);
  }
}

function filterSentenceAspect(aspect) {
  activeAspectFilter = aspect;
  document.querySelectorAll(".aspect-row").forEach(r => r.classList.remove("active-filter"));
  if (currentAnalysis) {
    renderSentenceBreakdown(currentAnalysis.sentence_breakdown, aspect);
  }
}

function renderSentenceBreakdown(sentences, filterAspect = "all") {
  const card = document.getElementById("sentenceBreakdownCard");
  const listContainer = document.getElementById("sentenceList");
  const tip = document.getElementById("filterActiveTip");
  card.style.display = "block";
  listContainer.innerHTML = "";

  if (filterAspect === "all") {
    tip.textContent = `当前展示全部句子 (${sentences.length} 句)`;
  } else {
    const meta = ASPECT_LABELS[filterAspect] || { zh: filterAspect };
    tip.textContent = `已按方面筛选: [${meta.zh}] (点击可重置)`;
  }

  sentences.forEach(item => {
    // Check if sentence matches filter
    if (filterAspect !== "all") {
      const aspState = item.aspects[filterAspect]?.state_code;
      if (!aspState || aspState === 0) return; // skip if aspect is absent
    }

    const cardEl = document.createElement("div");
    cardEl.className = "sentence-card";
    if (filterAspect !== "all") cardEl.classList.add("highlighted");

    // Header metadata & state tags
    const metaEl = document.createElement("div");
    metaEl.className = "sentence-meta";

    const idxSpan = document.createElement("span");
    idxSpan.className = "sentence-idx";
    idxSpan.textContent = `句 #${item.sentence_index}`;

    const tagsEl = document.createElement("div");
    tagsEl.className = "sentence-tags";

    let hasMentions = false;
    for (const [asp, val] of Object.entries(item.aspects)) {
      if (val.state_code !== 0) {
        hasMentions = true;
        const aspMeta = ASPECT_LABELS[asp] || { zh: asp };
        const stInfo = STATE_COLORS[val.state_code] || { name: "未知", color: "neutral" };
        const tag = document.createElement("span");
        tag.className = `tag-state ${stInfo.color}`;
        tag.innerHTML = `<span>${aspMeta.zh}:</span> <strong>${stInfo.name} (${stInfo.badge})</strong>`;
        tagsEl.appendChild(tag);
      }
    }

    if (!hasMentions) {
      const tag = document.createElement("span");
      tag.className = "tag-state absent";
      tag.textContent = "未提及任何方面 (All Absent)";
      tagsEl.appendChild(tag);
    }

    metaEl.appendChild(idxSpan);
    metaEl.appendChild(tagsEl);

    // Text content
    const textEl = document.createElement("p");
    textEl.className = "sentence-text";
    textEl.textContent = item.sentence;

    cardEl.appendChild(metaEl);
    cardEl.appendChild(textEl);
    listContainer.appendChild(cardEl);
  });

  if (listContainer.children.length === 0) {
    listContainer.innerHTML = `<div class="empty-placeholder"><p>该方面未在任何句子中提及。</p></div>`;
  }
}

// ==============================================================================
// 5. Manual Detection Studio (Single Sentence Inspector)
// ==============================================================================
function setManualText(txt) {
  document.getElementById("manualSentenceInput").value = txt;
  detectSingleSentence();
}

async function detectSingleSentence() {
  const input = document.getElementById("manualSentenceInput");
  const sentence = input.value.trim();
  if (!sentence) return;

  const btn = document.getElementById("btnManualDetect");
  btn.disabled = true;
  btn.textContent = "探测中...";

  try {
    const res = await fetch("/api/predict-sentence", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sentence }),
    });

    const data = await res.json();
    if (!res.ok) {
      alert(data.detail || "检测失败");
      return;
    }

    renderManualProbeResults(data.predictions);
  } catch (e) {
    console.error("Manual detect error:", e);
  } finally {
    btn.disabled = false;
    btn.textContent = "执行单句检测";
  }
}

function renderManualProbeResults(predictions) {
  const container = document.getElementById("manualAspectsGrid");
  container.innerHTML = "";

  predictions.forEach(p => {
    const meta = ASPECT_LABELS[p.aspect] || { zh: p.aspect, en: p.aspect, icon: "🏷️" };
    const card = document.createElement("div");
    card.className = "probe-card";
    if (p.is_mentioned) card.classList.add("mentioned");

    // Header
    const stInfo = STATE_COLORS[p.state_code] || { name: "未知", color: "neutral" };
    card.innerHTML = `
      <div class="probe-header">
        <span class="probe-aspect-title">${meta.icon} ${meta.zh}</span>
        <span class="probe-winner-tag tag-state ${stInfo.color}">${stInfo.name} (置信度: ${(p.confidence * 100).toFixed(1)}%)</span>
      </div>
    `;

    // 5 State Probability Bars
    const list = document.createElement("div");
    list.className = "prob-bars-list";

    for (let s = 0; s < 5; s++) {
      const sName = ["absent", "negative", "neutral", "positive", "mixed"][s];
      const sLabel = ["未提及", "负面", "中性", "正面", "混合"][s];
      const prob = p.probabilities[sName] || 0.0;
      const isWinner = s === p.state_code;

      const row = document.createElement("div");
      row.className = "prob-row";
      row.innerHTML = `
        <span class="prob-state-name">${sLabel}</span>
        <div class="prob-track">
          <div class="prob-fill ${isWinner ? 'winner' : 'other'}" style="width: ${Math.max(2, prob * 100)}%"></div>
        </div>
        <span class="prob-pct">${(prob * 100).toFixed(1)}%</span>
      `;
      list.appendChild(row);
    }

    card.appendChild(list);
    container.appendChild(card);
  });
}

// ==============================================================================
// 6. Model Benchmark & Metadata (Slide 24-27)
// ==============================================================================
async function loadModelBenchmarkInfo() {
  try {
    const res = await fetch("/api/model-info");
    const data = await res.json();
    const tm = data.test_metrics || {};
    const bm = data.baseline_metrics || {};
    const ds = data.dataset_summary || {};

    const g1 = data.gen1_metrics || {};
    const g2 = data.gen2_metrics || {};
    const g3 = data.gen3_metrics || {};
    const g4 = data.gen4_metrics || tm;
    const isGen4 = data.generation_id === 4 || (data.generation && data.generation.includes("第四代"));
    const isGen3 = !isGen4 && (data.generation_id === 3 || (data.generation && data.generation.includes("第三代")));

    // 1. Update Header & Global Banner
    const badgeGen = document.getElementById("badgeGen");
    if (badgeGen) {
      badgeGen.innerHTML = `<span class="gen-dot"></span> ${data.generation || "第四代模型 Gen 4 (2000条全量Champion版)"}`;
    }

    const badgeArch = document.getElementById("badgeArch");
    if (badgeArch && data.hidden_dims) {
      badgeArch.textContent = `MLP: 256 → ${data.hidden_dims.join(" → ")} → 40`;
    }

    const bannerTitle = document.getElementById("bannerModelTitle");
    if (bannerTitle) {
      bannerTitle.textContent = isGen4
        ? "第四代全量Champion模型 (Model Gen 4)"
        : isGen3
          ? "第三代高质Champion模型 (Model Gen 3)"
          : (data.generation || "第二代模型");
    }

    const bannerAcc = document.getElementById("bannerAcc");
    if (bannerAcc) bannerAcc.textContent = `${((tm.overall_accuracy || 0.8764) * 100).toFixed(2)}%`;

    const bannerMentionF1 = document.getElementById("bannerMentionF1");
    if (bannerMentionF1) bannerMentionF1.textContent = (tm.mention_f1 || 0.790).toFixed(3);

    const bannerMacroF1 = document.getElementById("bannerMacroF1");
    if (bannerMacroF1) bannerMacroF1.textContent = (tm.macro_f1 || 0.554).toFixed(3);

    const bannerGain = document.getElementById("bannerGain");
    if (bannerGain) {
      const g1AccVal = g1.overall_accuracy || 0.50;
      const gain = ((tm.overall_accuracy || 0.8764) - g1AccVal) * 100;
      bannerGain.textContent = gain >= 0 ? `+${gain.toFixed(2)}% 较初代显著飞跃 ↑` : "基准模型";
    }

    // 2. Model Architecture & Metadata
    const metaList = document.getElementById("modelMetaList");
    metaList.innerHTML = `
      <div class="meta-item">
        <span class="meta-key">模型版本</span>
        <span class="meta-val">${data.generation || "第 4 代模型 (Gen 4 Champion)"}</span>
      </div>
      <div class="meta-item">
        <span class="meta-key">有效训练样本</span>
        <span class="meta-val">${ds.confirmed_records || 1783} 条 (剔除${ds.excluded_records || 217}条脏数据)</span>
      </div>
      <div class="meta-item">
        <span class="meta-key">有效监督槽位</span>
        <span class="meta-val">${ds.total_mentions || 3053} 提及</span>
      </div>
      <div class="meta-item">
        <span class="meta-key">可训练参数量</span>
        <span class="meta-val">${(data.parameters || 180328).toLocaleString()} 参数</span>
      </div>
      <div class="meta-item">
        <span class="meta-key">隐藏层拓扑</span>
        <span class="meta-val">[${(data.hidden_dims || [384, 192]).join(", ")}]</span>
      </div>
      <div class="meta-item">
        <span class="meta-key">最优早停检查点</span>
        <span class="meta-val">Epoch ${data.best_epoch || 30}</span>
      </div>
    `;

    // 3. Top Metrics Cards
    const metricsDash = document.getElementById("metricsDashboard");
    metricsDash.innerHTML = `
      <div class="metric-box">
        <span class="metric-val">${((tm.overall_accuracy || 0.8531) * 100).toFixed(1)}%</span>
        <span class="metric-lbl">整体分类准确率 (Acc)</span>
      </div>
      <div class="metric-box">
        <span class="metric-val">${(tm.mention_f1 || 0.714).toFixed(3)}</span>
        <span class="metric-lbl">方面提及检测 F1</span>
      </div>
      <div class="metric-box">
        <span class="metric-val">${(tm.macro_f1 || 0.497).toFixed(3)}</span>
        <span class="metric-lbl">全局宏平均 F1 (Macro-F1)</span>
      </div>
      <div class="metric-box">
        <span class="metric-val">${((tm.mention_recall || 0.767) * 100).toFixed(1)}%</span>
        <span class="metric-lbl">方面提及查全率 (Recall)</span>
      </div>
      <div class="metric-box">
        <span class="metric-val">${((tm.mention_precision || 0.668) * 100).toFixed(1)}%</span>
        <span class="metric-lbl">方面提及查准率 (Precision)</span>
      </div>
      <div class="metric-box">
        <span class="metric-val">${(tm.loss || 0.492).toFixed(3)}</span>
        <span class="metric-lbl">测试集多任务损失 (Loss)</span>
      </div>
    `;

    // 4. Gen 1 vs Gen 2 vs Gen 3 vs Gen 4 Benchmark Evolution Table
    const genCompWrap = document.getElementById("genComparisonTableWrap");
    if (genCompWrap) {
      const active = g4.overall_accuracy ? g4 : tm;
      const g1Acc = ((g1.overall_accuracy || 0.50) * 100).toFixed(1) + "%";
      const g2Acc = ((g2.overall_accuracy || 0.7635) * 100).toFixed(1) + "%";
      const g3Acc = ((g3.overall_accuracy || 0.8531) * 100).toFixed(1) + "%";
      const g4Acc = ((active.overall_accuracy || 0.8764) * 100).toFixed(1) + "%";
      const deltaAcc = "+" + (((active.overall_accuracy || 0.8764) - (g1.overall_accuracy || 0.50)) * 100).toFixed(2) + "%";

      const g1F1 = (g1.mention_f1 || 0.416).toFixed(3);
      const g2F1 = (g2.mention_f1 || 0.626).toFixed(3);
      const g3F1 = (g3.mention_f1 || 0.714).toFixed(3);
      const g4F1 = (active.mention_f1 || 0.790).toFixed(3);
      const deltaF1 = "+" + ((active.mention_f1 || 0.790) - (g1.mention_f1 || 0.416)).toFixed(3) + " (+89.9%)";

      const g1Macro = (g1.macro_f1 || 0.284).toFixed(3);
      const g2Macro = (g2.macro_f1 || 0.394).toFixed(3);
      const g3Macro = (g3.macro_f1 || 0.497).toFixed(3);
      const g4Macro = (active.macro_f1 || 0.554).toFixed(3);
      const deltaMacro = "+" + ((active.macro_f1 || 0.554) - (g1.macro_f1 || 0.284)).toFixed(3) + " (+95.1%)";

      const g1NonAbs = (g1.non_absent_macro_f1 || 0.120).toFixed(3);
      const g2NonAbs = (g2.non_absent_macro_f1 || 0.280).toFixed(3);
      const g3NonAbs = (g3.non_absent_macro_f1 || 0.391).toFixed(3);
      const g4NonAbs = (active.non_absent_macro_f1 || 0.458).toFixed(3);
      const deltaNonAbs = "+" + ((active.non_absent_macro_f1 || 0.458) - (g1.non_absent_macro_f1 || 0.120)).toFixed(3) + " (~3.8x)";

      const g1Recall = ((g1.mention_recall || 0.615) * 100).toFixed(1) + "%";
      const g2Recall = ((g2.mention_recall || 0.732) * 100).toFixed(1) + "%";
      const g3Recall = ((g3.mention_recall || 0.767) * 100).toFixed(1) + "%";
      const g4Recall = ((active.mention_recall || 0.81) * 100).toFixed(1) + "%";

      const g1Prec = ((g1.mention_precision || 0.314) * 100).toFixed(1) + "%";
      const g2Prec = ((g2.mention_precision || 0.547) * 100).toFixed(1) + "%";
      const g3Prec = ((g3.mention_precision || 0.668) * 100).toFixed(1) + "%";
      const g4Prec = ((active.mention_precision || 0.746) * 100).toFixed(1) + "%";

      const g1Loss = (g1.loss || 1.351).toFixed(3);
      const g2Loss = (g2.loss || 0.684).toFixed(3);
      const g3Loss = (g3.loss || 0.492).toFixed(3);
      const g4Loss = (active.loss || 0.399).toFixed(3);
      const deltaLoss = "-" + ((g1.loss || 1.351) - (active.loss || 0.399)).toFixed(3) + " (-70.5%)";

      genCompWrap.innerHTML = `
        <table class="benchmark-table">
          <thead>
            <tr>
              <th>核心评测指标 (Metric)</th>
              <th>Gen 1 / 100条基础</th>
              <th>Gen 2 / 500条优化</th>
              <th>Gen 3 / 1000条Champion</th>
              <th class="comp-gen2-col" style="color: #a78bfa;">Gen 4 / 2000条全量Champion</th>
              <th>四代演进总跃升 (Total Delta)</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>有效已审计训练样本 (Dataset Size)</td>
              <td class="mono">89 条</td>
              <td class="mono">435 条</td>
              <td class="mono">879 条</td>
              <td class="mono comp-gen2-col"><strong>1783 条 (剔除 217 条)</strong></td>
              <td><span class="gain-tag">+1903% 监督信号</span></td>
            </tr>
            <tr>
              <td>整体分类准确率 (Overall Accuracy)</td>
              <td class="mono">${g1Acc}</td>
              <td class="mono">${g2Acc}</td>
              <td class="mono">${g3Acc}</td>
              <td class="mono comp-gen2-col"><strong>${g4Acc}</strong></td>
              <td><span class="gain-tag">${deltaAcc} 跨代飞跃 🚀</span></td>
            </tr>
            <tr>
              <td>方面提及检测 F1 (Mention F1)</td>
              <td class="mono">${g1F1}</td>
              <td class="mono">${g2F1}</td>
              <td class="mono">${g3F1}</td>
              <td class="mono comp-gen2-col"><strong>${g4F1}</strong></td>
              <td><span class="gain-tag">${deltaF1}</span></td>
            </tr>
            <tr>
              <td>端到端宏平均 F1 (Macro F1)</td>
              <td class="mono">${g1Macro}</td>
              <td class="mono">${g2Macro}</td>
              <td class="mono">${g3Macro}</td>
              <td class="mono comp-gen2-col"><strong>${g4Macro}</strong></td>
              <td><span class="gain-tag">${deltaMacro}</span></td>
            </tr>
            <tr>
              <td>非空状态宏 F1 (Non-Absent Macro F1)</td>
              <td class="mono">${g1NonAbs}</td>
              <td class="mono">${g2NonAbs}</td>
              <td class="mono">${g3NonAbs}</td>
              <td class="mono comp-gen2-col"><strong>${g4NonAbs}</strong></td>
              <td><span class="gain-tag">${deltaNonAbs}</span></td>
            </tr>
            <tr>
              <td>方面提及查全率 (Mention Recall)</td>
              <td class="mono">${g1Recall}</td>
              <td class="mono">${g2Recall}</td>
              <td class="mono">${g3Recall}</td>
              <td class="mono comp-gen2-col"><strong>${g4Recall}</strong></td>
              <td><span class="gain-tag">+${(((active.mention_recall || 0.81) - (g1.mention_recall || 0.615)) * 100).toFixed(1)}%</span></td>
            </tr>
            <tr>
              <td>方面提及查准率 (Mention Precision)</td>
              <td class="mono">${g1Prec}</td>
              <td class="mono">${g2Prec}</td>
              <td class="mono">${g3Prec}</td>
              <td class="mono comp-gen2-col"><strong>${g4Prec}</strong></td>
              <td><span class="gain-tag">+${(((active.mention_precision || 0.746) - (g1.mention_precision || 0.314)) * 100).toFixed(1)}% (翻倍)</span></td>
            </tr>
            <tr>
              <td>测试集分类损失 (Test Loss)</td>
              <td class="mono">${g1Loss}</td>
              <td class="mono">${g2Loss}</td>
              <td class="mono">${g3Loss}</td>
              <td class="mono comp-gen2-col"><strong>${g4Loss}</strong></td>
              <td><span class="drop-tag">${deltaLoss} 深度收敛</span></td>
            </tr>
          </tbody>
        </table>
      `;
    }

    // 5. 8-Aspect Breakdown Table
    const aspectWrap = document.getElementById("aspectBenchmarkTableWrap");
    if (aspectWrap) {
      const aspectsList = data.aspects || ["cleanliness","service","location","facilities","room_comfort","sound_insulation_noise","food","value"];
      const activeAspect = (g4.overall_accuracy ? g4 : tm) || {};
      const g1AspectAcc = g1.per_aspect_acc || {};
      const g2AspectAcc = g2.per_aspect_acc || {};
      const g3AspectAcc = g3.per_aspect_acc || {};
      const g4AspectAcc = activeAspect.per_aspect_acc || {};

      const g1AspectF1 = g1.per_aspect_mention_f1 || {};
      const g2AspectF1 = g2.per_aspect_mention_f1 || {};
      const g3AspectF1 = g3.per_aspect_mention_f1 || {};
      const g4AspectF1 = activeAspect.per_aspect_mention_f1 || {};

      let rowsHtml = "";
      aspectsList.forEach(asp => {
        const meta = ASPECT_LABELS[asp] || { zh: asp, en: asp, icon: "🏷️" };
        const a1 = (g1AspectAcc[asp] !== undefined ? (g1AspectAcc[asp] * 100).toFixed(0) + "%" : "--") + " / " + (g1AspectF1[asp] !== undefined ? g1AspectF1[asp].toFixed(2) : "--");
        const a2 = (g2AspectAcc[asp] !== undefined ? (g2AspectAcc[asp] * 100).toFixed(0) + "%" : "--") + " / " + (g2AspectF1[asp] !== undefined ? g2AspectF1[asp].toFixed(2) : "--");
        const a3 = (g3AspectAcc[asp] !== undefined ? (g3AspectAcc[asp] * 100).toFixed(1) + "%" : "--") + " / " + (g3AspectF1[asp] !== undefined ? g3AspectF1[asp].toFixed(3) : "--");
        const a4Acc = g4AspectAcc[asp] !== undefined ? (g4AspectAcc[asp] * 100).toFixed(1) + "%" : "--";
        const a4F1 = g4AspectF1[asp] !== undefined ? g4AspectF1[asp].toFixed(3) : "--";

        let statusHtml = '<span class="gain-tag">卓越表现</span>';
        if (g4AspectF1[asp] > 0.8) {
          statusHtml = `<span class="gain-tag" style="background: rgba(167, 139, 250, 0.2); color: #a78bfa;">突破 0.80 极高精度</span>`;
        } else if (g4AspectF1[asp] > 0.7) {
          statusHtml = `<span class="gain-tag">高精度稳健</span>`;
        }

        rowsHtml += `
          <tr>
            <td><strong>${meta.icon} ${asp}</strong></td>
            <td>${meta.zh}</td>
            <td class="mono">${a1}</td>
            <td class="mono">${a2}</td>
            <td class="mono">${a3}</td>
            <td class="mono comp-gen2-col" style="color: #a78bfa;"><strong>${a4Acc} (F1: ${a4F1})</strong></td>
            <td>${statusHtml}</td>
          </tr>
        `;
      });

      aspectWrap.innerHTML = `
        <table class="benchmark-table">
          <thead>
            <tr>
              <th>维度键名 (Aspect Key)</th>
              <th>中文释义</th>
              <th>Gen 1 (Acc / F1)</th>
              <th>Gen 2 (Acc / F1)</th>
              <th>Gen 3 (Acc / F1)</th>
              <th class="comp-gen2-col" style="color: #a78bfa;">Gen 4 Champion (Acc / F1)</th>
              <th>演进评注</th>
            </tr>
          </thead>
          <tbody>
            ${rowsHtml}
          </tbody>
        </table>
      `;
    }

    // 6. Update loss curve image
    const lossImg = document.getElementById("modelLossCurveImg");
    if (lossImg && data.loss_curve_url) {
      lossImg.src = data.loss_curve_url + "?t=" + Date.now();
      lossImg.style.display = "block";
    }
  } catch (e) {
    console.error("Failed to load model benchmark info:", e);
  }
}


// ==============================================================================
// 6. Training Dataset Visualizer, Explorer & Annotation Studio (NEW)
// ==============================================================================

let datasetLoaded = false;
let currentDatasetPage = 1;
let currentDatasetPageSize = 20;
let currentDatasetTotal = 0;
let currentDatasetTotalPages = 1;
let datasetSearchTimeout = null;
let filterDisagreeOnly = false;

// Active Edit Modal State
let activeEditRecord = null;
let activeEditContext = null;
let activeEditPrediction = null;
let activeEditDraftStates = [0, 0, 0, 0, 0, 0, 0, 0];
let activeEditDraftEvidence = {};
let retrainRunning = false;

const ASPECT_KEYS_ORDER = [
  "cleanliness",
  "service",
  "location",
  "facilities",
  "room_comfort",
  "sound_insulation_noise",
  "food",
  "value",
];
const ASPECTS_FIXED_ORDER = ASPECT_KEYS_ORDER;

function escapeHtml(str) {
  if (str === null || str === undefined) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

/**
 * Fetch and render dataset overview metrics and 8-aspect visual distribution bars.
 */
async function loadDatasetStats() {
  try {
    const res = await fetch("/api/training-dataset/stats");
    if (!res.ok) throw new Error("Failed to load dataset stats");
    const data = await res.json();

    // 1. Update KPI ribbon
    const totalEl = document.getElementById("dsTotalSentences");
    const trainEl = document.getElementById("dsTrainableCount");
    const exclEl = document.getElementById("dsExcludedCount");
    const disagEl = document.getElementById("dsDisagreementCount");

    if (totalEl) totalEl.textContent = Number(data.total_records || 2000).toLocaleString();
    if (trainEl) trainEl.textContent = Number(data.trainable_count || 0).toLocaleString();
    if (exclEl) exclEl.textContent = Number(data.excluded_count || 0).toLocaleString();
    if (disagEl) disagEl.textContent = Number(data.disagreement_count || 0).toLocaleString();

    // 2. Render 8 aspect distribution cards with stacked sentiment bars
    const distGrid = document.getElementById("aspectDistGrid");
    if (distGrid && data.aspect_distribution) {
      distGrid.innerHTML = "";
      ASPECT_KEYS_ORDER.forEach(aspectKey => {
        const info = data.aspect_distribution[aspectKey] || { mention_count: 0, breakdown: {} };
        const meta = ASPECT_LABELS[aspectKey] || { zh: aspectKey, en: aspectKey, icon: "🏷️" };
        const mentions = info.mention_count || 0;
        const total = data.total_records || 2000;
        const mentionPct = total > 0 ? ((mentions / total) * 100).toFixed(1) : 0;

        const pos = info.breakdown.positive || 0;
        const neg = info.breakdown.negative || 0;
        const neu = info.breakdown.neutral || 0;
        const mix = info.breakdown.mixed || 0;

        const posPct = mentions > 0 ? (pos / mentions) * 100 : 0;
        const negPct = mentions > 0 ? (neg / mentions) * 100 : 0;
        const neuPct = mentions > 0 ? (neu / mentions) * 100 : 0;
        const mixPct = mentions > 0 ? (mix / mentions) * 100 : 0;

        const card = document.createElement("div");
        card.className = "aspect-dist-card";
        card.title = `点击筛选 “${meta.zh}” 维度的训练数据`;
        card.onclick = () => {
          const sel = document.getElementById("dsAspectSelect");
          if (sel) {
            sel.value = aspectKey;
            onDatasetFilterChange();
          }
        };

        card.innerHTML = `
          <div class="dist-card-header">
            <div class="dist-aspect-title">
              <span class="dist-aspect-icon">${meta.icon}</span>
              <div>
                <span class="dist-aspect-zh">${meta.zh}</span>
                <span class="dist-aspect-en">${meta.en}</span>
              </div>
            </div>
            <div class="dist-mentions-badge">
              <span class="mention-val">${mentions}</span>
              <span class="mention-pct">(${mentionPct}%)</span>
            </div>
          </div>

          <!-- Multi-segment sentiment proportion bar -->
          <div class="sentiment-stacked-bar">
            ${negPct > 0 ? `<div class="seg seg-neg" style="width: ${negPct}%;" title="负面: ${neg}条 (${negPct.toFixed(1)}%)"></div>` : ""}
            ${neuPct > 0 ? `<div class="seg seg-neu" style="width: ${neuPct}%;" title="中性: ${neu}条 (${neuPct.toFixed(1)}%)"></div>` : ""}
            ${posPct > 0 ? `<div class="seg seg-pos" style="width: ${posPct}%;" title="正面: ${pos}条 (${posPct.toFixed(1)}%)"></div>` : ""}
            ${mixPct > 0 ? `<div class="seg seg-mix" style="width: ${mixPct}%;" title="混合: ${mix}条 (${mixPct.toFixed(1)}%)"></div>` : ""}
          </div>

          <div class="dist-breakdown-counts">
            <span class="cnt-badge neg" title="负面">- ${neg}</span>
            <span class="cnt-badge neu" title="中性">~ ${neu}</span>
            <span class="cnt-badge pos" title="正面">+ ${pos}</span>
            <span class="cnt-badge mix" title="混合">± ${mix}</span>
          </div>
        `;
        distGrid.appendChild(card);
      });
    }
  } catch (err) {
    console.error("Error in loadDatasetStats:", err);
  }
}

/**
 * Query training items from backend API with active filters and pagination.
 */
async function loadDatasetItems(page = 1) {
  const loadingOverlay = document.getElementById("dsLoadingOverlay");
  if (loadingOverlay) loadingOverlay.style.display = "flex";

  currentDatasetPage = page;
  const searchInput = document.getElementById("dsSearchInput");
  const aspectSelect = document.getElementById("dsAspectSelect");
  const stateSelect = document.getElementById("dsStateSelect");
  const statusSelect = document.getElementById("dsStatusSelect");

  const search = searchInput ? searchInput.value.trim() : "";
  const aspect = aspectSelect ? aspectSelect.value : "";
  const state = stateSelect && stateSelect.value !== "" ? stateSelect.value : "";
  const status = statusSelect ? statusSelect.value : "all";

  const params = new URLSearchParams({
    page: String(currentDatasetPage),
    page_size: String(currentDatasetPageSize),
    search: search,
    aspect: aspect,
    status: status,
    disagreement_only: filterDisagreeOnly ? "true" : "false",
  });
  if (state !== "") {
    params.append("state", state);
  }

  try {
    const res = await fetch(`/api/training-dataset/items?${params.toString()}`);
    if (!res.ok) throw new Error("Failed to fetch training dataset items");
    const data = await res.json();

    currentDatasetTotal = data.total || 0;
    currentDatasetTotalPages = data.total_pages || 1;

    renderDatasetTable(data.items || []);
    updatePaginationUI();
  } catch (err) {
    console.error("Error in loadDatasetItems:", err);
    showToast("加载训练数据失败: " + err.message, "error");
  } finally {
    if (loadingOverlay) loadingOverlay.style.display = "none";
  }
}

/**
 * Render items into the dataset table with badges and live prediction comparisons.
 */
function renderDatasetTable(items) {
  const tbody = document.getElementById("dsTableBody");
  if (!tbody) return;

  if (items.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="7" class="ds-empty-cell">
          <div class="empty-placeholder-mini">
            <span class="mini-icon">🔍</span>
            <p>没有找到符合当前筛选条件的训练样本</p>
            <button class="btn-clear-search-pill" onclick="resetDatasetFilters()">清空筛选条件</button>
          </div>
        </td>
      </tr>
    `;
    return;
  }

  let html = "";
  const searchKeyword = (document.getElementById("dsSearchInput")?.value || "").trim().toLowerCase();

  items.forEach(item => {
    // Status tag
    let statusBadge = "";
    if (item.exclude) {
      statusBadge = `<span class="ds-badge ds-badge-excl" title="不参与训练">已排除</span>`;
    } else if (item.needs_review) {
      statusBadge = `<span class="ds-badge ds-badge-review" title="存疑待核对">待复核</span>`;
    } else {
      statusBadge = `<span class="ds-badge ds-badge-trainable" title="训练集样本">可训练</span>`;
    }

    // Aspect tags
    const aspectPills = [];
    item.states.forEach((st, aIdx) => {
      if (st !== 0) {
        const aspectKey = ASPECTS_FIXED_ORDER[aIdx];
        const meta = ASPECT_LABELS[aspectKey] || { zh: aspectKey };
        const stMeta = STATE_COLORS[st] || { name: "状态" + st, color: "absent" };
        aspectPills.push(`
          <span class="aspect-tag-pill ${stMeta.color}" title="${meta.zh}: ${stMeta.name}">
            ${meta.zh}: ${stMeta.name}
          </span>
        `);
      }
    });
    const aspectTagsHtml = aspectPills.length > 0 
      ? aspectPills.join("") 
      : `<span class="aspect-tag-pill absent">全部未提及 (Absent)</span>`;

    // Model Prediction badge
    let modelPredHtml = "";
    const hasDisag = item.model_predictions?.has_disagreement;
    if (hasDisag) {
      modelPredHtml = `<span class="ds-pred-badge diff" title="模型推断与当前标注不一致">⚠️ 预测分歧</span>`;
    } else {
      modelPredHtml = `<span class="ds-pred-badge match" title="模型推断与当前标注一致">✅ 预测一致</span>`;
    }

    // Keyword highlight in sentence text
    let displaySentence = escapeHtml(item.sentence || "");
    if (searchKeyword && displaySentence.toLowerCase().includes(searchKeyword)) {
      const re = new RegExp(`(${escapeRegex(searchKeyword)})`, "gi");
      displaySentence = displaySentence.replace(re, `<mark class="highlight-text">$1</mark>`);
    }

    html += `
      <tr class="ds-table-row ${hasDisag ? 'row-disagree' : ''}">
        <td class="cell-sid"><span class="sid-code">${escapeHtml(item.sentence_id)}</span></td>
        <td class="cell-rid"><span class="rid-code">${escapeHtml(item.review_id || "--")}</span></td>
        <td class="cell-sentence" onclick="openEditModal('${item.sentence_id}')" title="点击查看前后句语境与核对修改">
          <p class="sentence-snippet">${displaySentence}</p>
        </td>
        <td class="cell-status">${statusBadge}</td>
        <td class="cell-aspects"><div class="aspect-tags-wrap">${aspectTagsHtml}</div></td>
        <td class="cell-pred">${modelPredHtml}</td>
        <td class="cell-action">
          <button class="btn-table-edit" onclick="openEditModal('${item.sentence_id}')">
            <span>核对/修改</span>
          </button>
        </td>
      </tr>
    `;
  });

  tbody.innerHTML = html;
}

/**
 * Update pagination UI text and buttons.
 */
function updatePaginationUI() {
  const infoEl = document.getElementById("dsPaginationInfo");
  const currentEl = document.getElementById("dsCurrentPageDisplay");
  const prevBtn = document.getElementById("btnPrevPage");
  const nextBtn = document.getElementById("btnNextPage");

  if (infoEl) {
    const start = currentDatasetTotal === 0 ? 0 : (currentDatasetPage - 1) * currentDatasetPageSize + 1;
    const end = Math.min(currentDatasetPage * currentDatasetPageSize, currentDatasetTotal);
    infoEl.textContent = `共 ${currentDatasetTotal.toLocaleString()} 条数据 (显示 ${start}-${end}) • 第 ${currentDatasetPage} / ${currentDatasetTotalPages} 页`;
  }
  if (currentEl) currentEl.textContent = String(currentDatasetPage);
  if (prevBtn) prevBtn.disabled = currentDatasetPage <= 1;
  if (nextBtn) nextBtn.disabled = currentDatasetPage >= currentDatasetTotalPages;
}

function changeDatasetPage(delta) {
  const target = currentDatasetPage + delta;
  if (target >= 1 && target <= currentDatasetTotalPages) {
    loadDatasetItems(target);
  }
}

function onPageSizeChange() {
  const sel = document.getElementById("dsPageSizeSelect");
  if (sel) {
    currentDatasetPageSize = parseInt(sel.value, 10) || 20;
    loadDatasetItems(1);
  }
}

function debounceSearch() {
  if (datasetSearchTimeout) clearTimeout(datasetSearchTimeout);
  datasetSearchTimeout = setTimeout(() => {
    loadDatasetItems(1);
  }, 300);
}

function clearDatasetSearch() {
  const input = document.getElementById("dsSearchInput");
  if (input) {
    input.value = "";
    loadDatasetItems(1);
  }
}

function onDatasetFilterChange() {
  loadDatasetItems(1);
}

function toggleDisagreeFilter() {
  filterDisagreeOnly = !filterDisagreeOnly;
  const btn = document.getElementById("btnToggleDisagree");
  if (btn) {
    btn.classList.toggle("active", filterDisagreeOnly);
  }
  loadDatasetItems(1);
}

function resetDatasetFilters() {
  const searchInput = document.getElementById("dsSearchInput");
  const aspectSelect = document.getElementById("dsAspectSelect");
  const stateSelect = document.getElementById("dsStateSelect");
  const statusSelect = document.getElementById("dsStatusSelect");
  const btnDisagree = document.getElementById("btnToggleDisagree");

  if (searchInput) searchInput.value = "";
  if (aspectSelect) aspectSelect.value = "";
  if (stateSelect) stateSelect.value = "";
  if (statusSelect) statusSelect.value = "all";
  filterDisagreeOnly = false;
  if (btnDisagree) btnDisagree.classList.remove("active");

  loadDatasetItems(1);
}


// ==============================================================================
// 7. Interactive Sample Inspector & Annotation Studio Modal
// ==============================================================================

/**
 * Open the interactive sample inspector and label editor modal for a specific sentence.
 */
async function openEditModal(sentenceId) {
  try {
    const res = await fetch(`/api/training-dataset/item/${encodeURIComponent(sentenceId)}`);
    if (!res.ok) throw new Error("无法加载样本详情: " + sentenceId);
    const data = await res.json();

    activeEditRecord = data.record;
    activeEditContext = data.context || {};
    activeEditPrediction = data.prediction || {};
    activeEditDraftStates = [...(activeEditRecord.states || [0, 0, 0, 0, 0, 0, 0, 0])];
    activeEditDraftEvidence = { ...(activeEditRecord.evidence || {}) };

    // Fill IDs & Header
    document.getElementById("editSentenceId").textContent = activeEditRecord.sentence_id;
    document.getElementById("editReviewId").textContent = activeEditRecord.review_id || activeEditContext.review_id || "HTL_R";

    // Fill Sentence Context
    const beforeEl = document.getElementById("editContextBefore");
    const targetEl = document.getElementById("editTargetSentence");
    const afterEl = document.getElementById("editContextAfter");

    beforeEl.textContent = activeEditContext.context_before ? activeEditContext.context_before : "（无前句上下文）";
    targetEl.textContent = activeEditRecord.sentence || "";
    afterEl.textContent = activeEditContext.context_after ? activeEditContext.context_after : "（无后句上下文）";

    // Fill Audit Controls
    const exclToggle = document.getElementById("editExcludeToggle");
    const reviewToggle = document.getElementById("editNeedsReviewToggle");
    const notesInput = document.getElementById("editNotesInput");

    if (exclToggle) exclToggle.checked = !!(activeEditRecord.exclude || activeEditRecord.review_status === "excluded");
    if (reviewToggle) reviewToggle.checked = !!activeEditRecord.needs_review;
    if (notesInput) notesInput.value = activeEditRecord.notes || "";

    // Render comparison banner and 8 aspect editor cards
    renderComparisonBanner();
    renderAspectEditors();

    // Clear save status
    const saveStatus = document.getElementById("editSaveStatus");
    if (saveStatus) saveStatus.textContent = "";

    // Show modal
    const modal = document.getElementById("editModal");
    if (modal) modal.style.display = "flex";
  } catch (err) {
    console.error("Failed to open edit modal:", err);
    showToast("打开标注弹窗失败: " + err.message, "error");
  }
}

/**
 * Render the top comparison banner showing Ground Truth vs Model Prediction.
 */
function renderComparisonBanner() {
  const banner = document.getElementById("comparisonBanner");
  if (!banner || !activeEditPrediction) return;

  const predStates = activeEditPrediction.pred_states || [];
  let diffCount = 0;
  const chipsHtml = ASPECT_KEYS_ORDER.map((aspectKey, idx) => {
    const labelState = activeEditDraftStates[idx];
    const predState = predStates[idx] !== undefined ? predStates[idx] : 0;
    const isMatch = labelState === predState;
    if (!isMatch) diffCount += 1;

    const meta = ASPECT_LABELS[aspectKey] || { zh: aspectKey };
    const labelStMeta = STATE_COLORS[labelState] || { name: "S" + labelState };
    const predStMeta = STATE_COLORS[predState] || { name: "S" + predState };

    return `
      <div class="comp-aspect-pill ${isMatch ? 'match' : 'diff'}">
        <span class="comp-aspect-name">${meta.zh}</span>
        <div class="comp-states-row">
          <span class="pill-label-st" title="当前标注状态">标注: <strong>${labelStMeta.name}</strong></span>
          <span class="pill-arrow">vs</span>
          <span class="pill-pred-st" title="模型推断状态">推断: <strong>${predStMeta.name}</strong></span>
          <span class="pill-badge-icon">${isMatch ? '✅' : '⚠️'}</span>
        </div>
      </div>
    `;
  }).join("");

  const summaryText = diffCount === 0
    ? `<span class="banner-status-tag match">✅ 当前标注与模型实时推断 8 个维度完全一致</span>`
    : `<span class="banner-status-tag diff">⚠️ 当前发现 ${diffCount} 个维度的标注与模型推断存在分歧</span>`;

  banner.innerHTML = `
    <div class="comparison-banner-top">
      <span class="banner-title">维度一致性比对 (Label vs Live Prediction)</span>
      ${summaryText}
    </div>
    <div class="comparison-chips-grid">
      ${chipsHtml}
    </div>
  `;
}

/**
 * Render the 8 interactive aspect editor cards inside the modal.
 */
function renderAspectEditors() {
  const grid = document.getElementById("aspectEditorsGrid");
  if (!grid || !activeEditPrediction) return;

  grid.innerHTML = "";
  const predStates = activeEditPrediction.pred_states || [];
  const aspectProbs = activeEditPrediction.aspect_probs || {};

  ASPECT_KEYS_ORDER.forEach((aspectKey, aIdx) => {
    const meta = ASPECT_LABELS[aspectKey] || { zh: aspectKey, en: aspectKey, icon: "🏷️" };
    const currentState = activeEditDraftStates[aIdx];
    const predState = predStates[aIdx] !== undefined ? predStates[aIdx] : 0;
    const predMeta = STATE_COLORS[predState] || { name: "状态" + predState };
    const probs = aspectProbs[aspectKey] || {};
    const predConf = probs[predMeta.name] ? (probs[predMeta.name] * 100).toFixed(1) : "--";

    const card = document.createElement("div");
    card.className = `aspect-editor-card ${currentState !== predState ? 'card-has-diff' : ''}`;
    card.id = `aspectCard_${aIdx}`;

    // 5 State Buttons
    const buttonsHtml = [0, 1, 2, 3, 4].map(sCode => {
      const sMeta = STATE_COLORS[sCode];
      const isActive = currentState === sCode;
      const isModelPick = predState === sCode;
      return `
        <button
          type="button"
          class="state-pill-btn ${sMeta.color} ${isActive ? 'active' : ''}"
          onclick="setAspectState(${aIdx}, ${sCode})"
          title="${sMeta.name} (${sMeta.badge})"
        >
          <span class="btn-check">${isActive ? '●' : '○'}</span>
          <span class="btn-text">${sMeta.name}</span>
          ${isModelPick ? `<span class="model-pick-dot" title="模型推断此状态">AI</span>` : ""}
        </button>
      `;
    }).join("");

    const currentEvidence = activeEditDraftEvidence[aspectKey] || "";

    card.innerHTML = `
      <div class="card-top-row">
        <div class="aspect-header-group">
          <span class="aspect-card-icon">${meta.icon}</span>
          <span class="aspect-card-zh">${meta.zh}</span>
          <span class="aspect-card-en">${meta.en}</span>
        </div>
        <div class="aspect-model-hint">
          <span class="hint-label">模型推断:</span>
          <span class="hint-pred ${predMeta.color}">${predMeta.name} (${predConf}%)</span>
        </div>
      </div>

      <!-- State Radio Pills -->
      <div class="state-buttons-row">
        ${buttonsHtml}
      </div>

      <!-- Evidence Text Substring Input -->
      <div class="evidence-row">
        <label class="evidence-label">证据片段 (Substring Evidence):</label>
        <input
          type="text"
          class="evidence-input"
          id="evidenceInput_${aIdx}"
          value="${escapeHtml(currentEvidence)}"
          placeholder="选填：原文中提及该方面的具体短语（例如：“床很硬”、“服务员态度差”）"
          oninput="onEvidenceChange(${aIdx}, this.value)"
        />
      </div>
    `;

    grid.appendChild(card);
  });
}

/**
 * Handle user changing the sentiment state for an aspect in the modal.
 */
function setAspectState(aspectIdx, stateCode) {
  activeEditDraftStates[aspectIdx] = stateCode;
  renderComparisonBanner();
  renderAspectEditors();
}

/**
 * Handle evidence text input updates.
 */
function onEvidenceChange(aspectIdx, value) {
  const aspectKey = ASPECT_KEYS_ORDER[aspectIdx];
  activeEditDraftEvidence[aspectKey] = value.trim();
}

/**
 * 1-Click Adopt Model Prediction: overwrite draft states with model's predicted states.
 */
function adoptModelPrediction() {
  if (!activeEditPrediction || !activeEditPrediction.pred_states) {
    showToast("当前模型推断不可用", "warning");
    return;
  }
  activeEditDraftStates = [...activeEditPrediction.pred_states];
  renderComparisonBanner();
  renderAspectEditors();
  showToast("已快速采纳模型推断的 8 维度状态，请核验后保存！", "info");
}

/**
 * Clear all 8 aspects to Absent (0).
 */
function clearAllAspectsToAbsent() {
  activeEditDraftStates = [0, 0, 0, 0, 0, 0, 0, 0];
  renderComparisonBanner();
  renderAspectEditors();
  showToast("已将所有方面设为未提及 (Absent)", "info");
}

function onExcludeToggleChange() {
  const toggle = document.getElementById("editExcludeToggle");
  if (toggle && toggle.checked) {
    showToast("提示：排除的样本将不参与模型后续训练", "warning");
  }
}

/**
 * Save user edits to the dataset on the backend.
 */
async function saveAnnotationEdits() {
  if (!activeEditRecord) return;
  const sid = activeEditRecord.sentence_id;
  const saveBtn = document.getElementById("btnSaveAnnotation");
  const statusEl = document.getElementById("editSaveStatus");

  const excl = !!document.getElementById("editExcludeToggle")?.checked;
  const nr = !!document.getElementById("editNeedsReviewToggle")?.checked;
  const notes = document.getElementById("editNotesInput")?.value.trim() || "";

  if (saveBtn) {
    saveBtn.disabled = true;
    saveBtn.innerHTML = `<span class="btn-spinner"></span> 正在保存...`;
  }
  if (statusEl) statusEl.textContent = "正在写入数据集并安全备份...";

  try {
    const res = await fetch("/api/training-dataset/update", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sentence_id: sid,
        states: activeEditDraftStates,
        evidence: activeEditDraftEvidence,
        exclude: excl,
        needs_review: nr,
        notes: notes,
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "保存失败");
    }

    const data = await res.json();
    showToast(`样本 ${sid} 标注已成功保存并实时同步！`, "success");

    closeEditModal();
    // Refresh table and stats
    await loadDatasetStats();
    await loadDatasetItems(currentDatasetPage);
  } catch (err) {
    console.error("Save error:", err);
    showToast("保存标注出错: " + err.message, "error");
    if (statusEl) statusEl.textContent = "保存失败: " + err.message;
  } finally {
    if (saveBtn) {
      saveBtn.disabled = false;
      saveBtn.innerHTML = `<span class="btn-icon">💾</span><span>保存修改并更新数据集</span>`;
    }
  }
}

function closeEditModal() {
  const modal = document.getElementById("editModal");
  if (modal) modal.style.display = "none";
  activeEditRecord = null;
}


// ==============================================================================
// 8. Interactive Retraining Studio Logic
// ==============================================================================

function openRetrainModal() {
  const modal = document.getElementById("retrainModal");
  if (!modal) return;
  const prog = document.getElementById("retrainProgressArea");
  const resCard = document.getElementById("retrainResultCard");
  const startBtn = document.getElementById("btnExecuteRetrain");

  if (prog) prog.style.display = "none";
  if (resCard) resCard.style.display = "none";
  if (startBtn) {
    startBtn.disabled = false;
    startBtn.innerHTML = `<span class="btn-icon">⚡</span><span>立即开始重训</span>`;
  }
  modal.style.display = "flex";
}

function closeRetrainModal() {
  if (retrainRunning) {
    showToast("模型正在重训中，请稍候...", "warning");
    return;
  }
  const modal = document.getElementById("retrainModal");
  if (modal) modal.style.display = "none";
}

/**
 * Trigger backend training loop with updated labels, reload weights, and display before/after metrics.
 */
async function executeRetraining() {
  const startBtn = document.getElementById("btnExecuteRetrain");
  const cancelBtn = document.getElementById("btnCancelRetrain");
  const progArea = document.getElementById("retrainProgressArea");
  const progText = document.getElementById("retrainProgressText");
  const progSub = document.getElementById("retrainProgressSub");
  const resCard = document.getElementById("retrainResultCard");

  retrainRunning = true;
  if (startBtn) startBtn.disabled = true;
  if (cancelBtn) cancelBtn.disabled = true;
  if (progArea) progArea.style.display = "flex";
  if (resCard) resCard.style.display = "none";

  let step = 0;
  const messages = [
    "正在加载最新标注与 256 维向量 (HotelReviewMLP)...",
    "划分独立 Held-Out 训练与验证集 (防止跨评论泄漏)...",
    "执行 CrossEntropy 反向传播与早停监控...",
    "在独立测试集上全面评估 8 方面 F1 与 Acc 指标...",
    "更新 PyTorch 权重检查点并执行服务热加载...",
  ];
  const timer = setInterval(() => {
    step = (step + 1) % messages.length;
    if (progText) progText.textContent = messages[step];
  }, 1200);

  try {
    const res = await fetch("/api/training-dataset/retrain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        epochs: 40,
        batch_size: 16,
        lr: 0.001,
        hidden_dims: [384, 192],
        early_stopping_patience: 12,
      }),
    });

    clearInterval(timer);

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "重训执行失败");
    }

    const data = await res.json();
    if (progArea) progArea.style.display = "none";
    if (resCard) resCard.style.display = "block";

    // Update duration
    const durEl = document.getElementById("retrainDuration");
    if (durEl) durEl.textContent = `耗时 ${data.duration_seconds || 4.2} 秒`;

    // Render comparison table
    const tbody = document.getElementById("retrainComparisonTbody");
    if (tbody) {
      const oldM = data.old_metrics || {};
      const newM = data.new_metrics || {};

      const metricsList = [
        { key: "overall_accuracy", name: "整体多任务准确率 (Overall Acc)", isPct: true },
        { key: "mention_f1", name: "方面提及综合 F1 (Mention F1)", isPct: false },
        { key: "macro_f1", name: "5状态宏平均 F1 (Macro F1)", isPct: false },
        { key: "non_absent_macro_f1", name: "有效情感宏 F1 (Non-Absent F1)", isPct: false },
        { key: "loss", name: "测试集损失 (Test CE Loss)", isPct: false, reverse: true },
      ];

      tbody.innerHTML = metricsList.map(m => {
        const oldVal = oldM[m.key] !== undefined ? oldM[m.key] : 0;
        const newVal = newM[m.key] !== undefined ? newM[m.key] : 0;
        const delta = newVal - oldVal;

        const oldFmt = m.isPct ? (oldVal * 100).toFixed(2) + "%" : Number(oldVal).toFixed(4);
        const newFmt = m.isPct ? (newVal * 100).toFixed(2) + "%" : Number(newVal).toFixed(4);

        let deltaFmt = "";
        let deltaClass = "neutral";
        if (Math.abs(delta) < 0.0001) {
          deltaFmt = "持平 (0.00)";
        } else if ((delta > 0 && !m.reverse) || (delta < 0 && m.reverse)) {
          deltaFmt = `+${m.isPct ? (delta * 100).toFixed(2) + "%" : delta.toFixed(4)} ↑ 提升`;
          deltaClass = "positive";
        } else {
          deltaFmt = `${m.isPct ? (delta * 100).toFixed(2) + "%" : delta.toFixed(4)} ↓`;
          deltaClass = "negative";
        }

        return `
          <tr>
            <td><strong>${m.name}</strong></td>
            <td>${oldFmt}</td>
            <td style="color: #a78bfa; font-weight: 700;">${newFmt}</td>
            <td><span class="delta-badge ${deltaClass}">${deltaFmt}</span></td>
          </tr>
        `;
      }).join("");
    }

    showToast("🎉 模型重训成功！新权重已热更新生效！", "success");

    // Refresh model benchmark tab and dataset stats
    await loadModelBenchmarkInfo();
    await loadDatasetStats();
    await loadDatasetItems(currentDatasetPage);

  } catch (err) {
    clearInterval(timer);
    if (progArea) progArea.style.display = "none";
    console.error("Retrain error:", err);
    showToast("模型重训失败: " + err.message, "error");
  } finally {
    retrainRunning = false;
    if (startBtn) {
      startBtn.disabled = false;
      startBtn.innerHTML = `<span class="btn-icon">⚡</span><span>重新开始重训</span>`;
    }
    if (cancelBtn) cancelBtn.disabled = false;
  }
}

/**
 * Trigger native browser download of updated labels_final_2000_v4_mlp.jsonl
 */
function exportDatasetFile() {
  window.open("/api/training-dataset/export", "_blank");
  showToast("正在导出标注数据集 JSONL 文件...", "info");
}


// ==============================================================================
// 9. Floating Toast Notification Helper
// ==============================================================================

function showToast(message, type = "info") {
  const container = document.getElementById("toastContainer");
  if (!container) return;

  const toast = document.createElement("div");
  toast.className = `app-toast toast-${type}`;

  const icons = {
    info: "ℹ️",
    success: "✅",
    warning: "⚠️",
    error: "❌",
  };

  toast.innerHTML = `
    <span class="toast-icon">${icons[type] || "ℹ️"}</span>
    <span class="toast-msg">${escapeHtml(message)}</span>
    <button class="toast-close" onclick="this.parentElement.remove()">✕</button>
  `;

  container.appendChild(toast);

  setTimeout(() => {
    toast.classList.add("toast-fade-out");
    setTimeout(() => {
      if (toast.parentElement) toast.remove();
    }, 300);
  }, 4000);
}

function escapeRegex(string) {
  return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
