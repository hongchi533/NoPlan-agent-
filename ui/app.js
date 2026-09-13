/* ─── 记忆助手 前端逻辑 ────────────────────── */

// 标签配色：前端所有（纯视图关注点，后端不感知）。加新标签时在这里补配色。
const TAG_COLORS = {
    "工作":{bg:"#E8EDFF",border:"#4F6AFF",text:"#2C3E8F"},
    "生活":{bg:"#FFF3E0",border:"#FF9800",text:"#E65100"},
    "运动":{bg:"#E8F5E9",border:"#4CAF50",text:"#1B5E20"},
    "社交":{bg:"#FCE4EC",border:"#E91E63",text:"#880E4F"},
    "学习":{bg:"#F3E5F5",border:"#9C27B0",text:"#4A148C"},
    "饮食":{bg:"#FFF8E1",border:"#FFC107",text:"#F57F17"},
    "娱乐":{bg:"linear-gradient(135deg,#FFE3B3,#FFB088)",border:"#FF8A5C",text:"#9C4221"},  // 日落渐变：金黄→珊瑚，暖而愉快
    "家务":{bg:"#E1F5FE",border:"#03A9F4",text:"#01579A"},
    "惊喜":{bg:"#FFEBEE",border:"#F44336",text:"#B71C1C"}
};
const DEFAULT_COLOR = {bg:"#F5F5F5",border:"#9E9E9E",text:"#424242"};
// mood emoji 由 /api/init 下发（后端生成回复/检索文本也用它，是真正的跨层共享常量）
let MOOD_EMOJI = {};
const WEEKDAY_NAMES = ["一","二","三","四","五","六","日"];
const TODAY = new Date().toISOString().split("T")[0];
const THIS_YEAR = new Date().getFullYear();

// 当前视图状态
let currentView = "home";
let currentMonth = TODAY.substring(0,7);  // "2026-08"
let currentWeekDate = TODAY;
let currentDayDate = TODAY;

// ─── 初始化 ──────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("todayDate").textContent = TODAY;
    loadInit();
    purgeLegacyRemindKeys();
    pollReminders();                    // 开页立刻领一次（页面关着期间攒下的，保鲜期内才会弹）
    setInterval(pollReminders, 30000);  // 之后每 30s 领一次信箱
    // 总览当天固定，不进周期轮询：回到页面的那一刻查一次（撤昨天的/补今天的）
    document.addEventListener("visibilitychange", () => {
        if (!document.hidden) maybeRefreshOverview();
    });
    document.getElementById("userInput").addEventListener("keydown", (e) => {
        if (e.key === "Enter") handleSend();
    });
});

// ─── Tab 切换 ────────────────────────────────
function switchTab(view) {
    currentView = view;
    document.querySelectorAll(".tab").forEach(t => t.classList.toggle("active", t.dataset.view === view));
    document.querySelectorAll(".view-panel").forEach(p => p.classList.add("hidden"));
    document.getElementById("view" + view.charAt(0).toUpperCase() + view.slice(1)).classList.remove("hidden");

    if (view === "month") loadMonthView();
    else if (view === "week") loadWeekView();
    else if (view === "day") loadDayView();
    else if (view === "tasks") loadTasks();
}

// ─── 发送消息（SSE 流式：等待动画 + 逐字渲染）───
// 发送永远即时受理（输入/按钮不锁，观感第一），消化端串行：本地 FIFO 队列 +
// 单消费者——同一时刻只有一条请求在飞，回复不会串流；服务端单飞轮锁是第二道
// 兜底（多标签页 / curl / Enter 连击等绕过本队列的路径）
let sendQueue = [];
let sending = false;

function handleSend() {
    const input = document.getElementById("userInput");
    const message = input.value.trim();
    if (!message) return;
    askNotifyPermission();   // 趁用户手势申请通知权限（错过手势浏览器会静默拒绝）
    input.value = "";        // 消息已被队列收走：立刻清空，随时可以继续打下一条
    // queued 标记"发送时上一条还在处理"：这轮回复追加成新卡（排队系列不覆盖）；
    // 空闲时发的消息立即开跑，新回复整段替换旧卡（闲时聊天不留流水）
    sendQueue.push({ text: message, queued: sending });
    updateQueueHint();
    drainSend();
}

function updateQueueHint() {
    const el = document.getElementById("queueHint");
    if (!el) return;
    const n = sendQueue.length;
    el.classList.toggle("hidden", n === 0);
    el.textContent = n > 0 ? `已收到，前面还有 ${n} 条在排队，会按顺序处理～` : "";
}

async function drainSend() {
    if (sending || !sendQueue.length) return;
    sending = true;
    const item = sendQueue.shift();
    const message = item.text;
    updateQueueHint();
    setSendLoading(true);
    if (!item.queued) document.getElementById("replyLog").innerHTML = "";   // 空闲后的新请求：旧回复整段撤下
    startThinking(true);   // 排队系列：开新卡，上一轮的回复留在上面
    let final = null;
    try {
        const res = await fetch("/api/chat/stream", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message }),
            signal: AbortSignal.timeout(180000)   // agent loop 最坏 8 轮迭代，给足余量
        });
        if (!res.ok || !res.body) throw new Error("网络连接失败");
        const reader = res.body.getReader(), dec = new TextDecoder();
        let buf = "";
        for (;;) {
            const { done, value } = await reader.read();
            if (done) break;
            buf += dec.decode(value, { stream: true });
            let i;
            while ((i = buf.indexOf("\n\n")) >= 0) {   // SSE 事件以空行分隔
                const raw = buf.slice(0, i); buf = buf.slice(i + 2);
                const line = raw.split("\n").find(l => l.startsWith("data: "));
                if (!line) continue;
                const ev = JSON.parse(line.slice(6));
                if (ev.type === "status") applyStatus(ev);
                else if (ev.type === "delta") appendDelta(ev.text);
                else if (ev.type === "reset") startThinking();   // 罕见：文本流到一半转工具，清回等待态
                else if (ev.type === "done") final = ev;
            }
        }
        if (!final) throw new Error("回复流意外中断");
        finishStream(final);
        if (final.tool_calls_log && final.tool_calls_log.length > 0) {
            showDebug(`循环${final.iterations}轮: ${final.tool_calls_log.map(t=>t.tool).join(" → ")}`);
        }
        loadInit();
    } catch (e) {
        stopThinking();
        // 失败写进当前轮的卡：残缺的流式分片被错误结果整卡覆写，这轮的结局可见
        const msg = "⚠️ " + e.message + "，这条没有发送成功，稍后再试～";
        if (activeCard && activeCard.isConnected) activeCard.textContent = msg;
        else showReply(msg);
    } finally {
        sending = false;
        setSendLoading(false);
        drainSend();   // 队列里还有就自动续传下一条
    }
}

// ─── 流式等待与逐字渲染 ─────────────────────
// 等待态：随机可爱的「图标+短语」轮换（转圈不空等）；工具事件来了换工具专属文案，
// 等待期间也能看见 agent 在干嘛；done.reply 是权威回复（诚实覆写发生在它之前）
const THINK_POOL = [
    ["🤔","正在思考"],["💭","想一想"],["🌱","整理思绪"],["✨","灵感加载中"],
    ["🐰","努力在想"],["🫧","头脑风暴"],["🌙","慢慢琢磨"],["🧸","认真想"],["🍃","灵光乍现前"]
];
const TOOL_STATUS = {
    parse_and_record:["📝","正在记下…"], get_overview:["📅","翻翻日程…"],
    search_memory:["🔍","在记忆里找找…"], maps_weather:["🌤","看看天气…"],
    register_task:["⏰","定个提醒…"], list_tasks:["📋","数数提醒…"], delete_task:["🗑","取消提醒…"],
    find_plan:["🔎","对一下日程…"], complete_plan:["✅","标记完成…"],
    update_event:["✏️","改一下日程…"], consolidate:["🧹","整理记忆…"], holiday_info:["🏖","查查假期…"],
    recall_capsule:["📦","开时光胶囊…"], recall_last_year:["🗓","翻去年的今天…"]
};
let _thinkTimer = null, _thinkIdx = -1;

// 当前轮回复卡：thinking/流式/最终文本都写进这一张；轮与轮之间追加新卡不覆盖。
// fresh=true 新开一张（新一轮消息开始）；缺省复用当前卡（轮内多轮思考/工具状态切换）
let activeCard = null;

function startThinking(fresh = false) {
    const card = fresh || !activeCard || !activeCard.isConnected ? newReplyCard() : activeCard;
    card.innerHTML = '<div class="thinking-row"><span class="thinking-emoji"></span>'
        + '<span class="thinking-label"></span><span class="thinking-dots"><i></i><i></i><i></i></span></div>';
    rotateThinking();
    clearInterval(_thinkTimer);
    _thinkTimer = setInterval(rotateThinking, 1600);
}

function newReplyCard() {
    const log = document.getElementById("replyLog");
    document.getElementById("agentReply").classList.remove("hidden");
    activeCard = document.createElement("div");
    activeCard.className = "reply-card";
    log.appendChild(activeCard);
    trimReplyLog();
    activeCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
    return activeCard;
}

function trimReplyLog() {
    const log = document.getElementById("replyLog");
    while (log.children.length > 5) log.removeChild(log.firstChild);   // 保最近 5 张，日志不至于顶飞页面
}

function rotateThinking() {
    const emoji = document.querySelector(".thinking-emoji"), label = document.querySelector(".thinking-label");
    if (!emoji || !label) return;
    let i; do { i = Math.floor(Math.random() * THINK_POOL.length); } while (i === _thinkIdx);
    _thinkIdx = i;
    emoji.textContent = THINK_POOL[i][0];
    label.textContent = THINK_POOL[i][1];
}

function applyStatus(ev) {
    if (ev.stage === "thinking") { startThinking(); return; }   // 每轮 LLM 调用前都会来：回到随机轮换
    if (!document.querySelector(".thinking-row")) startThinking();
    const pair = TOOL_STATUS[ev.tool] || ["🤖","忙一小会儿…"];
    clearInterval(_thinkTimer);   // 工具文案固定，暂停随机轮换
    const emoji = document.querySelector(".thinking-emoji"), label = document.querySelector(".thinking-label");
    if (emoji && label) { emoji.textContent = pair[0]; label.textContent = pair[1]; }
}

function appendDelta(text) {
    let span = document.getElementById("streamText");
    if (!span) {   // 第一个分片：当前卡等待态 → 打字态
        clearInterval(_thinkTimer);
        activeCard.innerHTML = '<span id="streamText"></span><span class="stream-cursor"></span>';
        span = document.getElementById("streamText");
    }
    span.textContent += text;
}

function stopThinking() { clearInterval(_thinkTimer); _thinkTimer = null; }

function finishStream(ev) {
    stopThinking();
    activeCard.textContent = ev.reply;   // done.reply 是权威文本，整卡覆写流式分片
}

// ─── 首页数据 ────────────────────────────────
async function loadInit() {
    try {
        const res = await api("/api/init", "GET");
        // 静默失败是最坏的失败：列表停在旧数据上，用户毫无察觉
        if (!res.ok) { showReply("⚠️ 数据加载失败了，页面显示可能不是最新的，稍后再刷新试试～"); return; }
        // 后端下发的 mood emoji（在首次渲染前就位）
        if (res.mood_emoji) MOOD_EMOJI = res.mood_emoji;
        renderEvents(res.events);
        renderFutureEvents(res.future_events);
        renderMoods(res.moods);
        renderOverview(res.overview);
        renderProactive(res.proactive);
    } catch (e) { console.error(e); }
}

function renderEvents(events) {
    const c = document.getElementById("eventList");
    if (!events || !events.length) { c.innerHTML = '<p class="empty-hint">还没有记录，说点什么吧 ✨</p>'; return; }
    const done = events.filter(e=>e.type==="done"), plan = events.filter(e=>e.type==="plan");
    let h = "";
    for (const e of done) h += renderEventItem(e,"done");
    if (done.length && plan.length) h += '<div style="border-top:1px dashed #ddd;margin:4px 0"></div>';
    for (const e of plan) h += renderEventItem(e,"plan");
    c.innerHTML = h;
}

function renderEventItem(e, type) {
    const isDone = type==="done";
    const tags = (e.tags||[]).map(t=>{
        const c = TAG_COLORS[t] || DEFAULT_COLOR;
        return `<span class="tag" style="background:${c.bg};color:${c.text}">${t}</span>`;
    }).join("");
    const moment = e.is_moment ? '<span class="moment-badge">MOMENT</span>' : "";
    const oc = isDone ? "" : `onclick="completePlan('${e.id}')"`;
    return `<div class="event-item event-${type}" data-event-id="${e.id}">
        <div class="event-check ${isDone?"check-done":"check-plan"}" ${oc}>${isDone?"✓":""}</div>
        <span class="event-time">${e.start_time||"全天"}</span>
        <span class="event-title">${e.title}</span>${moment}<div class="event-tags">${tags}</div>
        <button class="event-del" title="删除" onclick="askDelete('${e.id}', this)">✕</button></div>`;
}

// ─── 今日总览（07:00 触发落盘，当日有效；到点前不占位）───
let overviewDay = "";   // 已渲染总览的日期：等于今天就不重复拉，跨天/未生成才重新拉

function renderOverview(o) {
    const s = document.getElementById("overviewSection");
    if (!s) return;
    if (!o || !o.text) { s.classList.add("hidden"); return; }
    document.getElementById("overviewText").textContent = o.text;
    s.classList.remove("hidden");
    overviewDay = o.date;
}

// 主动推送（/api/init 下发）：去年今天 / 时光胶囊。
// 后端没下发就整块隐藏不占位——去年今天周年才响、胶囊进唤起窗才有候选
function renderProactive(pro) {
    const sec = document.getElementById("recallSection");
    if (!sec) return;
    const lyCard = document.getElementById("lastYearToday");
    const capCard = document.getElementById("timeCapsule");
    lyCard.classList.add("hidden");
    capCard.classList.add("hidden");
    if (!pro) { sec.classList.add("hidden"); return; }
    // 文本已是可直接上屏的成品（叙述文案一整段，或原文兜底的条目行），均无标题行
    const items = t => t.split("\n").map(s => s.trim()).filter(Boolean);
    if (pro.last_year) {
        document.getElementById("lastYearContent").innerHTML =
            items(pro.last_year).map(l => `<div class="recall-item">${l}</div>`).join("");
        lyCard.classList.remove("hidden");
    }
    if (pro.capsule) {
        document.getElementById("capsuleContent").innerHTML =
            items(pro.capsule).map(l => `<div class="recall-item">${l}</div>`).join("");
        capCard.classList.remove("hidden");
    }
    const any = !lyCard.classList.contains("hidden") || !capCard.classList.contains("hidden");
    if (any) sec.classList.remove("hidden"); else sec.classList.add("hidden");
}

async function refreshOverview() {
    try {
        const res = await api("/api/overview", "GET", null, 10000);
        if (res.ok) renderOverview(res.overview);
    } catch (_) { /* 静默：30s 后随下一片重试 */ }
}

function maybeRefreshOverview() {
    // 总览当天生成后固定，不需要周期拉：只在「回到页面」这类事件时查——
    // 跨天挂着页面时撤下昨天的；今天的还没上屏则补拉（生成由横幅送达触发刷新）
    const d = new Date(), pad = n => String(n).padStart(2, "0");
    const today = d.getFullYear() + "-" + pad(d.getMonth() + 1) + "-" + pad(d.getDate());
    if (overviewDay === today) return;
    refreshOverview();
}

function renderFutureEvents(events) {
    const c = document.getElementById("futureList");
    if (!events || !events.length) { c.innerHTML = '<p class="empty-hint">暂无计划</p>'; return; }
    events = events.slice(0, 3); // 首页只展示最近3条，其余去日历视图看
    const grouped = {};
    for (const e of events) { if (!grouped[e.date]) grouped[e.date] = []; grouped[e.date].push(e); }
    let h = "";
    for (const [d, items] of Object.entries(grouped)) {
        const wd = WEEKDAY_NAMES[new Date(d).getDay() === 0 ? 6 : new Date(d).getDay() - 1];
        h += `<div class="future-date-label">${d} 周${wd}</div>`;
        for (const e of items) {
            const tags = (e.tags||[]).map(t=>{
        const c = TAG_COLORS[t] || DEFAULT_COLOR;
        return `<span class="tag" style="background:${c.bg};color:${c.text}">${t}</span>`;
    }).join("");
            h += `<div class="event-item event-plan" data-event-id="${e.id}">
                <div class="event-check check-plan" onclick="completePlan('${e.id}')"></div>
                <span class="event-time">${e.start_time||"全天"}</span>
                <span class="event-title">${e.title}</span><div class="event-tags">${tags}</div>
                <button class="event-del" title="删除" onclick="askDelete('${e.id}', this)">✕</button></div>`;
        }
    }
    c.innerHTML = h;
}

function renderMoods(moods) {
    const c = document.getElementById("moodList");
    if (!moods || !moods.length) { c.innerHTML = '<p class="empty-hint">今天还没记录心情</p>'; return; }
    c.innerHTML = moods.map(m => `<div class="mood-item" data-mood-id="${m.id}">
        <span class="mood-emoji">${MOOD_EMOJI[m.mood]||"😐"}</span>
        <span class="mood-content">${m.content||""}</span>
        <span class="mood-date">${m.date}</span>
        <button class="event-del" title="删除" onclick="askDelete('${m.id}', this, 'moods')">✕</button></div>`).join("");
}

async function completePlan(eventId) {
    const el = document.querySelector(`[data-event-id="${eventId}"]`);
    if (el) el.classList.add("completing");
    // 直连完成端点（UI 亲自指认，不过 agent loop——同手动删除）；毫秒级返回
    try {
        const res = await api(`/api/events/${eventId}/complete`, "POST");
        if (!res.ok) showReply("⚠️ " + (res.reply || "完成没记录上，稍后再试～"));
        setTimeout(()=>{ loadInit(); if(currentView==="day") loadDayView(); }, 300);
    } catch(e) { showReply("⚠️ " + e.message + "，完成没记录上，稍后再试～"); }
}

function showReply(reply) {
    // 独立错误/兜底卡（打勾失败、删除失败、加载失败等 UI 动作报错，不占用当前轮的卡）；
    // 不再 500s 自动隐藏——回复日志是追加式的，整段隐藏会误伤上面还想读的历史卡
    newReplyCard().textContent = reply;
}

// ═══ 手帐模式：可切换皮肤（journal.css 的 body.journal 作用域），默认关，localStorage 记住 ═══
function toggleJournal() {
    const on = document.body.classList.toggle("journal");
    try { localStorage.setItem("journalTheme", on ? "1" : ""); } catch(e) {}
}
try { if (localStorage.getItem("journalTheme")) document.body.classList.add("journal"); } catch(e) {}

// ═══ 横版宽屏：body.wide 作用域（style.css 末段），默认关，localStorage 记住；
//     仅 ≥1100px 生效，窄屏窗口开着开关也自动竖版（CSS 媒体查询兜底）═══
function toggleWide() {
    const on = document.body.classList.toggle("wide");
    try { localStorage.setItem("wideMode", on ? "1" : ""); } catch(e) {}
}
try { if (localStorage.getItem("wideMode")) document.body.classList.add("wide"); } catch(e) {}
function showDebug(text) { const e=document.getElementById("replyDebug"); e.classList.remove("hidden"); e.textContent=text; setTimeout(()=>e.classList.add("hidden"),100000); }

// ═══ 提醒信箱（后端统一调度的送达面）════════════
// 触发职责已移交后端（app/scheduler.py）：页面关着照常 fire 进信箱，
// 页面开着每 30s 领一次。保鲜期由后端判（提醒 30 分钟 / 报告 12~24 小时），
// 超期的后端直接静默清账——这里领到的都值得弹。领取即清账，天然 at-most-once。
const KIND_ICON = { remind: "⏰", plan_reminder: "⏰", agent: "🤖", overview: "🌤", weekly: "📊" };

async function pollReminders() {
    try {
        const res = await api("/api/reminders/due", "GET", null, 10000);
        if (res.ok && res.items && res.items.length) deliverReminders(res.items);
    } catch (_) { /* 领取失败不弹错：30s 后下一片自然重试，信箱不会丢 */ }
}

async function deliverReminders(items) {
    // 按 id 去重：后端是确认式投递（at-least-once），未 ack 的条目 90s 后会重领——
    // 重领是为了覆盖"HTTP 响应在网络层丢了"的情况，重弹则由这里挡住
    const fresh = items.filter(i => {
        try { return localStorage.getItem("shown:" + i.id) === null; } catch (_) { return true; }
    });
    for (const i of fresh) {
        try { localStorage.setItem("shown:" + i.id, i.created_at || ""); } catch (_) {}
    }
    // 无论是否重弹都确认（幂等），否则条目会一直被重领到过期
    try { await api("/api/reminders/ack", "POST", { ids: items.map(i => i.id) }, 10000); }
    catch (_) { /* ack 失败不打扰用户：90s 后重领，上面的去重会挡住重弹 */ }
    if (!fresh.length) return;
    // agent 任务结果是"报告"不是一闪而过的提示：横幅 8s 即逝，同时落一张回复卡片
    // 留在卡片流里（页面关着时错过的，开页补弹走同一条路，卡片照补）
    for (const i of fresh) {
        if (i.kind === "agent") {
            const card = newReplyCard();
            card.classList.add("reply-card-task");
            card.textContent = `⏰ 定时任务：${i.text}`;
        }
    }
    const text = fresh.map(i => `${KIND_ICON[i.kind] || "🔔"} ${i.text}`).join("；");
    // 系统通知（权限允许时；失败打日志不吞——横幅永远兜底）
    if (typeof Notification !== "undefined" && Notification.permission === "granted") {
        try { new Notification("记忆助手", { body: text, tag: "mynoplan-" + TODAY }); }
        catch (err) { console.warn("[remind] 系统通知发送失败:", err); }
    } else if (typeof Notification !== "undefined") {
        console.warn("[remind] 通知权限不是 granted，当前状态:", Notification.permission);
    }
    showRemindBanner(text);
    // 总览横幅弹过就说明新总览已落盘，首页栏目跟着换新（不等下一片轮询）
    if (items.some(i => i.kind === "overview")) refreshOverview();
}

function purgeLegacyRemindKeys() {
    // ① 旧前端提醒引擎的记账键（已废弃，一次性清空）
    // ② 送达去重键（shown:id，值存 created_at）：超过 24h 的清掉——信箱最长保鲜期就是 24h
    const cutoff = Date.now() - 24 * 3600 * 1000;
    try {
        for (let i = localStorage.length - 1; i >= 0; i--) {
            const k = localStorage.key(i);
            if (!k) continue;
            if (k.startsWith("reminded:")) { localStorage.removeItem(k); continue; }
            if (k.startsWith("shown:")) {
                const v = Date.parse(localStorage.getItem(k) || "");
                if (!isNaN(v) && v < cutoff) localStorage.removeItem(k);
            }
        }
    } catch (_) {}
}

// ═══ 提醒事项页（用户注册的定时任务）════════════
async function loadTasks() {
    // datetime-local 默认值：现在 + 1 小时（只在空着时填，不打断用户改过的值）
    const atEl = document.getElementById("taskAt");
    if (atEl && !atEl.value) {
        const d = new Date(Date.now() + 3600 * 1000);
        const pad = n => String(n).padStart(2, "0");
        // 手工拼本地时间（toISOString 是 UTC，会偏一个时区）
        atEl.value = `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
    }
    try {
        const res = await api("/api/tasks", "GET");
        if (res.ok) renderTasks(res.tasks || []);
    } catch (e) { console.error(e); }
}

function renderTasks(tasks) {
    const c = document.getElementById("taskList");
    if (!tasks.length) { c.innerHTML = '<p class="empty-hint">还没有定时任务，用上面的表单或直接对话添加</p>'; return; }
    c.innerHTML = tasks.map(t => {
        const icon = t.kind === "remind" ? "🔔" : "🤖";
        const body = (t.payload && (t.payload.text || t.payload.prompt)) || "";
        return `<div class="task-item" data-task-id="${t.id}">
            <span class="task-icon">${icon}</span>
            <div class="task-main">
                <div class="task-title">${body}</div>
                <div class="task-when">${taskScheduleLabel(t.schedule)} · 下次 ${fmtFire(t.next_fire)}</div>
            </div>
            <button class="event-del" title="取消" onclick="askDelete('${t.id}', this, 'tasks')">✕</button>
        </div>`;
    }).join("");
}

function taskScheduleLabel(s) {
    if (!s) return "";
    const hm = `${String(s.hour).padStart(2,"0")}:${String(s.minute||0).padStart(2,"0")}`;
    if (s.type === "once") return fmtFire(s.at);
    if (s.type === "daily") return `每天 ${hm}`;
    if (s.type === "weekly") return `每周${WEEKDAY_NAMES[s.weekday] || "?"} ${hm}`;
    if (s.type === "monthly") return `每月${s.day}日 ${hm}`;
    if (s.type === "yearly") return `每年${s.month}月${s.day}日 ${hm}`;
    return "";
}

function fmtFire(iso) {
    // "2026-09-01T07:00:00" → "9月1日 07:00"
    if (!iso) return "未知";
    const m = String(iso).match(/^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/);
    return m ? `${Number(m[2])}月${Number(m[3])}日 ${m[4]}:${m[5]}` : iso;
}

async function handleAddTask() {
    const textEl = document.getElementById("taskText"), atEl = document.getElementById("taskAt");
    const text = textEl.value.trim(), at = atEl.value;
    if (!text) { showRemindBanner("先写一下要提醒什么～"); return; }
    if (!at) { showRemindBanner("选一下提醒时间～"); return; }
    try {
        // 表单只做一次性 direct 提醒（零 LLM）；重复/agent 任务走对话注册，同一调度器
        const res = await api("/api/tasks", "POST", { mode: "direct", at, text });
        if (!res.ok) { showRemindBanner("添加没成功：" + (res.reply || "稍后再试～")); return; }
        textEl.value = "";
        loadTasks();
    } catch (e) { showRemindBanner("添加没成功：" + e.message + "，稍后再试～"); }
}

let _bannerTimer = null;
function showRemindBanner(text) {
    const b = document.getElementById("remindBanner");
    b.textContent = text;
    b.classList.add("show");
    clearTimeout(_bannerTimer);
    _bannerTimer = setTimeout(() => b.classList.remove("show"), 8000);
}

function askNotifyPermission() {
    if (typeof Notification !== "undefined" && Notification.permission === "default") {
        Notification.requestPermission();   // 浏览器要求在用户手势里调用（挂在 handleSend）
    }
}

// ═══════════════════════════════════════════════
//  月视图
// ═══════════════════════════════════════════════

async function loadMonthView() {
    try {
        const res = await api(`/api/calendar?view=month&d=${currentMonth}`, "GET");
        if (!res.ok) return;
        renderMonthView(res);
    } catch(e) { console.error(e); }
}

function navMonth(delta) {
    let [y,m] = currentMonth.split("-").map(Number);
    m += delta;
    if (m < 1) { m = 12; y--; }
    if (m > 12) { m = 1; y++; }
    // 限制在当年
    if (y < THIS_YEAR || y > THIS_YEAR) return;
    currentMonth = `${y}-${String(m).padStart(2,"0")}`;
    loadMonthView();
}

function renderMonthView(data) {
    document.getElementById("monthTitle").textContent = data.month;

    // 翻页按钮禁用边界
    const [y,m] = data.month.split("-").map(Number);
    document.querySelectorAll(".cal-nav")[0].disabled = (y === THIS_YEAR && m === 1);
    document.querySelectorAll(".cal-nav")[1].disabled = (y === THIS_YEAR && m === 12);

    const container = document.getElementById("monthCells");
    const weekdayStart = data.weekday_start; // 0=周一
    let html = "";

    // 前置空白
    for (let i = 0; i < weekdayStart; i++) {
        html += '<div class="month-cell empty"><span class="month-date"></span></div>';
    }

    for (const day of data.days) {
        const isToday = day.date === TODAY;
        const cls = ["month-cell", isToday ? "today" : ""].filter(Boolean).join(" ");

        // 色点：当日有事件的标签才显示，无事件不显示
        let dots = "";
        if (day.tags.length > 0) {
            const uniqueTags = [...new Set(day.tags)];
            dots = `<div class="month-dots">${uniqueTags.map(tag => {
                const color = TAG_COLORS[tag] || DEFAULT_COLOR;
                return `<span class="month-dot" style="background:${color.border}"></span>`;
            }).join("")}</div>`;
        }

        // 心情
        const moodStr = day.mood ? `<span class="month-mood">${MOOD_EMOJI[day.mood]||""}</span>` : "";

        const dateNum = day.date.split("-")[2];
        html += `<div class="${cls}" onclick="goToDay('${day.date}')">
            <span class="month-date">${dateNum}</span>
            <div class="month-dots">${dots}</div>${moodStr}</div>`;
    }

    container.innerHTML = html;
}

function goToDay(dateStr) {
    currentDayDate = dateStr;
    switchTab("day");
}

// ═══════════════════════════════════════════════
//  周视图
// ═══════════════════════════════════════════════

async function loadWeekView() {
    try {
        const res = await api(`/api/calendar?view=week&d=${currentWeekDate}`, "GET");
        if (!res.ok) return;
        renderWeekView(res);
    } catch(e) { console.error(e); }
}

function navWeek(delta) {
    const d = new Date(currentWeekDate);
    d.setDate(d.getDate() + delta * 7);
    const newStr = d.toISOString().split("T")[0];
    // 限制在当年
    if (d.getFullYear() !== THIS_YEAR) return;
    currentWeekDate = newStr;
    loadWeekView();
}

function renderWeekView(data) {
    const start = data.week_start;
    const end = data.days[data.days.length-1].date;
    document.getElementById("weekTitle").textContent = `${start} ~ ${end}`;

    // 翻页边界
    const startDate = new Date(start);
    const endDate = new Date(end);
    document.querySelectorAll(".cal-nav")[2].disabled = (startDate.getFullYear() < THIS_YEAR || (startDate.getFullYear() === THIS_YEAR && startDate.getMonth() === 0 && startDate.getDate() <= 7));
    document.querySelectorAll(".cal-nav")[3].disabled = (endDate.getFullYear() > THIS_YEAR || (endDate.getFullYear() === THIS_YEAR && endDate.getMonth() === 11));

    let html = "";
    for (const day of data.days) {
        const cls = ["week-col", day.is_today ? "today" : ""].filter(Boolean).join(" ");
        const dateNum = day.date.split("-")[2];

        // 色块缩略（全天事件不占时长，chip 形式排在列顶）
        let blocks = "";
        for (const e of day.events) {
            const tag = (e.tags && e.tags[0]) || "default";
            const color = TAG_COLORS[tag] || DEFAULT_COLOR;
            if (!e.start_time) {
                blocks += `<div class="week-block week-block-allday" style="background:${color.bg};border-color:${color.border};color:${color.text}">🎂 ${e.title}</div>`;
                continue;
            }
            const durationMin = calcDuration(e);
            const heightPx = Math.max(Math.round(durationMin / 60 * 20), 14);
            const bCls = e.type === "done" ? "week-block-done" : "week-block-plan";
            blocks += `<div class="week-block ${bCls}" style="background:${color.bg};border-color:${color.border};color:${color.text};height:${heightPx}px">${e.title}</div>`;
        }

        const mood = day.moods && day.moods.length ? `<div class="week-mood">${MOOD_EMOJI[day.moods[0].mood]||""}</div>` : "";

        html += `<div class="${cls}" onclick="goToDay('${day.date}')">
            <div class="week-col-header"><div class="week-col-date">${dateNum}</div><div class="week-col-weekday">${WEEKDAY_NAMES[day.weekday]}</div></div>
            ${blocks}${mood}</div>`;
    }
    document.getElementById("weekGrid").innerHTML = html;
}

// ═══════════════════════════════════════════════
//  日视图（时间轴 + 色块）
// ═══════════════════════════════════════════════

async function loadDayView() {
    try {
        const res = await api(`/api/calendar?view=day&d=${currentDayDate}`, "GET");
        if (!res.ok) return;
        renderDayView(res);
    } catch(e) { console.error(e); }
}

function navDay(delta) {
    const d = new Date(currentDayDate);
    d.setDate(d.getDate() + delta);
    if (d.getFullYear() !== THIS_YEAR) return;
    currentDayDate = d.toISOString().split("T")[0];
    loadDayView();
}

function renderDayView(data) {
    document.getElementById("dayTitle").textContent = data.date;

    // 翻页边界
    const d = new Date(data.date);
    document.querySelectorAll(".cal-nav")[4].disabled = (d.getFullYear() === THIS_YEAR && d.getMonth() === 0 && d.getDate() === 1);
    document.querySelectorAll(".cal-nav")[5].disabled = (d.getFullYear() === THIS_YEAR && d.getMonth() === 11 && d.getDate() === 31);

    const events = data.events || [];
    const moods = data.moods || [];

    // ─── 全天事件侧边栏（右侧独立列）：无 start_time 的不进时间轴 ───
    const allDay = events.filter(e => !e.start_time);
    const timed = events.filter(e => e.start_time);
    const bar = document.getElementById("allDayBar");
    if (allDay.length) {
        bar.classList.remove("hidden");
        bar.innerHTML = '<div class="all-day-title">🎂 全天</div>' + allDay.map(e => {
            const tag = (e.tags && e.tags[0]) || "default";
            const c = TAG_COLORS[tag] || DEFAULT_COLOR;
            const isDone = e.type === "done";
            const checkCls = isDone ? "day-block-check-done" : "day-block-check-plan";
            const checkOnclick = isDone ? "" : `onclick="completeDayPlan('${e.id}', this)"`;
            return `<span class="all-day-chip" data-event-id="${e.id}" style="background:${c.bg};border-color:${c.border};color:${c.text}">
                <span class="day-block-check ${checkCls}" ${checkOnclick}>${isDone ? "✓" : ""}</span>
                <span class="all-day-chip-title">${e.title}</span>
                <button class="event-del" title="删除" onclick="askDelete('${e.id}', this)">✕</button></span>`;
        }).join("");
    } else {
        bar.classList.add("hidden");
        bar.innerHTML = "";
    }

    const container = document.getElementById("dayTimeline");

    // ─── 左列：时间刻度（全天 24h）───
    const nowHour = new Date().getHours();
    let html = "";
    for (let h = 0; h < 24; h++) {
        // 今天：红线以上（过去）微暗5%，未来保持明亮
        const pastHour = data.is_today && h < nowHour;
        html += `<div class="day-hour-row${pastHour ? " day-hour-past" : ""}">
            <span class="day-hour-label">${String(h).padStart(2,"0")}:00</span></div>`;
    }
    container.innerHTML = html;
    // 打开/翻页即见 8:00–22:00，凌晨与深夜往上下滚动可见
    container.parentElement.scrollTop = 8 * 60;

    // ─── 右列：有时刻的事件色块（done + plan；开始时间相同的并列分列）───
    for (const b of layoutDayBlocks(timed)) {
        container.appendChild(createDayBlock(b.e, b.e.type, b.col, b.cols));
    }

    // ─── 当前时间红线（仅今天显示）───
    if (data.is_today) {
        const now = new Date();
        const nowMin = now.getHours() * 60 + now.getMinutes();
        const topPx = nowMin;  // 1min = 1px，全天自 0:00 起
        if (topPx >= 0 && topPx <= 24 * 60) {
            const line = document.createElement("div");
            line.className = "day-now-line";
            line.style.top = `${topPx}px`;
            container.appendChild(line);
        }
    }

    // ─── 心情固定底栏（时间轴滚动时钉在视图底部；常驻——空日子也留框，同主页行为）───
    const moodBox = document.getElementById("dayMood");
    if (moods.length > 0) {
        moodBox.innerHTML = '<div class="day-mood-title">💫 心情</div>' + moods.map(m =>
            `<span class="day-mood-item" data-mood-id="${m.id}">
                <span class="day-mood-emoji">${MOOD_EMOJI[m.mood] || "😐"}</span><span class="day-mood-content">${m.content || ""}</span>
                <button class="event-del" title="删除" onclick="askDelete('${m.id}', this, 'moods')">✕</button></span>`
        ).join(" ");
    } else {
        moodBox.innerHTML = '<div class="day-mood-title">💫 心情</div><p class="empty-hint">这天还没记录心情</p>';
    }
}

/** 将 "HH:MM" 转为自 0:00 起的分钟数 */
function timeToMin(t) {
    if (!t) return null;
    const [h, m] = t.split(":").map(Number);
    return h * 60 + m;
}

/** 并发布局：开始时间完全一致的事件并列分列横排（"两点开会 + 两点吃药"并排可见）。
 *  开始时间不同的事件保持重叠叠放——色块半透明，错时重叠可读，无需分列。
 *  无 start_time 的同桶处理（都聚在轴顶，同样参与分列）。
 */
function layoutDayBlocks(events) {
    const buckets = new Map();  // 开始时刻（分钟数）→ 同刻事件列表
    for (const e of events) {
        const key = e.start_time ? timeToMin(e.start_time) : -1;
        if (!buckets.has(key)) buckets.set(key, []);
        buckets.get(key).push(e);
    }
    const placed = [];
    for (const list of buckets.values()) {
        list.forEach((e, col) => placed.push({ e, col, cols: list.length }));
    }
    return placed;
}

function createDayBlock(e, type, col = 0, cols = 1) {
    const tag = (e.tags && e.tags[0]) || "default";
    const color = TAG_COLORS[tag] || DEFAULT_COLOR;
    const durationMin = calcDuration(e);
    const heightPx = Math.max(durationMin, 30);

    // 色块在时间轴上的位置（done 和 plan 都按 start_time 定位，全天轴 0:00 为起点）
    let topPx = 0;
    if (e.start_time) {
        topPx = timeToMin(e.start_time);
    }

    const block = document.createElement("div");
    const cls = type === "done" ? "done-block" : "plan-block";
    block.className = `day-event-block ${cls}`;
    block.dataset.eventId = e.id;  // 手动删除时向上找卡片用（doDelete 的 closest 锚点）
    block.style.top = `${topPx}px`;
    block.style.height = `${heightPx}px`;
    // 同刻并列：cols 列均分容器宽度；单列时退化为与旧版 left:4/right:4 完全一致
    block.style.left = `calc(4px + (100% - 8px) * ${col / cols})`;
    block.style.width = `calc((100% - 8px) / ${cols}${cols > 1 ? " - 6px" : ""})`;
    block.style.background = color.bg;
    block.style.borderColor = color.border;
    block.style.color = color.text;

    // 打勾（plan 可点击，done 已完成）
    const checkCls = type === "done" ? "day-block-check-done" : "day-block-check-plan";
    const checkIcon = type === "done" ? "✓" : "";
    const checkOnclick = type === "plan" ? `onclick="completeDayPlan('${e.id}', this)"` : "";

    block.innerHTML = `<span class="day-block-del" title="删除" onclick="askDelete('${e.id}', this)">✕</span>
        <span class="day-block-check ${checkCls}" ${checkOnclick}>${checkIcon}</span>
        <span class="day-block-title">${e.title}</span>
        ${e.note ? `<div class="day-block-note">${e.note}</div>` : ""}`;

    return block;
}

async function completeDayPlan(eventId, checkEl) {
    // 动画：○ 弹成 ✓ + 边框虚→实（内里不变——虚实的区分只在边框）
    const block = checkEl.parentElement;
    checkEl.className = "day-block-check day-block-check-done";
    checkEl.textContent = "✓";
    checkEl.classList.add("pop");
    block.classList.remove("plan-block");
    block.classList.add("done-block");

    // 直连完成端点（UI 亲自指认不过 agent loop——同首页打勾）；失败可见，不静默
    try {
        const res = await api(`/api/events/${eventId}/complete`, "POST");
        if (!res.ok) showReply("⚠️ " + (res.reply || "完成没记录上，稍后再试～"));
        setTimeout(()=>loadDayView(), 300);
    } catch(e) { showReply("⚠️ " + e.message + "，完成没记录上，稍后再试～"); }
}

// ═══════════════════════════════════════════════
//  手动删除（事件/心情通用，首页列表 + 日视图共用）
//  两段式确认：点 ✕ 变"确认"（红色），再点才真删；3 秒不动自动复原——防误触
// ═══════════════════════════════════════════════

let _delTimer = null;

function askDelete(id, btn, kind = "events") {
    if (btn.dataset.armed) { doDelete(kind, id, btn); return; }
    // 同屏只允许一个待确认的删除按钮
    document.querySelectorAll(".event-del.armed, .day-block-del.armed").forEach(b => disarmDel(b));
    btn.dataset.armed = "1";
    btn.classList.add("armed");
    btn.textContent = "确认";
    _delTimer = setTimeout(() => disarmDel(btn), 3000);
}

function disarmDel(btn) {
    if (!btn) return;
    delete btn.dataset.armed;
    btn.classList.remove("armed");
    btn.textContent = "✕";
}

async function doDelete(kind, id, btn) {
    clearTimeout(_delTimer);
    btn.textContent = "…";
    try {
        const res = await api(`/api/${kind}/${id}`, "DELETE");
        // 静默失败是最坏的失败（列表会停在旧数据上），失败一定可见
        if (!res.ok) { showReply("⚠️ " + (res.reply || "删除没有成功，稍后再试～")); disarmDel(btn); return; }
        // 成功：淡出后刷新（首页 + 日视图 + 提醒页）
        const item = btn.closest("[data-event-id], [data-mood-id], [data-task-id]");
        if (item) { item.style.transition = "opacity .25s"; item.style.opacity = "0"; }
        setTimeout(() => { loadInit(); if (currentView === "day") loadDayView(); if (currentView === "tasks") loadTasks(); }, 250);
    } catch (e) {
        showReply("⚠️ " + e.message + "，删除没有成功，稍后再试～");
        disarmDel(btn);
    }
}

// ─── 工具函数 ────────────────────────────────

function calcDuration(e) {
    if (e.start_time && e.end_time) {
        const [sh,sm] = e.start_time.split(":").map(Number);
        const [eh,em] = e.end_time.split(":").map(Number);
        return (eh*60+em) - (sh*60+sm);
    }
    if (e.start_time) return 60; // 有开始无结束，默认1h
    return 30; // 无时间，默认30min
}

async function api(path, method="GET", body=null, timeoutMs=60000) {
    const opts = { method, headers: { "Content-Type": "application/json" }, signal: AbortSignal.timeout(timeoutMs) };
    if (body) opts.body = JSON.stringify(body);
    try {
        const res = await fetch(path, opts);
        return await res.json();
    } catch (e) {
        throw new Error(e.name === "TimeoutError" ? "请求超时了" : "网络连接失败");
    }
}

function setSendLoading(loading) {
    // 只换图标不锁按钮：随时可点，点了就入队（串行消化在前端队列 + 服务端单飞轮锁）
    const text=document.getElementById("sendText"),spinner=document.getElementById("sendLoading");
    text.classList.toggle("hidden",loading); spinner.classList.toggle("hidden",!loading);
}
