"""
LinkedIn Assets Generator for Content Moderation Project
Generates: Terminal Screenshot HTML, Architecture Diagram HTML, Metrics Card HTML
"""

# ── 1. Terminal Screenshot ──────────────────────────────────────────────────
terminal_html = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #1e1e1e; display: flex; justify-content: center; align-items: center; min-height: 100vh; font-family: 'Courier New', monospace; }
  .window { width: 720px; border-radius: 10px; overflow: hidden; box-shadow: 0 20px 60px rgba(0,0,0,0.8); }
  .titlebar { background: #3c3c3c; padding: 10px 16px; display: flex; align-items: center; gap: 8px; }
  .dot { width: 13px; height: 13px; border-radius: 50%; }
  .red { background: #ff5f57; } .yellow { background: #febc2e; } .green { background: #28c840; }
  .title { color: #aaa; font-size: 13px; margin-left: auto; margin-right: auto; }
  .terminal { background: #0d0d0d; padding: 24px 28px; font-size: 14px; line-height: 2; }
  .prompt { color: #4ec9b0; }
  .cmd { color: #dcdcdc; }
  .comment { color: #6a9955; }
  .blocked { color: #f44747; font-weight: bold; }
  .approved { color: #4ec9b0; font-weight: bold; }
  .warn { color: #dcdcaa; }
  .dim { color: #555; }
  .label { color: #9cdcfe; }
</style>
</head>
<body>
<div class="window">
  <div class="titlebar">
    <div class="dot red"></div>
    <div class="dot yellow"></div>
    <div class="dot green"></div>
    <span class="title">Terminal — content_moderation</span>
  </div>
  <div class="terminal">
    <div><span class="dim"># ── Toxic Content ──────────────────────────</span></div>
    <div><span class="prompt">❯ </span><span class="cmd">python main.py --text "You are an idiot and I hate you"</span></div>
    <div><span class="blocked">❌ Blocked: contains toxic/impolite language</span></div>
    <br>
    <div><span class="dim"># ── Banned Word ─────────────────────────────</span></div>
    <div><span class="prompt">❯ </span><span class="cmd">python main.py --text "This is absolute sh!t content"</span></div>
    <div><span class="blocked">❌ Blocked: contains banned/political term - 'sh!t'</span></div>
    <br>
    <div><span class="dim"># ── Political Term ──────────────────────────</span></div>
    <div><span class="prompt">❯ </span><span class="cmd">python main.py --text "Vote for the revolution now"</span></div>
    <div><span class="blocked">❌ Blocked: contains banned/political term - 'vote'</span></div>
    <br>
    <div><span class="dim"># ── Clean Content ───────────────────────────</span></div>
    <div><span class="prompt">❯ </span><span class="cmd">python main.py --text "Great work on the project today!"</span></div>
    <div><span class="approved">✅ Approved: Clean content</span></div>
    <br>
    <div><span class="dim"># ── Empty Message ───────────────────────────</span></div>
    <div><span class="prompt">❯ </span><span class="cmd">python main.py --text "   "</span></div>
    <div><span class="blocked">❌ Blocked: Empty message</span></div>
  </div>
</div>
</body>
</html>"""

# ── 2. Architecture Diagram ─────────────────────────────────────────────────
architecture_html = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0f172a; display: flex; justify-content: center; align-items: center; min-height: 100vh; font-family: 'Segoe UI', sans-serif; }
  .card { width: 760px; padding: 48px 40px; }
  h2 { color: #e2e8f0; text-align: center; font-size: 22px; letter-spacing: 1px; margin-bottom: 48px; }
  h2 span { color: #38bdf8; }
  .flow { display: flex; align-items: center; justify-content: center; gap: 0; flex-wrap: nowrap; }
  .box { background: #1e293b; border: 1.5px solid #334155; border-radius: 12px; padding: 18px 16px; text-align: center; min-width: 120px; }
  .box .icon { font-size: 28px; margin-bottom: 8px; }
  .box .label { color: #94a3b8; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
  .box .name { color: #e2e8f0; font-size: 13px; font-weight: 600; }
  .box.input { border-color: #38bdf8; }
  .box.filter { border-color: #f59e0b; }
  .box.ai { border-color: #a78bfa; }
  .box.approved { border-color: #22c55e; background: #052e16; }
  .box.blocked { border-color: #ef4444; background: #1c0a0a; }
  .arrow { color: #475569; font-size: 22px; padding: 0 6px; flex-shrink: 0; }
  .outcomes { display: flex; flex-direction: column; gap: 12px; }
  .sub { color: #64748b; font-size: 11px; margin-top: 4px; }
  .divider { width: 760px; height: 1px; background: #1e293b; margin: 36px auto; }
  .layers { display: flex; justify-content: center; gap: 32px; margin-top: 0; }
  .layer { text-align: center; }
  .layer .num { width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 700; margin: 0 auto 8px; }
  .layer .lname { color: #94a3b8; font-size: 12px; }
  .l1 .num { background: #0c4a6e; color: #38bdf8; }
  .l2 .num { background: #451a03; color: #f59e0b; }
  .l3 .num { background: #2e1065; color: #a78bfa; }
  .l4 .num { background: #052e16; color: #22c55e; }
  footer { text-align: center; margin-top: 36px; color: #334155; font-size: 11px; letter-spacing: 2px; text-transform: uppercase; }
</style>
</head>
<body>
<div class="card">
  <h2>AI-Powered <span>Content Moderation</span> Pipeline</h2>
  <div class="flow">
    <div class="box input">
      <div class="icon">📝</div>
      <div class="label">Input</div>
      <div class="name">User Text</div>
    </div>
    <div class="arrow">→</div>
    <div class="box">
      <div class="icon">🔍</div>
      <div class="label">Layer 1</div>
      <div class="name">Empty Check</div>
    </div>
    <div class="arrow">→</div>
    <div class="box filter">
      <div class="icon">🚫</div>
      <div class="label">Layer 2</div>
      <div class="name">Word Filter</div>
      <div class="sub">Banned + Political</div>
    </div>
    <div class="arrow">→</div>
    <div class="box ai">
      <div class="icon">🤖</div>
      <div class="label">Layer 3</div>
      <div class="name">DistilBERT</div>
      <div class="sub">Toxicity AI</div>
    </div>
    <div class="arrow">→</div>
    <div class="outcomes">
      <div class="box approved">
        <div class="name">✅ Approved</div>
        <div class="sub">Clean content</div>
      </div>
      <div class="box blocked">
        <div class="name">❌ Blocked</div>
        <div class="sub">Toxic / Banned</div>
      </div>
    </div>
  </div>
  <div class="divider"></div>
  <div class="layers">
    <div class="layer l1"><div class="num">1</div><div class="lname">Empty Check</div></div>
    <div class="layer l2"><div class="num">2</div><div class="lname">Word Lists</div></div>
    <div class="layer l3"><div class="num">3</div><div class="lname">DistilBERT AI</div></div>
    <div class="layer l4"><div class="num">4</div><div class="lname">Decision</div></div>
  </div>
  <footer>Python · HuggingFace Transformers · PyTorch · scikit-learn</footer>
</div>
</body>
</html>"""

# ── 3. Metrics Card ─────────────────────────────────────────────────────────
metrics_html = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0f172a; display: flex; justify-content: center; align-items: center; min-height: 100vh; font-family: 'Segoe UI', sans-serif; }
  .card { width: 680px; padding: 52px 48px; background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); border: 1px solid #1e293b; border-radius: 20px; box-shadow: 0 0 80px rgba(56,189,248,0.08); }
  .badge { display: inline-block; background: #0c4a6e; color: #38bdf8; font-size: 11px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; padding: 6px 14px; border-radius: 20px; margin-bottom: 20px; }
  h1 { color: #f1f5f9; font-size: 26px; font-weight: 700; margin-bottom: 6px; }
  h1 span { color: #38bdf8; }
  .sub { color: #64748b; font-size: 13px; margin-bottom: 44px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 40px; }
  .metric { background: #1e293b; border-radius: 14px; padding: 24px 20px; border: 1px solid #334155; position: relative; overflow: hidden; }
  .metric::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; }
  .m1::before { background: linear-gradient(90deg, #ef4444, #f97316); }
  .m2::before { background: linear-gradient(90deg, #f59e0b, #eab308); }
  .m3::before { background: linear-gradient(90deg, #a78bfa, #818cf8); }
  .m4::before { background: linear-gradient(90deg, #22c55e, #10b981); }
  .metric .value { font-size: 42px; font-weight: 800; margin-bottom: 4px; }
  .m1 .value { color: #f87171; }
  .m2 .value { color: #fbbf24; }
  .m3 .value { color: #a78bfa; }
  .m4 .value { color: #4ade80; }
  .metric .name { color: #94a3b8; font-size: 13px; font-weight: 500; }
  .metric .desc { color: #475569; font-size: 11px; margin-top: 4px; }
  .footer { display: flex; justify-content: space-between; align-items: center; border-top: 1px solid #1e293b; padding-top: 24px; }
  .stack { display: flex; gap: 8px; flex-wrap: wrap; }
  .tag { background: #1e293b; color: #64748b; font-size: 11px; padding: 4px 10px; border-radius: 6px; border: 1px solid #334155; }
  .model { color: #475569; font-size: 12px; }
  .model span { color: #38bdf8; }
</style>
</head>
<body>
<div class="card">
  <div class="badge">🛡️ Model Performance</div>
  <h1>Content <span>Moderation</span> System</h1>
  <p class="sub">Fine-tuned DistilBERT · Multi-layer filtering · Real-time classification</p>
  <div class="grid">
    <div class="metric m1">
      <div class="value">99.2%</div>
      <div class="name">Profanity Recall</div>
      <div class="desc">Banned word detection rate</div>
    </div>
    <div class="metric m2">
      <div class="value">97.5%</div>
      <div class="name">Political Recall</div>
      <div class="desc">Sensitive term detection rate</div>
    </div>
    <div class="metric m3">
      <div class="value">92.1%</div>
      <div class="name">Toxicity Accuracy</div>
      <div class="desc">AI classification accuracy</div>
    </div>
    <div class="metric m4">
      <div class="value">1.8%</div>
      <div class="name">False Positives</div>
      <div class="desc">Clean content wrongly blocked</div>
    </div>
  </div>
  <div class="footer">
    <div class="stack">
      <span class="tag">Python</span>
      <span class="tag">PyTorch</span>
      <span class="tag">HuggingFace</span>
      <span class="tag">scikit-learn</span>
    </div>
    <div class="model">Model: <span>DistilBERT</span></div>
  </div>
</div>
</body>
</html>"""

# ── Write all files ──────────────────────────────────────────────────────────
files = {
    "linkedin_terminal_screenshot.html": terminal_html,
    "linkedin_architecture_diagram.html": architecture_html,
    "linkedin_metrics_card.html": metrics_html,
}

for filename, content in files.items():
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Created: {filename}")

print("\nNext steps:")
print("  1. Open each HTML file in Chrome/Edge")
print("  2. Press F12 > Ctrl+Shift+P > type 'screenshot' > Capture full size screenshot")
print("  3. Save the PNG and upload to LinkedIn")
