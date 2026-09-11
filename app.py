import re
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
app = Flask(__name__)
data = pd.read_csv(Path(__file__).parent / "complaints.csv").dropna(subset=["complaint_text","category","subcategory"])
v = TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True)
x = v.fit_transform(data.complaint_text.astype(str)); model = MultinomialNB(alpha=.35).fit(x,data.category)
subs = {c:MultinomialNB(alpha=.35).fit(v.transform(g.complaint_text.astype(str)),g.subcategory) for c,g in data.groupby("category") if g.subcategory.nunique()>1}
@app.get("/")
def home(): return render_template_string(PAGE,categories=sorted(data.category.unique()),total=len(data))
@app.post("/api/classify")
def classify():
 t=re.sub(r"\s+"," ",(request.get_json() or {}).get("complaint","")).strip()
 if len(t)<8:return jsonify(error="Please enter at least 8 characters."),400
 z=v.transform([t]); p=model.predict_proba(z)[0]; i=p.argmax(); c=str(model.classes_[i]); s=c if c not in subs else str(subs[c].predict(z)[0])
 return jsonify(category=c,subcategory=s,confidence=round(float(p[i])*100,1))
PAGE='''<title>Complaint Compass</title><style>body{margin:0;background:#f2f8f5;color:#14231f;font:16px Arial}.w{max-width:900px;margin:auto;padding:42px 22px}h1{font:700 58px Georgia;margin:8px 0}.tag{color:#087a72;font-weight:bold}.box{background:white;border:1px solid #d9e4df;border-radius:22px;padding:34px;box-shadow:0 20px 50px #1232}textarea{width:100%;height:140px;margin:18px 0;padding:14px;box-sizing:border-box;border:1px solid #cfded7;border-radius:12px;font:16px Arial}button{float:right;background:#087a72;color:#fff;border:0;border-radius:10px;padding:13px 18px;font-weight:bold}.result{display:none;clear:both;margin-top:70px;background:#e9f7f0;padding:20px;border-radius:15px}.show{display:block}.muted{color:#65736d}</style><main class=w><div class=tag>✦ PASSENGER SUPPORT INTELLIGENCE</div><h1>Complaint Compass</h1><p class=muted>Classify passenger issues with an NLP model trained on {{total}} labelled examples.</p><section class=box><h2>Classify a complaint</h2><textarea id=q placeholder='A passenger fainted and needs immediate help.'></textarea><button id=go>Analyse complaint →</button><p class=muted>Supported: {{categories|join(', ')}}</p><p id=err></p><div id=out class=result><b id=cat></b><p>Subcategory: <span id=sub></span></p><p>Confidence: <span id=score></span></p></div></section></main><script>go.onclick=async()=>{let r=await fetch('/api/classify',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({complaint:q.value})}),d=await r.json();if(!r.ok){err.textContent=d.error;return}cat.textContent=d.category;sub.textContent=d.subcategory;score.textContent=d.confidence+'%';out.classList.add('show')}</script>'''
if __name__=="__main__":app.run(debug=True)
