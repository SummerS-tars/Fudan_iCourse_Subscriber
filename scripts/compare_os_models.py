import sys,os,json,time,hashlib,sqlite3,concurrent.futures
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from openai import OpenAI
from src.ai import bucketer
from src.ai.summarizer import Summarizer
from src.runtime import config
out=Path('comparison');out.mkdir(exist_ok=True)
c=sqlite3.connect('data/icourse.db');c.row_factory=sqlite3.Row
r=c.execute("SELECT l.*,c.title AS course_title FROM lectures l JOIN courses c ON l.course_id=c.course_id WHERE l.course_id='40054' AND l.sub_title LIKE '%2026-10-15%' AND l.summary IS NOT NULL LIMIT 1").fetchone()
if not r: raise RuntimeError('OS H target not found')
r=dict(r)
pages=[dict(x) for x in c.execute("SELECT page_num,created_sec,text FROM ppt_pages WHERE sub_id=? AND ocr_status='done' AND text IS NOT NULL AND text!='' ORDER BY created_sec",(r['sub_id'],))]
prompt,mode=bucketer.assemble(r['transcript'],None,pages)
(out/'source.txt').write_text(prompt)
(out/'pro-existing.md').write_text(r['summary'])
meta={'sub_id':r['sub_id'],'title':r['sub_title'],'course':r['course_title'],'mode':mode,'input_chars':len(prompt),'input_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'original_model':r.get('summary_model'),'note':'Original Pro may have used transient timed segments; all new calls use identical reconstructed flat prompt.'}
def run(name,key,url,model):
 start=time.monotonic();result={'provider':name,'model':model}
 try:
  client=OpenAI(api_key=os.environ[key],base_url=url,max_retries=0)
  # Preserve production prompt/parameters. Record raw token usage separately.
  class Capture:
   def __init__(self,inner):self.inner=inner;self.chat=self;self.completions=self;self.usage=None;self.finish=None
   def create(self,**kw):
    kw['timeout']=600
    resp=self.inner.chat.completions.create(**kw);self.usage=resp.usage.model_dump() if resp.usage else None;self.finish=resp.choices[0].finish_reason if resp.choices else None;return resp
  capture=Capture(client)
  text=Summarizer._call_llm(None,capture,model,r['course_title'],prompt)
  if not text or not text.strip():raise ValueError('Empty answer')
  (out/(name+'.md')).write_text(text)
  result.update(ok=True,chars=len(text),usage=capture.usage,finish_reason=capture.finish)
 except Exception as e:result.update(ok=False,error=str(e),error_type=type(e).__name__)
 result['seconds']=round(time.monotonic()-start,2)
 print(json.dumps(result,ensure_ascii=False),flush=True);return result
specs=[('pro-controlled','DASHSCOPE_API_KEY','https://api-inference.modelscope.cn/v1/','deepseek-ai/DeepSeek-V4-Pro-0813'),('deepseek-flash','DEEPSEEK_API_KEY','https://api.deepseek.com/v1','deepseek-flash'),('gemini-flash','GEMINI_API_KEY','https://generativelanguage.googleapis.com/v1beta/openai/','gemini-3.8-flash')]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:meta['results']=list(pool.map(lambda x:run(*x),specs))
(out/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
