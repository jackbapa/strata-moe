import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tools'))
from bench_a3b_latency import Client,stream_request
from bench_a3b_supplemental import NATURAL_PROMPTS
out=Path(__file__).parent
client=Client('http://127.0.0.1:8096',600)
model=client.json('/v1/models')['data'][0]['id']
cases=list(NATURAL_PROMPTS.items())+[
 ('natural-english','Explain how operating systems manage virtual memory. Cover page tables, page faults, swapping, and the difference between virtual address space and physical RAM. Use a worked example, and explain why a large committed memory number does not necessarily mean equally large physical RAM use. Aim for about 500 words.'),
 ('natural-sql','Design a PostgreSQL schema for a small online store with customers, orders, order_items and products. Include primary keys, foreign keys and appropriate constraints. Give SQL queries for top 5 customers by total completed sales and monthly revenue, and explain indexes needed. Output SQL first.'),
 ('natural-reasoning','一个水箱有两根进水管和一根排水管。A管单开6小时注满，B管单开9小时注满，排水管单开18小时排空。从空箱开始，先只开A管1小时，再同时开A、B和排水管。求此后还需多少小时注满。用分数一步步推导，再设计一个不同数字的例子验证方法。')
]
results=[]
(out/'training-cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2),encoding='utf8')
for i,(name,text) in enumerate(cases):
 body={'model':model,'messages':[{'role':'user','content':text}],'max_tokens':512,'temperature':0,'seed':12345,'stream':True,'stream_options':{'include_usage':True},'chat_template_kwargs':{'enable_thinking':False}}
 r=stream_request(client,body,out,name)
 r.update(case_id=name,training=True)
 results.append(r)
 (out/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 print(json.dumps(r,ensure_ascii=False),flush=True)
print('DONE; use only final routes-cumulative.tsv once, not intermediate cumulative snapshots',flush=True)
