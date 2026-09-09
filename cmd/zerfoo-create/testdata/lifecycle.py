import subprocess,json,tempfile,time,pathlib,sys
state=tempfile.mkdtemp(prefix='zerfoo-create-check-')
base=[str(pathlib.Path(sys.argv[1]).resolve()),'--state',state,'--data-root',str(pathlib.Path('tabular/testdata/model_creation').resolve())]
p=subprocess.Popen(base+['mcp'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
seq=0
def rpc(method,params):
 global seq
 seq+=1;p.stdin.write(json.dumps({'jsonrpc':'2.0','id':seq,'method':method,'params':params})+'\n');p.stdin.flush()
 response=json.loads(p.stdout.readline());assert 'error' not in response,response
 return response['result']
def tool(name,args):
 result=rpc('tools/call',{'name':name,'arguments':args});assert not result.get('isError'),result
 return json.loads(result['content'][0]['text'])
rpc('initialize',{'protocolVersion':'2025-11-25','capabilities':{},'clientInfo':{'name':'lifecycle-check','version':'1'}})
p.stdin.write(json.dumps({'jsonrpc':'2.0','method':'notifications/initialized'})+'\n');p.stdin.flush()
discovered=rpc('tools/list',{})['tools']
assert len(discovered)==12
plan_schema=next(t['inputSchema'] for t in discovered if t['name']=='plan_create')
assert 'definition' in plan_schema['properties']
assert 'hidden_dims' not in plan_schema['required']
project=tool('project_create',{'objective':'Classify iris from measurements'})
data=tool('dataset_inspect',{'project':project['id'],'path':'iris.csv','target':'species','seed':42})
definition={
 'version':1,'name':'iris_residual',
 'inputs':[{'name':'features','dtype':'float32','shape':[-1,4]}],
 'parameters':[
  {'name':'w1','dtype':'float32','shape':[4,16],'initializer':'he_normal'},
  {'name':'b1','dtype':'float32','shape':[16],'initializer':'zeros'},
  {'name':'w2','dtype':'float32','shape':[16,3],'initializer':'he_normal'},
  {'name':'b2','dtype':'float32','shape':[3],'initializer':'zeros'}],
 'nodes':[
  {'name':'left','operator':'Dense','version':1,'inputs':{'x':{'node':'features','port':'value'}},'parameters':{'weights':'w1','bias':'b1'}},
  {'name':'right','operator':'Dense','version':1,'inputs':{'x':{'node':'features','port':'value'}},'parameters':{'weights':'w1','bias':'b1'}},
  {'name':'sum','operator':'Add','version':1,'inputs':{'a':{'node':'left','port':'y'},'b':{'node':'right','port':'y'}}},
  {'name':'active','operator':'ReLU','version':1,'inputs':{'x':{'node':'sum','port':'y'}}},
  {'name':'head','operator':'Dense','version':1,'inputs':{'x':{'node':'active','port':'y'}},'parameters':{'weights':'w2','bias':'b2'}}],
 'outputs':[{'name':'logits','source':{'node':'head','port':'y'}}]}
plan=tool('plan_create',{'project':project['id'],'dataset':data['id'],'rationale':'Explicit residual DSL classifier','definition':definition,'epochs':20,'batch_size':15,'learning_rate':0.01,'seed':42})
assert plan['version']==2 and len(plan['definition_sha256'])==64,plan
run=tool('run_start',{'plan':plan['id'],'idempotency_key':'disconnect-test'})
assert tool('run_start',{'plan':plan['id'],'idempotency_key':'disconnect-test'})['id']==run['id']
p.stdin.close();assert p.wait(timeout=10)==0
for _ in range(100):
 result=subprocess.run(base+['run_status',json.dumps({'id':run['id']})],capture_output=True,text=True,check=True)
 status=json.loads(result.stdout)
 if status['status'] not in ['queued','running']:break
 time.sleep(.1)
assert status['status']=='succeeded',status
result=subprocess.run(base+['model_predict',json.dumps({'run':run['id'],'rows':[[5.1,3.5,1.4,.2]]})],capture_output=True,text=True,check=True)
prediction=json.loads(result.stdout)
assert prediction['predictions'][0]['label']=='Iris-setosa',prediction
assert prediction['artifact_sha256']==status['artifact_sha256']
print(json.dumps({'checks':{'mcp_tool_discovery':12,'explicit_residual_dsl':True,'real_training':True,'idempotent_start':True,'training_survives_mcp_exit':True,'fresh_cli_prediction':True},'steps':status['progress']['steps'],'validation':status['validation'],'artifact_sha256':status['artifact_sha256']}))
