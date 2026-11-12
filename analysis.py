"""Reproducible random forest regression on the attributed local UCI snapshot."""
from pathlib import Path
import csv,json,platform,hashlib
import numpy as np
import sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,KFold,GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
P=Path(__file__).resolve().parent
with (P/'data.csv').open() as f: rows=list(csv.DictReader(f))
features=['Age','Dist.MRT','N.Cstores']
X=np.array([[float(r[c]) for c in features] for r in rows]);y=np.array([float(r['Y']) for r in rows])
assert len(rows)==414 and np.isfinite(X).all() and np.isfinite(y).all()
assert len(set(r['No'] for r in rows))==414
tr,te=train_test_split(np.arange(len(y)),test_size=.25,random_state=42)
assert not set(tr)&set(te)
# Parameter selection is confined to training rows. Explicit max_features randomizes features.
cv=KFold(5,shuffle=True,random_state=42)
search=GridSearchCV(RandomForestRegressor(n_estimators=300,random_state=42,n_jobs=1),{'max_depth':[None,5,10],'min_samples_leaf':[1,3,5],'max_features':[.67,1.]},scoring='neg_root_mean_squared_error',cv=cv,n_jobs=2,return_train_score=True)
search.fit(X[tr],y[tr])
models={'Training mean':DummyRegressor(),'Linear regression':LinearRegression(),'Random forest':search.best_estimator_}
metrics={};pred={}
for name,m in models.items():
 if name!='Random forest':m.fit(X[tr],y[tr])
 p=m.predict(X[te]);pred[name]=p
 metrics[name]={'RMSE':float(np.sqrt(mean_squared_error(y[te],p))),'MAE':float(mean_absolute_error(y[te],p)),'R2':float(r2_score(y[te],p))}
 assert np.isclose(metrics[name]['RMSE'],np.sqrt(np.mean((y[te]-p)**2)))
result={'train_n':len(tr),'test_n':len(te),'seed':42,'features':features,'best_parameters':search.best_params_,'training_cv_RMSE':-search.best_score_,'test_metrics':metrics,'python':platform.python_version(),'sklearn':sklearn.__version__,'data_sha256':hashlib.sha256((P/'data.csv').read_bytes()).hexdigest()}
(P/'metrics.json').write_text(json.dumps(result,indent=2))
with (P/'predictions.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['record_id','actual',*pred]);w.writerows([[rows[i]['No'],y[i],*[pred[n][j] for n in pred]] for j,i in enumerate(te)])
with (P/'split.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['record_id','split']);w.writerows([[rows[i]['No'],'train' if i in set(tr) else 'test'] for i in range(len(rows))])
with (P/'cv_results.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['parameters','mean_cv_RMSE','sd_cv_RMSE']);w.writerows([[str(p),-m,s] for p,m,s in zip(search.cv_results_['params'],search.cv_results_['mean_test_score'],search.cv_results_['std_test_score'])])
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(1,2,figsize=(13,5.6))
colors=['#8499aa','#416784','#00a6a6'];names=list(metrics);vals=[metrics[n]['RMSE'] for n in names]
ax[0].barh(names,vals,color=colors);ax[0].invert_yaxis();ax[0].set_xlim(0,max(vals)*1.2);ax[0].set_xlabel('Test RMSE (10,000 NT$ per ping)')
for j,v in enumerate(vals):ax[0].text(v+.15,j,f'{v:.2f}',va='center')
p=pred['Random forest'];ax[1].scatter(y[te],p,c='#008f95',alpha=.75,edgecolors='white',linewidths=.4)
lo=min(y[te].min(),p.min())-2;hi=max(y[te].max(),p.max())+2
ax[1].plot([lo,hi],[lo,hi],'--',c='#657786',label='Perfect prediction');ax[1].set(xlim=(lo,hi),ylim=(lo,hi),xlabel='Actual price per unit area',ylabel='Random forest prediction');ax[1].legend()
fig.suptitle('Real estate prices in Python | Random forest regression',fontweight='bold',fontsize=18)
fig.text(.5,.025,f'UCI Real Estate Valuation, Yeh (2018), CC BY 4.0 | Seed 42 | Train {len(tr)}, test {len(te)} | Historical random split, not a future-market test',ha='center',fontsize=10)
fig.tight_layout(rect=[0,.07,1,.94]);fig.savefig(P/'evaluation.png',dpi=170);plt.close(fig)
print(json.dumps(result,indent=2))
