"""Development-first V2 scorecard; no data loading or model training on import.

Patient-cluster percentile bootstrap, with independent synthetic resampling.
Paired reconstructions share source groups in both bootstrap and judge splits.
Five judge splits are a sensitivity range, NOT a confidence interval.
"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits
from .gen_eval import clinical_features, validate_ecgs, nearest_distance
from .common import CLASSES
from synthetic_generation.realism.features import NAMES

GROUPS={'rhythm':list(range(5)), 'morphology':list(range(5,13)),
        'noise_wander':list(range(13,21)), 'global':list(range(21,25))}
METRICS=('precision','recall','density','coverage')


def interval(value, draws):
    draws=np.asarray(draws,dtype=float)
    valid=draws[np.isfinite(draws)]
    return dict(value=float(value),ci95=np.quantile(valid,[.025,.975]).tolist() if len(valid) else None,
                valid_replicates=len(valid),replicates=len(draws))


def clusters(groups):
    groups=np.asarray(groups)
    if groups.ndim!=1 or not len(groups):raise ValueError('Nonempty one-dimensional groups required')
    return [np.flatnonzero(groups==g) for g in np.unique(groups)]


def draw_clusters(parts,rng):
    return np.concatenate([parts[i] for i in rng.integers(len(parts),size=len(parts))])


def resamples(gr,gs,replicates=500,seed=9200,paired=False):
    ar,bs=clusters(gr),clusters(gs)
    rng=np.random.default_rng(seed)
    if paired:
        if not np.array_equal(gr,gs):raise ValueError('Paired bootstrap requires aligned source groups')
    for _ in range(replicates):
        ri=draw_clusters(ar,rng)
        yield ri,ri.copy() if paired else draw_clusters(bs,rng)


def _prdc_distances(rr,ss,rs,k=5):
    if k<1 or min(len(rr),len(ss))<=k:raise ValueError('PRDC needs more than k rows per pool')
    rr,ss=rr.copy(),ss.copy()
    np.fill_diagonal(rr,np.inf);np.fill_diagonal(ss,np.inf)
    r=np.partition(rr,k-1,axis=1)[:,k-1]
    s=np.partition(ss,k-1,axis=1)[:,k-1]
    return np.asarray([(rs<=r[:,None]).any(0).mean(),(rs<=s[None,:]).any(1).mean(),
                       (rs<=r[:,None]).sum(0).mean()/k,(rs.min(1)<=r).mean()])


def prdc_uncertainty(real,synthetic,gr,gs,replicates=500,seed=9200,paired=False,k=5):
    if len(gr)!=len(real) or len(gs)!=len(synthetic):raise ValueError('Embedding/group alignment')
    if replicates<1:raise ValueError('At least one bootstrap replicate required')
    if real.ndim!=2 or synthetic.ndim!=2 or real.shape[1]!=synthetic.shape[1]:raise ValueError('Embedding dimensions')
    if not np.isfinite(real).all() or not np.isfinite(synthetic).all():raise ValueError('Nonfinite embedding')
    rr,ss,rs=cdist(real,real),cdist(synthetic,synthetic),cdist(real,synthetic)
    point=_prdc_distances(rr,ss,rs,k)
    draws=[]
    for ri,si in resamples(gr,gs,replicates,seed,paired):
        draws.append(_prdc_distances(rr[np.ix_(ri,ri)],ss[np.ix_(si,si)],rs[np.ix_(ri,si)],k))
    return {key:interval(point[i],np.asarray(draws)[:,i]) for i,key in enumerate(METRICS)}


def _feature_stats(a,b):
    aa,bb=a[np.isfinite(a)],b[np.isfinite(b)]
    qa=np.subtract(*np.quantile(aa,[.75,.25])) if len(aa) else np.nan
    qb=np.subtract(*np.quantile(bb,[.75,.25])) if len(bb) else np.nan
    w=wasserstein_distance(aa,bb) if len(aa) and len(bb) else np.nan
    return [w,w/qa if qa>1e-9 else np.nan,np.median(aa) if len(aa) else np.nan,
            np.median(bb) if len(bb) else np.nan,aa.std() if len(aa) else np.nan,
            bb.std() if len(bb) else np.nan,qa,qb,1-len(aa)/len(a),1-len(bb)/len(b)]


def feature_uncertainty(fr,fs,gr,gs,replicates=500,seed=9200,paired=False):
    if fr.shape!=(len(gr),25) or fs.shape!=(len(gs),25):raise ValueError('Feature/group alignment')
    keys=['wasserstein','wasserstein_over_real_iqr','median_real','median_synthetic','std_real',
          'std_synthetic','iqr_real','iqr_synthetic','missing_real_fraction','missing_synthetic_fraction']
    point=np.asarray([_feature_stats(a,b) for a,b in zip(fr.T,fs.T)])
    samples=[]
    for ri,si in resamples(gr,gs,replicates,seed,paired):
        samples.append([_feature_stats(a,b) for a,b in zip(fr[ri].T,fs[si].T)])
    samples=np.asarray(samples)
    return {name:{key:interval(point[j,i],samples[:,j,i]) for i,key in enumerate(keys)} for j,name in enumerate(NAMES)}


def feature_judges(fr,fs,yr,ys,gr,gs,seed=9200,importance=True):
    if fr.shape!=(len(yr),25) or fs.shape!=(len(ys),25):raise ValueError('25 features required')
    if len(gr)!=len(fr) or len(gs)!=len(fs):raise ValueError('Feature/group alignment')
    if not np.isin(np.r_[yr,ys],range(4)).all():raise ValueError('Class labels outside 0..3')
    x=np.r_[fr,fs].copy();x[~np.isfinite(x)]=np.nan
    y=np.r_[np.zeros(len(fr)),np.ones(len(fs))]
    labels=np.r_[yr,ys].astype(int);groups=np.r_[gr,gs]
    if len(groups)!=len(y):raise ValueError('Group count')
    split=list(GroupShuffleSplit(n_splits=5,test_size=.3,random_state=seed).split(x,y,groups))
    results={}
    with threadpool_limits(limits=2):
        for name,cols in {'full':list(range(25)),**GROUPS}.items():
            xx=np.c_[x[:,cols],np.eye(4)[labels]]
            runs=[];imps=[]
            for j,(fit,dev) in enumerate(split):
                assert not set(groups[fit]) & set(groups[dev])
                if min(len(np.unique(y[fit])),len(np.unique(y[dev])))<2:
                    raise ValueError('Judge split lacks both real and synthetic labels')
                clf=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,random_state=seed+j)
                clf.fit(xx[fit],y[fit]);pred=clf.predict_proba(xx[dev])[:,1]
                runs.append(float(roc_auc_score(y[dev],pred)))
                if importance:
                    imps.append(permutation_importance(clf,xx[dev],y[dev],n_repeats=3,
                                scoring='roc_auc',random_state=seed+j).importances_mean)
            out=dict(value=float(np.mean(runs)),sd=float(np.std(runs,ddof=1)),minimum=min(runs),maximum=max(runs),
                     runs=runs,uncertainty='Five grouped 70/30 splits; SD and range, not a 95% CI')
            if imps:
                avg=np.mean(imps,axis=0);names=[NAMES[c] for c in cols]+[f'class_{c}' for c in CLASSES]
                out['top8_permutation_importance']=[dict(feature=names[i],auc_drop=float(avg[i]))
                                                   for i in np.argsort(-avg)[:8]]
            results[name]=out
    return results


def rhythm_rows(x,features,afib_cv):
    from synthetic_generation.beats import detect_r_peaks
    from synthetic_generation.leads import to_independent_8
    gaps=[]
    for signal in to_independent_8(x):
        peaks=detect_r_peaks(signal)
        gaps.append(bool(len(peaks)>1 and np.diff(peaks).max()>200))
    hr=features[:,0]
    return np.c_[features[:,4]>=3,np.isfinite(hr)&((hr<30)|(hr>200)),gaps,
                 np.isfinite(features[:,3])&(features[:,3]>afib_cv),~np.isfinite(hr)].astype(float)


def rhythm_uncertainty(rows,groups,replicates=500,seed=9200):
    keys=['fraction_ge3_peaks','hr_outside_30_200','any_rr_gap_over_2s','rr_cv_above_real_afib_median','hr_missing']
    parts=clusters(groups);rng=np.random.default_rng(seed)
    draws=np.asarray([rows[draw_clusters(parts,rng)].mean(0) for _ in range(replicates)])
    return {k:interval(rows[:,j].mean(),draws[:,j]) for j,k in enumerate(keys)}


def privacy_uncertainty(train,real,synthetic,gt,gr,gs,replicates=500,seed=9200):
    # Reference training set stays fixed: uncertainty is conditional on its release.
    ds=nearest_distance(synthetic,train);dr=nearest_distance(real,train)
    dm=nearest_distance(train,synthetic);dn=nearest_distance(real,synthetic)
    def stats(a,b):
        return [np.median(ds[b])/max(np.median(dr[a]),1e-12),np.mean(ds[b]<1e-6),np.median(ds[b]),np.median(dr[a])]
    point=stats(np.arange(len(real)),np.arange(len(synthetic)))
    draws=np.asarray([stats(a,b) for a,b in resamples(gr,gs,replicates,seed)])
    keys=['ratio_of_medians','exact_copy_fraction','synthetic_to_train_median','real_to_train_median']
    out={k:interval(point[i],draws[:,i]) for i,k in enumerate(keys)}
    def auc(a,b):return roc_auc_score(np.r_[np.ones(len(a)),np.zeros(len(b))],-np.r_[dm[a],dn[b]])
    out['membership_auc']=interval(auc(np.arange(len(train)),np.arange(len(real))),
                                 [auc(a,b) for a,b in resamples(gt,gr,replicates,seed+1)])
    out['note']='Limited negative-nearest-release-distance attack; no privacy guarantee; fixed fit reference'
    return out


def score_prepared(real,synthetic,train,embeddings,afib_cv,replicates=500,seed=9200,paired=False,privacy=True):
    """Prepared dictionaries contain X,y,groups,features,emb_task/emb_ssl, rhythm.

    Embeddings must already have a fixed fit-only scale. Match counts BEFORE call.
    """
    yr,ys,yt=real['y'],synthetic['y'],train['y']
    gr,gs,gt=real['groups'],synthetic['groups'],train['groups']
    if set(gr)&set(gt):raise ValueError('Real evaluation patients overlap fit privacy reference')
    result={'feature_judges':feature_judges(real['features'],synthetic['features'],yr,ys,gr,gs,seed),
            'by_class':{},'replicates':replicates,'afib_cv_threshold':afib_cv,
            'paired':paired,'interval_method':'percentile patient-cluster bootstrap; synthetic rows independent unless paired'}
    for c in np.unique(ys):
        ri,si,ti=np.flatnonzero(yr==c),np.flatnonzero(ys==c),np.flatnonzero(yt==c)
        if len(ri)!=len(si) or (privacy and len(ti)!=len(si)):raise ValueError('Class counts must match')
        row={'n_real':len(ri),'n_synthetic':len(si),'n_fit_reference':len(ti)}
        for emb in embeddings:
            row[emb]=prdc_uncertainty(real[emb][ri],synthetic[emb][si],gr[ri],gs[si],replicates,seed,paired)
        row['features']=feature_uncertainty(real['features'][ri],synthetic['features'][si],gr[ri],gs[si],replicates,seed,paired)
        row['rhythm_real']=rhythm_uncertainty(real['rhythm'][ri],gr[ri],replicates,seed)
        row['rhythm_synthetic']=rhythm_uncertainty(synthetic['rhythm'][si],gs[si],replicates,seed)
        if c==1:
            for key in ('rhythm_real','rhythm_synthetic'):row[key].pop('rr_cv_above_real_afib_median')
        if privacy:
            for space in (*embeddings,'X'):
                row[f'privacy_{space}']=privacy_uncertainty(train[space][ti].reshape(len(ti),-1),real[space][ri].reshape(len(ri),-1),
                         synthetic[space][si].reshape(len(si),-1),gt[ti],gr[ri],gs[si],replicates,seed)
        result['by_class'][CLASSES[c]]=row
    return result
