# link_analysis.py
import networkx as nx
import math
import random
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from networkx.algorithms import link_prediction
from networkx.algorithms.community import greedy_modularity_communities

def to_networkx_from_neo4j(graph_obj):
    """
    Convert a py2neo.Graph (neo4j) to a NetworkX graph.
    If you don't use Neo4j, ignore this function.
    """
    try:
        from py2neo import Graph as PyGraph
    except Exception:
        raise RuntimeError("py2neo required for Neo4j -> networkx conversion")
    G = nx.Graph()
    nrows = graph_obj.run("MATCH (n) RETURN id(n) as id, labels(n) as labels, properties(n) as props").data()
    for r in nrows:
        nid = r["id"]
        lab = r["labels"][0] if r["labels"] else "Entity"
        props = r["props"]
        G.add_node(nid, label=lab, **props)
    rels = graph_obj.run("MATCH (a)-[r]->(b) RETURN id(a) as a_id, id(b) as b_id, type(r) as rel, properties(r) as props").data()
    for r in rels:
        a = r["a_id"]; b = r["b_id"]; rel = r["rel"]; props = r["props"]
        G.add_edge(a, b, rel=rel, **props)
    return G

def compute_centralities(G, weight=None):
    degree = dict(G.degree(weight=weight))
    pagerank = nx.pagerank(G, weight=weight)
    try:
        eig = nx.eigenvector_centrality_numpy(G)
    except Exception:
        eig = nx.eigenvector_centrality(G, max_iter=200)
    betw = nx.betweenness_centrality(G, weight=weight, normalized=True)
    clos = nx.closeness_centrality(G, distance=None)
    df = pd.DataFrame.from_dict({
        "degree": degree,
        "pagerank": pagerank,
        "eigenvector": eig,
        "betweenness": betw,
        "closeness": clos
    }, orient="index").T
    df.index.name = "node"
    df = df.reset_index()
    return df, {"degree": degree, "pagerank": pagerank, "eigenvector": eig, "betweenness": betw, "closeness": clos}

def top_k_influencers(centrality_df, k=20, score_col="pagerank"):
    out = centrality_df.sort_values(score_col, ascending=False).head(k).reset_index(drop=True)
    return out

def top_k_bridges(centrality_df, k=20):
    out = centrality_df.sort_values("betweenness", ascending=False).head(k).reset_index(drop=True)
    return out

def detect_communities(G):
    communities = list(greedy_modularity_communities(G))
    node2comm = {}
    for i,comm in enumerate(communities):
        for n in comm:
            node2comm[n] = i
    comm_df = pd.DataFrame(list(node2comm.items()), columns=["node","community"])
    return communities, comm_df

def heuristic_link_scores(G, top_n=1000):
    pairs = set()
    nodes = list(G.nodes())
    for u in nodes:
        for v in nodes:
            if u>=v:
                continue
            if G.has_edge(u,v):
                continue
            pairs.add((u,v))
    pairs = list(pairs)
    if len(pairs) > top_n:
        pairs = random.sample(pairs, top_n)
    jaccard = list(link_prediction.jaccard_coefficient(G, pairs))
    adamic = list(link_prediction.adamic_adar_index(G, pairs))
    pref = list(link_prediction.preferential_attachment(G, pairs))
    resalloc = list(link_prediction.resource_allocation_index(G, pairs))
    records = []
    for (u,v,j),(u2,v2,aa),(u3,v3,pa),(u4,v4,ra) in zip(jaccard, adamic, pref, resalloc):
        records.append({"u":u,"v":v,"jaccard":j,"adamic_adar":aa,"pref_attach":pa,"resource_alloc":ra})
    df = pd.DataFrame(records)
    df["score_ensemble"] = df[["jaccard","adamic_adar","pref_attach","resource_alloc"]].fillna(0).mean(axis=1)
    df = df.sort_values("score_ensemble", ascending=False).reset_index(drop=True)
    return df

def supervised_link_prediction(G, sample_size_pos=5000, sample_size_neg=5000, test_size=0.2, random_state=42):
    edges = list(G.edges())
    if len(edges) == 0:
        raise ValueError("Graph has no edges")
    pos_samples = random.sample(edges, min(sample_size_pos, len(edges)))
    pos_pairs = [(u,v,1) for u,v in pos_samples]
    non_edges = list(nx.non_edges(G))
    neg_samples = random.sample(non_edges, min(sample_size_neg, len(non_edges)))
    neg_pairs = [(u,v,0) for u,v in neg_samples]
    pairs = pos_pairs + neg_pairs
    rows = []
    for u,v,label in pairs:
        cn = len(list(nx.common_neighbors(G,u,v)))
        try:
            jac = next(link_prediction.jaccard_coefficient(G, [(u,v)]))[2]
        except StopIteration:
            jac = 0.0
        try:
            aa = next(link_prediction.adamic_adar_index(G, [(u,v)]))[2]
        except StopIteration:
            aa = 0.0
        pa = next(link_prediction.preferential_attachment(G, [(u,v)]))[2]
        try:
            ra = next(link_prediction.resource_allocation_index(G, [(u,v)]))[2]
        except StopIteration:
            ra = 0.0
        deg_u = G.degree(u)
        deg_v = G.degree(v)
        rows.append({"u":u,"v":v,"cn":cn,"jaccard":jac,"adamic_adar":aa,"pref_attach":pa,"resource_alloc":ra,"deg_u":deg_u,"deg_v":deg_v,"label":label})
    df = pd.DataFrame(rows)
    X = df[["cn","jaccard","adamic_adar","pref_attach","resource_alloc","deg_u","deg_v"]].fillna(0)
    y = df["label"].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=test_size, random_state=random_state, stratify=y)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, probs) if len(set(y_test))>1 else None
    ap = average_precision_score(y_test, probs) if len(set(y_test))>1 else None
    return {"model":clf, "scaler":scaler, "auc":auc, "ap":ap, "features_df":df}

def predict_top_links_supervised(G, clf_info, candidate_sample=20000, top_k=100):
    non_edges = list(nx.non_edges(G))
    if len(non_edges) == 0:
        return pd.DataFrame([], columns=["u","v","score"])
    if len(non_edges) > candidate_sample:
        candidates = random.sample(non_edges, candidate_sample)
    else:
        candidates = non_edges
    rows = []
    for u,v in candidates:
        cn = len(list(nx.common_neighbors(G,u,v)))
        try:
            jac = next(link_prediction.jaccard_coefficient(G, [(u,v)]))[2]
        except StopIteration:
            jac = 0.0
        try:
            aa = next(link_prediction.adamic_adar_index(G, [(u,v)]))[2]
        except StopIteration:
            aa = 0.0
        pa = next(link_prediction.preferential_attachment(G, [(u,v)]))[2]
        try:
            ra = next(link_prediction.resource_allocation_index(G, [(u,v)]))[2]
        except StopIteration:
            ra = 0.0
        deg_u = G.degree(u)
        deg_v = G.degree(v)
        rows.append({"u":u,"v":v,"cn":cn,"jaccard":jac,"adamic_adar":aa,"pref_attach":pa,"resource_alloc":ra,"deg_u":deg_u,"deg_v":deg_v})
    cand_df = pd.DataFrame(rows)
    Xcand = cand_df[["cn","jaccard","adamic_adar","pref_attach","resource_alloc","deg_u","deg_v"]].fillna(0)
    Xs = clf_info["scaler"].transform(Xcand)
    probs = clf_info["model"].predict_proba(Xs)[:,1]
    cand_df["score"] = probs
    out = cand_df.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)
    return out

def annotate_graph_with_communities_and_scores(G, centrality_df=None, comm_df=None, prefix="analysis"):
    if centrality_df is not None:
        for _,r in centrality_df.iterrows():
            n = r["node"]
            for c in ["degree","pagerank","eigenvector","betweenness","closeness"]:
                if n in G:
                    G.nodes[n][f"{prefix}_{c}"] = float(r[c])
    if comm_df is not None:
        for _,r in comm_df.iterrows():
            n = r["node"]
            if n in G:
                G.nodes[n][f"{prefix}_community"] = int(r["community"])
    return G

# Example usage string returned for quick reference
EXAMPLE = """
Usage example:

from link_analysis import *
G = nx.read_graphml('crime_graph.graphml')   # or build/read your NetworkX graph
central_df, central_map = compute_centralities(G)
print(top_k_influencers(central_df, k=10, score_col='pagerank'))
print(top_k_bridges(central_df, k=10))

communities, comm_df = detect_communities(G)
print(comm_df.head())

heur_df = heuristic_link_scores(G, top_n=5000)
print(heur_df.head(20))

clf_info = supervised_link_prediction(G, sample_size_pos=2000, sample_size_neg=2000)
print('AUC', clf_info['auc'], 'AP', clf_info['ap'])

preds = predict_top_links_supervised(G, clf_info, candidate_sample=10000, top_k=50)
print(preds.head())
"""

