"""
Baseline de feature engineering com LLM (estilo CAAFE) no dataset
Heart Disease (UCI, Cleveland), classificacao binaria.

Desenho experimental (evita viés de selecao):
  - Repete N vezes um split externo estratificado (80% dev / 20% teste).
  - Dentro de cada split, o LLM propoe features e a aceitacao e decidida
    por validacao cruzada INTERNA, usando apenas o conjunto dev.
  - O conjunto de teste so e usado no final, para comparar
    baseline x baseline+features do LLM.
  - Toda feature gerada deve ser row-wise (sem estatisticas de outras
    linhas). Isso e verificado automaticamente (guarda contra vazamento).

Uso:
  pip install pandas numpy scikit-learn ucimlrepo anthropic
  export ANTHROPIC_API_KEY=...
  python feature_engineering_llm.py                  # baixa o dataset via ucimlrepo
  python feature_engineering_llm.py --csv cleveland.csv
  python feature_engineering_llm.py --mock           # testa o pipeline sem API

ATENCAO: o codigo gerado pelo LLM e executado com exec(). A checagem estatica
abaixo reduz riscos mas NAO e um sandbox. Rode em ambiente isolado
(container/VM) se for usar com dados que nao sejam publicos.
"""
import argparse
import ast
import json
import re
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedShuffleSplit,
    cross_val_score,
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

DATASET_DESCRIPTION = (
    "Cleveland Clinic Foundation subset of the UCI Heart Disease dataset: about 303 "
    "patients, 13 clinical attributes. Target: 1 if angiographic heart disease is "
    "present, 0 otherwise."
)

# Confira estas descricoes/codificacoes na documentacao oficial do UCI.
COLUMN_DESCRIPTIONS = {
    "age": "numeric, age in years",
    "sex": "categorical, 1 = male, 0 = female",
    "cp": "categorical, chest pain type (1 typical angina, 2 atypical angina, 3 non-anginal pain, 4 asymptomatic)",
    "trestbps": "numeric, resting blood pressure in mm Hg on admission",
    "chol": "numeric, serum cholesterol in mg/dl",
    "fbs": "categorical, fasting blood sugar > 120 mg/dl (1 = true, 0 = false)",
    "restecg": "categorical, resting ECG (0 normal, 1 ST-T wave abnormality, 2 left ventricular hypertrophy)",
    "thalach": "numeric, maximum heart rate achieved",
    "exang": "categorical, exercise-induced angina (1 = yes, 0 = no)",
    "oldpeak": "numeric, ST depression induced by exercise relative to rest",
    "slope": "categorical, slope of the peak exercise ST segment (1 upsloping, 2 flat, 3 downsloping)",
    "ca": "numeric count, number of major vessels (0-3) colored by fluoroscopy; may be NaN",
    "thal": "categorical, thalassemia (3 normal, 6 fixed defect, 7 reversible defect); may be NaN",
}


def fmt(x, nd=4):
    """Formata numero com virgula decimal."""
    return f"{x:.{nd}f}".replace(".", ",")


# ----------------------------------------------------------------- dados
def load_data(csv_path=None):
    if csv_path:
        df = pd.read_csv(csv_path, na_values="?")
        target_col = "num" if "num" in df.columns else "target"
        X, y_raw = df[FEATURES], df[target_col]
    else:
        from ucimlrepo import fetch_ucirepo

        ds = fetch_ucirepo(id=45)  # Heart Disease (confira o id no UCI)
        X, y_raw = ds.data.features[FEATURES], ds.data.targets.iloc[:, 0]
    y = (y_raw > 0).astype(int)  # 0 = sem doenca; 1..4 = doenca
    X = X.apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    y = y.reset_index(drop=True)
    print(f"Dataset: {X.shape[0]} linhas, {X.shape[1]} features, "
          f"prevalencia da classe 1 = {fmt(y.mean(), 3)}")
    if X.shape[0] != 303:
        print("AVISO: o Cleveland processado tem 303 linhas; confira a fonte dos dados.")
    return X, y


# ---------------------------------------------------------------- modelo
def make_model(seed):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),  # dentro do fold: sem vazamento
        ("rf", RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)),
    ])


def cv_auc(X, y, seed, repeats):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=seed)
    return cross_val_score(make_model(seed), X, y, cv=cv, scoring="roc_auc")


def fit_auc(Xtr, ytr, Xte, yte, seed):
    model = make_model(seed).fit(Xtr, ytr)
    return roc_auc_score(yte, model.predict_proba(Xte)[:, 1])


# ------------------------------------------------- codigo gerado pelo LLM
BLOCKED_NAMES = {"open", "eval", "exec", "compile", "__import__", "getattr",
                 "setattr", "globals", "locals", "os", "sys", "subprocess", "input"}


def extract_code(reply):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", reply, re.DOTALL)
    return m.group(1).strip() if m else None


def static_check(code):
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in {"pandas", "numpy"}:
                    raise ValueError(f"import nao permitido: {a.name}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in {"pandas", "numpy"}:
                raise ValueError(f"import nao permitido: {node.module}")
        elif isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            raise ValueError(f"nome nao permitido: {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError(f"atributo nao permitido: {node.attr}")


def compile_features(code):
    static_check(code)
    ns = {"pd": pd, "np": np}
    exec(code, ns)
    fn = ns.get("add_features")
    if not callable(fn):
        raise ValueError("a funcao add_features nao foi definida")
    return fn


def apply_features(fn, X, validate=True):
    """Aplica fn a X e devolve (novas_colunas, problemas)."""
    out = fn(X.copy())
    if not isinstance(out, pd.DataFrame) or len(out) != len(X):
        raise ValueError("add_features deve devolver um DataFrame com o mesmo numero de linhas")
    if not out[X.columns].equals(X):
        raise ValueError("add_features alterou as colunas originais")
    new_cols = [c for c in out.columns if c not in X.columns]
    problems, kept = [], {}

    sub = None
    if validate:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(len(X), size=min(60, len(X) // 2), replace=False))
        sub = fn(X.iloc[idx].copy())

    for c in new_cols:
        if not str(c).startswith("fe_"):
            problems.append(f"{c}: nome sem prefixo fe_")
            continue
        s = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan).astype(float)
        if validate:
            if s.notna().mean() < 0.5 or s.nunique() <= 1:
                problems.append(f"{c}: constante ou quase toda NaN")
                continue
            # guarda contra vazamento: valor da linha nao pode depender das outras
            a = pd.to_numeric(sub[c], errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(float)
            b = s.iloc[idx].to_numpy(float)
            if not np.allclose(a, b, rtol=1e-6, atol=1e-9, equal_nan=True):
                problems.append(f"{c}: depende de outras linhas (possivel vazamento)")
                continue
        kept[c] = s
    return pd.DataFrame(kept, index=X.index), problems


def build_X(X, accepted):
    parts = [X]
    for fn, cols in accepted:
        new, _ = apply_features(fn, X, validate=False)
        parts.append(new[cols])
    return pd.concat(parts, axis=1)


# ---------------------------------------------------------------- prompt
def build_prompt(n_features, history, current_auc):
    tpl = (Path(__file__).with_name("prompt_template.txt")).read_text(encoding="utf-8")
    cols = "\n".join(f"- {c} ({d})" for c, d in COLUMN_DESCRIPTIONS.items())
    hist = "\n".join(history) if history else "(no attempts yet)"
    return (tpl.replace("<<DATASET>>", DATASET_DESCRIPTION)
               .replace("<<COLUMNS>>", cols)
               .replace("<<HISTORY>>", hist)
               .replace("<<CURRENT_AUC>>", current_auc)
               .replace("<<N_FEATURES>>", str(n_features)))


# ------------------------------------------------------------------- LLM
MOCK_REPLIES = [
    "```python\ndef add_features(df):\n    out = df.copy()\n"
    "    out['fe_hr_ratio'] = df['thalach'] / (220 - df['age'])\n"
    "    out['fe_bp_chol'] = df['trestbps'] * df['chol'] / 1000\n"
    "    out['fe_st_flat_down'] = ((df['slope'] >= 2) & (df['oldpeak'] > 0)).astype(float)\n"
    "    return out\n```",
    "```python\ndef add_features(df):\n    out = df.copy()\n"
    "    out['fe_chol_z'] = (df['chol'] - df['chol'].mean()) / df['chol'].std()\n"
    "    out['fe_age_exang'] = df['age'] * df['exang']\n"
    "    return out\n```",
    "```python\nimport os\ndef add_features(df):\n    return df.copy()\n```",
]


def make_llm(args):
    if args.mock:
        state = {"i": 0}

        def mock(prompt):
            r = MOCK_REPLIES[state["i"] % len(MOCK_REPLIES)]
            state["i"] += 1
            return r
        return mock

    import anthropic

    client = anthropic.Anthropic()

    def call(prompt):
        kw = {}
        if args.temperature is not None:
            kw["temperature"] = args.temperature
        r = client.messages.create(model=args.model, max_tokens=2000,
                                   messages=[{"role": "user", "content": prompt}], **kw)
        return "".join(b.text for b in r.content if b.type == "text")
    return call


# -------------------------------------------------------------- execucao
def run_split(i, X, y, tr, te, args, llm):
    seed = args.seed + i
    X_dev, y_dev = X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True)
    X_te, y_te = X.iloc[te].reset_index(drop=True), y.iloc[te].reset_index(drop=True)

    cur, cur_scores = X_dev.copy(), cv_auc(X_dev, y_dev, seed, args.inner_repeats)
    accepted, history = [], []
    log = {"split": i, "inner_baseline_auc": float(cur_scores.mean()), "rounds": []}

    for it in range(args.iterations):
        prompt = build_prompt(args.features_per_iter, history, fmt(cur_scores.mean()))
        reply = llm(prompt)
        rnd = {"iteration": it, "prompt": prompt, "reply": reply, "events": []}
        log["rounds"].append(rnd)

        code = extract_code(reply)
        if code is None:
            history.append("- (the previous reply had no python code block)")
            rnd["events"].append("sem bloco de codigo")
            continue
        try:
            fn = compile_features(code)
            new, problems = apply_features(fn, X_dev)
        except Exception as e:  # codigo invalido
            history.append(f"- INVALID code ({type(e).__name__}: {e}); fix and avoid this.")
            rnd["events"].append(f"codigo invalido: {e}")
            continue
        for p in problems:
            history.append(f"- REJECTED {p}")
            rnd["events"].append(f"rejeitada: {p}")

        kept = []
        for col in new.columns:
            if col in cur.columns:
                history.append(f"- REJECTED {col}: name already used")
                continue
            cand = pd.concat([cur, new[[col]]], axis=1)
            sc = cv_auc(cand, y_dev, seed, args.inner_repeats)
            gain = sc.mean() - cur_scores.mean()
            if gain > args.min_delta:
                cur, cur_scores = cand, sc
                kept.append(col)
                history.append(f"- ACCEPTED {col}: inner CV AUC gain {fmt(gain)}")
                rnd["events"].append(f"aceita {col} (ganho {fmt(gain)})")
            else:
                history.append(f"- REJECTED {col}: inner CV AUC gain {fmt(gain)} (not enough)")
                rnd["events"].append(f"rejeitada {col} (ganho {fmt(gain)})")
        if kept:
            accepted.append((fn, kept))
            log.setdefault("accepted_code", []).append({"cols": kept, "code": code})

    base_te = fit_auc(X_dev, y_dev, X_te, y_te, seed)
    enr_te = fit_auc(build_X(X_dev, accepted), y_dev, build_X(X_te, accepted), y_te, seed)
    n_acc = sum(len(c) for _, c in accepted)
    log.update(test_auc_baseline=float(base_te), test_auc_enriched=float(enr_te), n_accepted=n_acc)
    print(f"[split {i}] teste AUC baseline={fmt(base_te)} | com features LLM={fmt(enr_te)} "
          f"| features aceitas={n_acc}")
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--iterations", type=int, default=3, help="chamadas ao LLM por split")
    ap.add_argument("--features-per-iter", type=int, default=3)
    ap.add_argument("--outer-repeats", type=int, default=5, help="numero de splits dev/teste")
    ap.add_argument("--inner-repeats", type=int, default=2, help="repeticoes do CV 5-fold interno")
    ap.add_argument("--min-delta", type=float, default=0.003, help="ganho minimo de AUC para aceitar")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mock", action="store_true", help="LLM simulado (teste do pipeline)")
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    X, y = load_data(args.csv)
    llm = make_llm(args)
    splitter = StratifiedShuffleSplit(n_splits=args.outer_repeats, test_size=0.2,
                                      random_state=args.seed)
    logs = []
    t0 = time.time()
    for i, (tr, te) in enumerate(splitter.split(X, y)):
        logs.append(run_split(i, X, y, tr, te, args, llm))

    b = np.array([l["test_auc_baseline"] for l in logs])
    e = np.array([l["test_auc_enriched"] for l in logs])
    d = e - b
    sd = lambda v: v.std(ddof=1) if len(v) > 1 else 0.0
    print("\n=== Resumo (AUC no conjunto de teste) ===")
    print(f"baseline        : {fmt(b.mean())} +- {fmt(sd(b))}")
    print(f"baseline + LLM  : {fmt(e.mean())} +- {fmt(sd(e))}")
    print(f"diferenca media : {fmt(d.mean())} +- {fmt(sd(d))}  "
          f"(positiva em {int((d > 0).sum())}/{len(d)} splits)")
    print(f"tempo: {fmt(time.time() - t0, 1)} s")
    print("Obs.: os splits se sobrepoem, entao nao sao independentes; "
          "evite testes de hipotese ingenuos sobre estes numeros.")

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps({"args": vars(args), "splits": logs}, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"log completo (prompts, respostas, codigo aceito): {path}")


if __name__ == "__main__":
    main()
