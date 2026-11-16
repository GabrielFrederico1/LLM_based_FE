"""
Baseline de feature engineering com LLM (estilo CAAFE) no dataset
Titanic (Titanic-Dataset.csv), classificacao binaria (Survived).

Desenho experimental (evita viés de selecao):
  - Repete N vezes um split externo estratificado (80% dev / 20% teste).
  - Dentro de cada split, o LLM propoe features e a aceitacao e decidida
    por validacao cruzada INTERNA, usando apenas o conjunto dev.
  - O conjunto de teste so e usado no final, para comparar
    baseline x baseline+features do LLM.
  - Toda feature gerada deve ser row-wise (sem estatisticas de outras
    linhas). Isso e verificado automaticamente (guarda contra vazamento).

Uso:
  pip install pandas numpy scikit-learn google-genai
  # coloque sua chave do Gemini em API_KEY.txt
  python feature_engineering_llm.py                  # usa Titanic-Dataset.csv na pasta
  python feature_engineering_llm.py --csv outro.csv  # caminho alternativo
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
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedShuffleSplit,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

FEATURES = ["Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]

DATASET_DESCRIPTION = (
    "Titanic passenger survival dataset: 891 passengers, 10 attributes "
    "(demographic, ticket, fare, cabin, and family information). "
    "Target: 1 if the passenger survived, 0 otherwise."
)

COLUMN_DESCRIPTIONS = {
    "Pclass": "categorical (integer 1, 2, 3), ticket class (1 = 1st/upper, 2 = 2nd/middle, 3 = 3rd/lower)",
    "Name": "text string, passenger full name including title (e.g., 'Braund, Mr. Owen Harris')",
    "Sex": "categorical string, passenger sex ('male' or 'female')",
    "Age": "numeric, age in years (fractional if less than 1); may be NaN (177 missing)",
    "SibSp": "numeric integer, number of siblings or spouses aboard the Titanic",
    "Parch": "numeric integer, number of parents or children aboard the Titanic",
    "Ticket": "text string, ticket number/code (e.g., 'A/5 21171', '347082')",
    "Fare": "numeric, passenger fare paid",
    "Cabin": "text string, cabin number (e.g., 'C85'); often NaN (687 missing)",
    "Embarked": "categorical string, port of embarkation ('C' = Cherbourg, 'Q' = Queenstown, 'S' = Southampton); may be NaN (2 missing)",
}


def fmt(x, nd=4):
    """Formata numero com virgula decimal."""
    return f"{x:.{nd}f}".replace(".", ",")


# ----------------------------------------------------------------- dados
def load_data(csv_path=None):
    """Carrega o dataset Titanic a partir de Titanic-Dataset.csv (ou caminho informado)."""
    path = Path(csv_path) if csv_path else Path(__file__).with_name("Titanic-Dataset.csv")
    df = pd.read_csv(path)
    X = df[FEATURES].copy().reset_index(drop=True)
    y = df["Survived"].astype(int).reset_index(drop=True)
    print(f"Dataset: {X.shape[0]} linhas, {X.shape[1]} features, "
          f"prevalencia da classe 1 = {fmt(y.mean(), 3)}")
    return X, y


# ---------------------------------------------------------------- modelo
def make_model(seed):
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), make_column_selector(dtype_include=np.number)),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="__missing__")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", max_categories=20, sparse_output=False)),
        ]), make_column_selector(dtype_exclude=np.number)),
    ])
    return Pipeline([
        ("prep", prep),  # dentro do fold: sem vazamento
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
    "    out['fe_family_size'] = df['SibSp'] + df['Parch'] + 1\n"
    "    out['fe_fare_per_person'] = df['Fare'] / out['fe_family_size']\n"
    "    out['fe_has_cabin'] = df['Cabin'].notna().astype(float)\n"
    "    return out\n```",
    "```python\ndef add_features(df):\n    out = df.copy()\n"
    "    out['fe_fare_z'] = (df['Fare'] - df['Fare'].mean()) / df['Fare'].std()\n"
    "    out['fe_is_female_upper'] = ((df['Sex'] == 'female') & (df['Pclass'] == 1)).astype(float)\n"
    "    return out\n```",
    "```python\nimport os\ndef add_features(df):\n    return df.copy()\n```",
]


def load_api_key():
    """Le a chave da API do Gemini exclusivamente do arquivo API_KEY.txt."""
    key_file = Path(__file__).with_name("API_KEY.txt")
    if not key_file.exists():
        raise RuntimeError(f"Arquivo de chave nao encontrado: {key_file}")
    for raw_line in key_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            _, line = line.split("=", 1)
            line = line.strip()
        line = line.strip("\"'")
        if line:
            return line
    raise RuntimeError(f"Nenhuma chave encontrada em {key_file}")


def make_llm(args):
    """Cria a funcao de chamada ao LLM (simulado com --mock ou Gemini via google-genai)."""
    if args.mock:
        state = {"i": 0}

        def mock(prompt):
            r = MOCK_REPLIES[state["i"] % len(MOCK_REPLIES)]
            state["i"] += 1
            return r
        return mock

    from google import genai
    from google.genai import types

    api_key = load_api_key()
    client = genai.Client(api_key=api_key)

    def call(prompt):
        cfg_kw = {
            "max_output_tokens": 8192,
            "thinking_config": types.ThinkingConfig(thinking_level=args.thinking_level),
        }
        if args.temperature is not None:
            cfg_kw["temperature"] = args.temperature
        r = client.models.generate_content(
            model=args.model,
            contents=prompt,
            config=types.GenerateContentConfig(**cfg_kw),
        )
        if not r.text:
            finish = r.candidates[0].finish_reason if r.candidates else "desconhecido"
            print(f"[AVISO] Resposta vazia do Gemini (finish_reason={finish})")
        return r.text or ""
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
    ap.add_argument("--csv", default=None, help="caminho para CSV ou .data local")
    ap.add_argument("--model", default="gemini-4-hydrogen", help="modelo Gemini (padrao: gemini-4-hydrogen, Free Tier)")
    ap.add_argument("--thinking-level", default="LOW", choices=["MINIMAL", "LOW", "MEDIUM", "HIGH"],
                    help="nivel de raciocinio do Gemini (padrao: LOW)")
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
