# Putting this on the internet

The submission needs a link a judge can click. This is how to get one, and what
had to change to make it possible.

## The problem

The working set is about 110 MB: a 72 MB glucose table plus 39 MB of cached
forecasts. That is more than belongs in a git repository, more than a free host
wants to clone on every deploy, and — more importantly — **it is not ours to
redistribute.** The OpenAPS Data Commons is donated patient data.

## The bundle

`python -m scripts.make_demo_bundle` writes a **4 MB** `demo_data/` folder:

- four held-out wearers, 30–60 days each, chosen as the stretch of their record
  containing the most lows, so the demo is a real span rather than a reel of
  good moments;
- their **precomputed** forecasts, so the hosted app never touches the archive;
- the shipped model (0.95 MB) so the Live page still works;
- the results JSON the evidence pages read.

The app detects this automatically. `demo_mode()` is true when the full cache is
absent and the bundle is present, and the loaders fall back accordingly — the
wearer list narrows to what actually shipped, and per-wearer facts come from
`summary.json` instead of the raw table.

Verify a deploy before pushing one:

```bash
python -m scripts.make_demo_bundle
mkdir -p /tmp/deploy_test
tar cf - app.py src views scripts demo_data requirements.txt .streamlit \
  | (cd /tmp/deploy_test && tar xf -)
cd /tmp/deploy_test && streamlit run app.py
```

That copies exactly what a host would see. If it runs there, it will run there.

## Streamlit Community Cloud — free, and the shortest path

1. Push this folder to a **public** GitHub repository.
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. **New app** → pick the repo → main file `app.py` → Deploy.
4. You get `https://<something>.streamlit.app`. That is the submission link.

Nothing else to configure: `requirements.txt` is the dependency list and
`.streamlit/config.toml` sets the theme.

Two things to know. The free tier **sleeps after inactivity** and takes 30–60
seconds to wake, so open the link a few minutes before anyone else does. And
`demo_data/` must be committed — it is the app's whole dataset up there.

## What must stay out of the repository

`.gitignore` already covers these; check before pushing:

- `data/` — the raw archive and the full parquet cache
- `artifacts/*.pt` — trained weights, except the one copied into the bundle
- `artifacts/*.log`

The bundle contains model outputs over donated traces, not the traces
themselves, which is the line worth keeping.

## If you would rather not use Streamlit Cloud

`Dockerfile` is not included but the app is a plain Streamlit process:

```
pip install -r requirements.txt
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

Any host that runs a long-lived Python process works — Railway, Render, Fly.
Vercel and Netlify do not, since they expect a static site or short-lived
functions and this is neither.
