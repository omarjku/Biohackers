# Deploy to Hugging Face Spaces (Docker) — permanent public URL

The app ships as a Docker image with AMRFinderPlus + its database baked in, so a
HF **Docker** Space runs it as-is and **uploaded FASTAs work**. Your effort is
~5 min; then HF builds the image unattended (~15–25 min) and gives you a
permanent URL like `https://huggingface.co/spaces/<you>/genome-firewall`.

Everything HF needs is already in the repo: the `Dockerfile`, the Space metadata
(YAML block at the top of `README.md`: `sdk: docker`, `app_port: 8501`), the
model data (`data/processed/`), and example genomes.

## Steps

### 1. Create the Space
1. Sign in at <https://huggingface.co> (free account).
2. Go to <https://huggingface.co/new-space>.
3. Fill in:
   - **Owner / Space name:** `genome-firewall`
   - **License:** MIT
   - **Select the Space SDK:** **Docker** → **Blank** template
   - **Hardware:** CPU basic (free) — 2 vCPU / 16 GB is enough
   - **Visibility:** Public
4. Create it. HF makes an (almost empty) git repo for the Space.

### 2. Push this repo to the Space
You need a HF **write** access token: <https://huggingface.co/settings/tokens> →
New token → type **Write** → copy it.

From this repo (on the `demo/docker-setup` branch):
```bash
# add the Space as a git remote (use YOUR username)
git remote add hf https://huggingface.co/spaces/<your-hf-username>/genome-firewall

# push our branch to the Space's main branch
git push hf demo/docker-setup:main
```
When git prompts:
- **Username:** your HF username
- **Password:** paste the **write access token** (not your login password)

HF detects the `Dockerfile` and starts building automatically. Watch the
**Building** logs on the Space page.

### 3. Add your OpenAI key (for AI explanations)
On the Space page → **Settings** → **Variables and secrets** → **New secret**:
- **Name:** `OPENAI_API_KEY`
- **Value:** `sk-...your key...`

HF injects it as an environment variable, so the app turns on AI explanations
automatically (the toggle is on by default). Without it, the app still runs with
built-in deterministic explanations. **Do not commit the key** — the secret is
the correct place for it.

### 4. Use it
When the build finishes (Status → **Running**), open the Space URL. Upload a
whole-genome *E. coli* FASTA (~5 Mb) or pick a bundled example. First upload
takes ~30–90 s while AMRFinderPlus annotates.

## Updating later
Push again to redeploy:
```bash
git push hf demo/docker-setup:main
```

## Troubleshooting
- **Build fails on the AMRFinderPlus / micromamba step** — rare on HF's amd64
  builders (the arm64 issue that hits Apple Silicon does not apply here).
  Re-check the build log; a transient network error on the DB download just
  needs a rebuild (Settings → Factory rebuild).
- **App loads but is blank in the iframe** — the Dockerfile already runs
  Streamlit with `--server.enableCORS=false --server.enableXsrfProtection=false`,
  which is the fix; make sure you pushed the latest `Dockerfile`.
- **"amrfinder not found" on upload** — means an old image without the baked
  database; trigger a Factory rebuild.
- **A file is rejected as too large** — our example `*.fna` are ~5 MB (under
  HF's limit). If you add bigger genomes, track them with Git LFS:
  `git lfs track "*.fna"`.
