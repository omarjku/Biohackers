# Genome Firewall — Demo Setup on a Fresh Machine

Goal: run the app in Docker on a clean computer and expose it at a short public
URL for a live demo. Every command is copy-paste. Budget ~25 min the first time
(most of it an unattended image build).

> **Why a bundle, not `git clone`?** The trained model data (`data/processed/`)
> and the example genomes (`data/raw/fasta_demo/`) are gitignored — they are NOT
> on GitHub. Use the `genome-firewall-demo.zip` bundle, which contains them.

---

## 0. What you need

- A Mac (Apple Silicon or Intel), Windows, or Linux machine.
- The file **`genome-firewall-demo.zip`** (AirDrop / USB / cloud from Omar).
- ~5 GB free disk (Docker image is ~2–3 GB).
- Internet (the build downloads the AMR database once).

---

## 1. Install Docker Desktop

**macOS (Homebrew):**
```bash
brew install --cask docker
```
Or download from https://www.docker.com/products/docker-desktop and drag to
Applications.

**Windows / Linux:** download Docker Desktop from the link above and install.

Then **launch Docker Desktop** (Applications → Docker, or `open -a Docker` on Mac).
On first launch: click **Accept** on the agreement, enter the machine password if
asked, and **skip any sign-in** (no account needed).

Wait until the whale 🐳 menu-bar icon is steady / the app says **"Engine
running."** Verify:
```bash
docker info
```
If that prints server details (not an error), the engine is up.

> **Stuck on "Starting…" for a long time?** Quit Docker Desktop fully and reopen
> it. On Apple Silicon, also make sure macOS is updated. As a last resort:
> Docker Desktop → Settings (gear) → **Troubleshoot** → **Reset to factory
> defaults**, then reopen.

---

## 2. Unzip the bundle

```bash
cd ~/Desktop
unzip genome-firewall-demo.zip -d genome-firewall
cd genome-firewall
```
You should see `Dockerfile`, `src/`, `requirements.txt`, and `data/processed/`.
Confirm the model data made it across:
```bash
ls data/processed/features.csv   # must exist
```

---

## 3. Build the image

```bash
docker build -t genome-firewall .
```
⏳ **First build ~10–20 min**, image ~2–3 GB. It installs AMRFinderPlus and
downloads the AMR database *into the image*, so nothing is fetched at runtime.
Cached afterwards — later builds are seconds.

> **If the build fails on the `micromamba install` / AMRFinderPlus step**
> (common on Apple Silicon — no native arm64 package), rebuild forcing Intel
> emulation:
> ```bash
> docker build --platform=linux/amd64 -t genome-firewall .
> ```
> Slower to annotate but reliable for a demo.

---

## 4. Run it

```bash
docker run --rm -p 8501:8501 genome-firewall
```
Open **http://localhost:8501** in a browser.

Leave this terminal running — it's the live server. `Ctrl+C` stops it.

**Optional — GPT explanations toggle** (needs an OpenAI key; app works fully
without it):
```bash
docker run --rm -p 8501:8501 -e OPENAI_API_KEY=sk-... genome-firewall
```

---

## 5. Test it

**Instant path (bundled examples):** open the *"…or run a bundled example E. coli
genome"* expander and pick one. These render in seconds (their annotation is
pre-cached).

**Real path (upload a new FASTA):** use the file uploader at the top. AMRFinderPlus
annotates it live — **~30–90 s per genome** — then the cards appear. Only
*E. coli* is in scope; other species will run but the results are meaningless.

Every result shows the mandatory "confirm with standard laboratory testing"
disclaimer — that's by design.

---

## 6. Short public URL for the demo

Free, no signup — a Cloudflare quick tunnel. **Keep step 4 running**, then in a
**second terminal**:

**macOS:**
```bash
brew install cloudflared
cloudflared tunnel --url http://localhost:8501
```
**Windows/Linux:** install cloudflared from
https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/
then run the same `cloudflared tunnel --url http://localhost:8501`.

It prints a URL like `https://random-words.trycloudflare.com`. That's your
shareable link — live as long as both the `docker run` and the `cloudflared`
commands keep running. The URL changes each time you restart the tunnel.

> **Tunnel loads but the app stays blank / "Please wait…"?** Stop step 4 and
> rerun with tunnel-friendly flags:
> ```bash
> docker run --rm -p 8501:8501 genome-firewall \
>   streamlit run src/app.py --server.port=8501 --server.address=0.0.0.0 \
>   --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false
> ```

---

## 7. Troubleshooting quick table

| Symptom | Fix |
|---|---|
| `Cannot connect to the Docker daemon` | Docker Desktop isn't running — open it, wait for "Engine running." |
| Build fails on AMRFinderPlus/conda | Rebuild with `--platform=linux/amd64` (see step 3). |
| `port is already allocated` on run | Something's on 8501. Use another: `-p 8502:8501`, then open `localhost:8502`. |
| App shows "AMRFinderPlus is not installed" on upload | You're running outside the container, or the image build skipped the DB. Rebuild the image. |
| Upload just spins | Normal — annotation takes 30–90 s. Watch the "Executing pipeline" status box. |
| cloudflared URL blank | Use the CORS/XSRF flags in step 6. |

---

## 8. Shut down

- Stop the tunnel: `Ctrl+C` in the cloudflared terminal.
- Stop the app: `Ctrl+C` in the `docker run` terminal.
- Free disk later: `docker image rm genome-firewall`.
