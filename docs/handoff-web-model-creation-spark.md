# Zerfoo website/model creation: self-contained Spark takeover

Snapshot: 2026-09-10 America/Los_Angeles (some tool timestamps are September 11 UTC).
Owner requested GPT-5.3-Codex-Spark take over to conserve Astra usage.
Continue implementation from this snapshot; do not replan from scratch.

## 1. User decisions and authority

- Zerfoo is for creating, training, evaluating and running AI. It must not be
  positioned narrowly as an inference engine.
- Company: **Sire Run, Inc.**, website **https://sire.run**. Feza is a former
  company. Remove Feza from current company attribution, but preserve historical
  engineering records and legacy URLs where needed for redirects.
- Primary website: **https://zer.foo**. Migrate from **https://zerfoo.feza.ai**,
  retaining documentation paths and setting up permanent legacy redirects.
- Website conversation designs a portable project. NEVER train visitor models,
  upload full datasets, execute generated code or dispatch visitor jobs to the
  founder's home DGX. Local Claude Code/Codex does training and prediction.
- The user explicitly authorized implementation and migration: “Plan it and do
  it all.” Do not ask again about routine implementation or deployment.
- Public chat LLM budget explicitly approved: **$20 TOTAL, then stop hosted
  chat**. Not $20/day or per visitor. Local-agent handoff remains available.
- Website LLM: **OpenRouter `z-ai/glm-5.3-flash`**, existing
  `OPENROUTER_API_KEY`. Do not switch model/provider without a reason and owner
  direction. No Astra bulk work and no unrequested subagents.
- First actually verified runnable recipe: numeric CSV classification, CPU
  float32, Dense(16) -> ReLU -> Dense(classes), cross entropy + AdamW. Other
  tasks produce explicitly unsupported design briefs for local engineering.
- Research notes are candidates, not proof of architecture support. Preserve
  their unreviewed status until a bounded semantic review has been performed.
- User wants a one-line instruction visitors paste into Claude Code/Codex; it
  must teach the agent how to use Zerfoo and Kazi on their own hardware.

## 2. Exact working copies and current state

Core checkout: `/Users/dndungu/Code/zerfoo/zerfoo`.
Current main: `53499981f4e33acc27fadfd9f72a5a762cb0e579`.
Another agent merged PR #1005 during this work, including the distillation
adapter, ADR 098 and original web plan. PR #1004 was already merged as
`bfbb707111127cedb017d741f8b9337da1ed8632`. Do not repeat old draft-PR claims.
Read `ajent.social` at startup; don't overwrite other agents' changes.

Website checkout: **`/private/tmp/zerfoo-web-launch`**.
Remote: `https://github.com/zerfoo/zerfoo.github.io`.
Base HEAD: `0f66239dace2828e04aad62fc64b5eb87f0a99c9`.
All website implementation in this handoff is UNCOMMITTED in that checkout.
It is not in the core checkout. Preserve this directory and commit its changes
on an implementation branch early. Do not reclone and lose the changes.

Important website files:

| File | Current purpose |
|---|---|
| `content/_index.html` | Replaced inference-only homepage with create/train/run positioning |
| `hugo.yaml`, `static/CNAME` | New canonical domain zer.foo |
| `static/create/index.html` | Two-pane conversation and model preview |
| `static/create/style.css` | Responsive blue/white design with self-hosted Inter |
| `static/create/app.mjs` | Calls design API, renders safe text, downloads ZIP, copies handoff |
| `static/create/bundle.mjs` | Fixed-file ZIP generator, local Python training/prediction runners |
| `static/start/index.html` | Agent bootstrap instructions, Kazi discovery workflow |
| `worker/index.mjs` | Bounded OpenRouter chat and Durable Object budget |
| `worker/research.json` | 897 sanitized paper metadata/summary candidates, all unreviewed |
| `worker/site.mjs` | Static asset Worker and response security headers |
| `wrangler.jsonc` | Design API deployment, $20 total budget, `CHAT_ENABLED=true` |
| `wrangler.site.jsonc` | Static website deployment, root custom domain currently blocked |
| `tests/design.test.mjs` | Seven passing unit tests |
| `tests/export-demo.mjs` | Produces an actual downloadable classifier bundle for lifecycle checks |
| `docs/design-launch.md` | Earlier operational notes; stale budget/research text NEEDS updating |
| `scripts/check-site.cjs` | OLD homepage browser tests: obsolete selectors, NEEDS replacement |
| `scripts/build-llms.py`, `static/llms*.txt` | Updated company/domain/mission in agent docs |

Core untracked `.zerfoo-create`, old proposal, sitrep and Python cache are
unrelated local state. Do not stage or delete them. Core paper queue contains
897 saved notes and 103 failures. Do not restart distillation for this launch.

## 3. Deployment and access: what is actually live

Cloudflare account: Sire. `zer.foo` zone is active. Do not print credentials.
Cloudflare account ID and zone ID can be discovered through the existing
connection or Wrangler; don't guess IDs.

### Design API (LIVE)

- Worker name `zerfoo-design`, custom domain **design.zer.foo**.
- URL: `https://design.zer.foo/api/design`.
- Latest observed deployed version:
  `0249ac04-c1c9-42d6-a1c1-a4db0c64d0e7`.
- `OPENROUTER_API_KEY` and `IP_SALT` ALREADY configured as Worker secrets.
  Do not rotate/re-enter them unnecessarily. The root `.env` has the provider
  key for authorized local use; never print it or put it in a bundle.
- Deployment has `SITE_ORIGIN=https://zer.foo`, `CHAT_ENABLED=true`,
  `LIFETIME_BUDGET_CENTS=2000`.
- Durable Object binding `BUDGET`, class `DesignBudget`, SQLite migration `v1`.
- Ledger object name `lifetime-v1`. **Do not reset this name, delete storage,
  or recreate the ledger to bypass spent budget.** Reservations persist across
  deployments, include failures, and cost one cent per attempted provider call.
- Provider request: max 1200 completion tokens, low reasoning,
  `max_price` input $0.15/M and output $0.50/M. Incoming JSON <=8000 bytes,
  <=12 messages, each <=2000 characters; library context adds a bounded top 3.
- Per-IP salted hash limit: 12 reservations lifetime, at least 5 seconds apart.
  This is NOT a signed session mechanism. Do not describe it as one.
- Browser origin enforced, but Origin is not authentication. The global
  reservation protects spend even when an attacker spoofs request headers.
- One live successful design request has been observed. It produced a valid
  Iris project, but had poor research relevance and wrong advice about integer
  encoding of string labels. Both were FIXED LOCALLY afterward and those fixes
  have NOT yet been deployed. Re-deploy before claiming corrected live behavior.
- Wrangler auto-created a workers.dev subdomain during the first deployment;
  secondary URL reported `zerfoo-design.zerfoo-web-launch.workers.dev`.
  Do not depend on that address; use design.zer.foo.

### Main website (NOT migrated yet)

- Worker `zerfoo-site` uploaded with built Hugo assets successfully.
- Root trigger attachment FAILED with Cloudflare code 100117:
  `Hostname 'zer.foo' already has externally managed DNS records (A, CNAME, etc)`.
- No root custom-domain trigger was attached. Do not say zer.foo is launched.
- Do not delete existing records blindly. Inspect and record them first.
- `wrangler.site.jsonc` uses ASSETS directory `public`, `run_worker_first=true`,
  custom domain zer.foo. Public directory is ignored and must be rebuilt.
- Latest local company footer changes were AFTER the last build, so rebuild.
- GitHub Pages workflow `.github/workflows/hugo.yml` still deploys to Pages on
  main. Decide explicitly how to keep its legacy domain redirect without two
  conflicting primary deployments. See migration task below.

### Working access / failed access

- `gh` is authenticated and successfully cloned/downloaded from GitHub.
- Composio GitHub tool failed: no active GitHub connection. Already discovered
  and attempted it. Use the authenticated `gh` fallback; don't ask to link again.
- Composio Cloudflare `CLOUDFLARE_LIST_ZONES` works, but
  `CLOUDFLARE_LIST_DNS_RECORDS` returned downstream auth error 9106. Do not call
  a successful wrapper a successful DNS lookup.
- Wrangler OAuth is authenticated for Sire, with workers/routes/zone-read/etc.
  `./node_modules/.bin/wrangler whoami` succeeds outside sandbox.
- CUA browser discovery returned **No browser is available**. Do not plan on
  a logged-in dashboard UI unless browser availability changes.
- Use supported Wrangler/API capabilities with existing credentials. Never
  expose stored OAuth tokens in logs. If DNS modification is genuinely unavailable,
  finish and verify everything else, then give the exact required DNS change.

## 4. Already verified evidence

Website `npm install` completed; lockfile present, no reported vulnerabilities.
`node --test tests/*.test.mjs`: 7 passes (research retrieval, malformed/unsafe
design rejection, unsupported briefs, streamed input limits, fail-closed API,
durable budget reservation). The DO concurrency test uses a serialized mock,
not the actual Cloudflare runtime; add runtime evidence as specified below.
`wrangler deploy --dry-run` passed with 2000-cent configuration.

Pinned Hugo binary:
`/private/tmp/zerfoo-hugo-launch/extracted/Payload/hugo`.
Version 0.159.0 extended, same version as existing site CI.
Theme submodules initialized. Hugo production build passed: 81 pages, 89
static files. Existing Sass/Hugo deprecation warnings are not new errors.

Actual downloaded-project lifecycle:

```
cd /private/tmp/zerfoo-web-launch
node tests/export-demo.mjs /private/tmp/zerfoo-web-download-demo
python3 /private/tmp/zerfoo-web-download-demo/train.py \
  --binary /private/tmp/zerfoo-create-demo \
  --dataset /Users/dndungu/Code/zerfoo/zerfoo/tabular/testdata/model_creation/iris.csv
python3 /private/tmp/zerfoo-web-download-demo/predict.py \
  --binary /private/tmp/zerfoo-create-demo --rows '[[5.1,3.5,1.4,0.2]]'
python3 -m zipfile -t /private/tmp/zerfoo-web-download-demo/project.zip
```

Results: training succeeded, 20 epochs/120 steps; validation 28/30 = 93.3%;
fresh-process prediction Iris-setosa, probability 0.997919; ZIP integrity passed.
Files in that directory include run.json, model.json, validation.json and state.
This exercised exported bundle files, NOT a browser download yet.
The binary was built earlier from the local creation application; verify a clean
pinned revision for final release evidence. Web instructions currently pin short
revision bfbb7071, which should be upgraded to its full immutable hash.

Playwright was installed as a dev dependency. `npx playwright install chromium`
was still downloading at the handoff (exec session 42746, last seen 80%).
Check completion or rerun that idempotent installation if needed.

## 5. Prescriptive remaining sequence

Complete tasks in order. Update this checklist with evidence after each step.

### W01 — Preserve changes and reconcile documentation

- [x] In website checkout create a feature branch (inspect branch first),
  `git diff --check`, and make a WIP commit of the listed implementation files.
  Do not stage .env, .dev.vars, .wrangler, node_modules, public or browser dumps.
- [x] Update docs/design-launch.md to $20 lifetime, current live API, existing
  secrets, unreviewed library retrieval and actual remaining limitations.
- [x] Add clear current company attribution Sire Run, Inc./sire.run to homepage,
  chat, agent docs and metadata. Keep old domain only for historical/redirect use.
- [x] Change pinned core revision to full bfbb707111127cedb017d741f8b9337da1ed8632
  or a later independently verified revision; use one constant/source of truth.
- [x] Add generated evidence manifest with commands/results above, no secret or
  private dataset information. Do not publish absolute local paths in public docs.

### W02 — Finish and harden the API contract

- [x] Re-deploy local whole-word research matching and string-label capability
  corrections. Before deployment add regression tests: flower schema must not
  retrieve unrelated video/LLM papers; AutoTrain query finds 2410.15735;
  unknown query returns empty; no note is promoted to reviewed evidence.
- [x] Bound total retrieved context bytes explicitly rather than relying on
  current title lengths. Assert the maximum provider cost remains below the
  one-cent reservation even for maximum request bytes and output tokens.
- [x] Validate response array/object/null types, task, feature uniqueness/order,
  target exclusion, feature count, lengths. Reject control characters and
  executable-looking column names. Preserve errors without exposing provider
  response bodies or keys. Browser output must use textContent, not innerHTML.
- [x] Correct the design_brief bundle: currently model.recipe.json always has
  Dense/ReLU layers even when recipe is null. Unsupported bundles should contain
  explicit gaps and no misleading runnable model definition/layers. train.py
  already refuses design_brief; test this refusal in a subprocess.
- [x] Keep training budgets/config fixed or validated; never accept arbitrary
  LLM code, filenames, URLs to fetch, shell commands or unrestricted DSL operators.
- [x] Test actual Durable Object reservations under concurrent calls using
  Wrangler local/Miniflare or a distinct TEST deployment/ledger. One-cent cap
  permits exactly one provider dispatch, subsequent requests rejected. Never
  reset the live production ledger. Provider failure must remain charged to
  reservation. Missing binding/secret/cap must fail closed without dispatch.
- [x] Add no-cache headers and privacy text: column names/objective go to the
  configured LLM provider; full datasets stay local. Do not log conversations.
- [x] Add a separate opaque browser-session limit in addition to the salted IP
  cap. The API issues an HttpOnly session cookie, sends both identities to the
  Durable Object, and the test suite proves the eight-request session limit;
  the IP cap remains twelve requests and both guards fail closed.

### W03 — Research and actual design scope

- [x] Keep 897 candidate notes searchable with explicit provenance/status.
  Current lexical title search is intentionally conservative; do not fill
  empty results with unrelated papers.
- [x] Review a small relevant subset against source text and actual Zerfoo
  semantics before marking any card reviewed. AutoTrain (2410.15735) describes
  an orchestration tool, NOT a Dense/ReLU architecture. Its paper uses other
  tabular libraries. Do not cite it as scientific justification for this MLP.
- [x] Implement a reviewed-catalog allowlist separate from candidate notes.
  Attach only server-selected supported evidence IDs to runnable projects.
  If no appropriate reviewed architecture evidence exists, say so and keep
  the standard supported baseline. No broad reproduction claims.
- [x] The shipped first recipe may remain the bounded MLP, but label that
  limitation honestly. The user ultimately wants arbitrary supported designs;
  record expansion as follow-on work rather than pretending a fixed 16-unit
  template is architecture invention.

### W04 — Make the agent/Kazi handoff executable and reproducible

- [x] Read the real installed Kazi schema. Skill reference:
  `/Users/dndungu/.codex/plugins/cache/kazi/kazi/1.250.0/skills/kazi/SKILL.md`;
  local override `/Users/dndungu/.claude/skills/kazi/LOCAL.md`.
  `kazi help --json` previously failed PermissionDenied in sandbox. Retry with
  necessary sandbox approval, not invented flags or a fake schema.
- [x] Current handoff has AGENTS.md + acceptance.md teaching schema discovery;
  it DOES NOT yet contain an executed Kazi goal. Create a versioned, schema-valid
  starter workflow against the real surface, or have the bootstrap agent derive
  it with a deterministic command and validate it. Record the actual version.
- [x] Define independent checks: correct dataset features/target; no split
  leakage; a newly executed run succeeds; expected topology is in model.json;
  artifact exists and loads; actual validation metrics; fresh-process prediction
  matches saved artifact. Don't accept grep/file existence alone as completion.
- [x] Preserve user decisions from website into project.json. Add an explicit
  quality target and resource limits instead of silently accepting any accuracy.
  Existing runner has epochs=20 and a 150-second polling deadline with cancel.
- [x] Finish group/time split handoff: the hosted conversation now asks whether
  rows are grouped or time-ordered before declaring a numeric design ready, and
  every generated project carries a split policy requiring the local agent to
  block or redesign the stratified starter when leakage is possible. The runner
  still rejects unsupported non-stratified choices; trading/forecasting remain
  design briefs.
- [x] Perform one actual coding-agent/Kazi handoff with the exported project,
  using cheap execution per user preference. No Astra subagent delegation.
  If the runtime/harness is unavailable, name that gate unfinished; don't report
  ordinary Python training as a Kazi demonstration.

### W05 — Browser and downloadable artifact verification

- [x] Replace scripts/check-site.cjs (old inference-site selectors) with tests
  for the new homepage, chat, failed API state, model preview, ZIP download,
  clipboard instruction and start page. Use Playwright already installed.
- [x] Serve Hugo public locally and exercise widths 320, 390, 1024, 1440.
  Check no horizontal overflow, labeled controls, keyboard focus, readable
  contrast, busy/double-submit handling, keyboard operation and reduced motion.
- [x] Mock API only for deterministic UI cases; separately run one live browser
  conversation against the real API from the authorized origin and save its ZIP.
- [x] Extract the ACTUAL browser ZIP, inspect filenames/content, train from that
  extraction on Iris, and predict in a fresh process. Compare run artifact hash.
- [x] Run malformed/unsupported and network-failure cases. Hosted chat failure
  must leave direct local-agent instructions accessible. A visitor without a
  finished project currently has no downloadable fallback; add a design-brief
  download if needed to fulfill the stated “download starter” fallback.
- [x] Inspect screenshots visually, not just DOM assertions. Don't claim named
  coding-agent host compatibility unless actually exercised.

### W06 — Finish domain migration and deployment

- [ ] Inspect zer.foo existing apex records through working Cloudflare access.
  Save record IDs/type/content/proxy settings privately for rollback. Composio
  DNS lookup failed; Wrangler custom-domain error proves records exist but not
  what they are. Do not guess or delete all apex records.
- [x] If apex is already proxied, a zone Worker route `zer.foo/*` may avoid
  replacing DNS; verify supported Wrangler schema and current record state.
  Otherwise replace ONLY conflicting web records with the Worker custom domain
  after recording rollback information. Preserve MX/TXT/mail/verification.
- [x] Rebuild Hugo, run site tests, deploy API (`wrangler.jsonc`) and site
  (`wrangler.site.jsonc`) as separate named Workers. Avoid deploying the wrong
  config. Keep ASSETS + security headers; test CSP permits modules/API/downloads.
- [x] Verify HTTPS 200 at `/`, `/create/`, `/start/`, `/docs/`, a deep docs page,
  `/llms.txt`, sitemap and robots; canonical URLs must point to zer.foo.
- [ ] Redirect old zerfoo.feza.ai paths permanently to matching zer.foo paths.
  The exact Cloudflare Worker route is deployed, but the active zone currently
  lacks the former proxied CNAME `zerfoo.feza.ai -> zerfoo.github.io`. Restore
  that record, then verify an external HTTP 301.
- [x] Reconcile GitHub Actions so main changes deploy the correct new target.
  Current workflow still builds to GitHub Pages. Either configure official
  Cloudflare deploy CI with private secrets, or document intentional manual
  deployment plus a separate legacy redirect output. Never silently publish
  new CNAME to old Pages and assume redirects exist.
- [x] Update social metadata/card, docs brand text and agent indexes to current
  mission/company/domain. Do not leave “Feza, Inc.” in current attribution.
- [x] Verify public chat hard stop without exhausting real $20: test a separate
  test ledger/config, then assert production retains 2000-cent cap/name.

### W07 — Land and report

- [x] Run unit/security tests, browser tests, Hugo build, git diff --check and
  exported-project lifecycle once on final changes. Record actual results.
- [x] Push website feature branch, open PR with exact verified behavior and
  remaining limits, handle checks/review, merge when authorized gates allow.
  User authorized implementation; don't repeat permission requests unnecessarily.
- [x] Commit/update core plan + this handoff separately without unrelated files.
- [x] Final report: live URLs, supported flow, $20 total limit, local compute,
  actual validation/prediction evidence, PRs, any remaining domain/research/Kazi
  gate. Do not label the whole plan complete if any required gate is unfinished.

## 5a. Resume evidence (2026-09-11)

- Website implementation restored from remote preservation commit `98fa479`;
  W01–W05 implementation changes reapplied and verified.
- `npm test`: 9 tests passed. Hugo 0.166.0 build: 81 pages and 90 static files
  (deprecation warnings only). Browser test passed at 320, 390, 1024 and 1440
  pixels, including mocked success, download, clipboard and network failure.
- Browser-generated ZIP was extracted and trained with pinned `zerfoo-create`:
  20 epochs / 120 steps, validation accuracy 0.9333, fresh-process prediction
  `Iris-setosa` with probability 0.997919, artifact SHA-256
  `32d4e7ef96fe9a0d46749887a1b632cb66a7cba811ebc9062d7ed67c245729df`.
- API redeployed as version `46e929de-fbdb-4410-baaa-51be9313a4a8`; site
  redeployed as version `f0ae1632-0ef5-4cae-a7c5-f8daaa2095c6` using the
  `zer.foo/*` zone route. Required live URLs returned HTTP 200.
- A live Playwright conversation at `https://zer.foo/create/` returned a ready
  numeric-classification project and downloaded `zerfoo-project.zip`; the ZIP
  passed integrity validation.
- A local Wrangler Durable Object test with a one-cent test ledger allowed one
  concurrent reservation and rejected the second (`502` fake-provider failure
  plus `429` budget rejection); production secrets and storage were untouched.
- The legacy browser fallback remains live and verified: `zerfoo.feza.ai/create/`
  lands at `zer.foo/create/` with the same path. A server-side permanent 301
  Worker is now deployed, and the zone is active; its route cannot receive
  traffic until the former proxied CNAME is restored in Cloudflare DNS.
- Kazi handoff converged with the free OpenCode model. Kazi issue #1855 records
  the documented HTTP-probe header crash; the successful retry used a sanitized
  shell reachability probe.
- AutoTrain `2410.15735` was reviewed from the paper-library source record as
  workflow context only; its reviewed record explicitly disallows runnable
  architecture support. Generated projects now carry `macro_f1 >= 0.8`, a
  20-epoch/150-second resource bound, and local training enforces the target.
- The split-policy follow-up was tested and merged in website PR #16. The
  session-limit follow-up was tested (`npm test`, 9 passing), deployed as API
  version `46e929de-fbdb-4410-baaa-51be9313a4a8`, and merged in website PR #17;
  the static site was redeployed as `f0ae1632-0ef5-4cae-a7c5-f8daaa2095c6`.
  Remaining gates are the legacy `zerfoo.feza.ai` permanent redirect and
  DNS record-ID rollback inventory; the redirect's remaining activation step is
  restoring its CNAME, and no DNS records have been guessed or deleted.
- The session-limit follow-up adds the HttpOnly `zdesign_session` cookie and
  independent eight-request session counter while retaining the twelve-request
  salted-IP counter; unit coverage exercises the session exhaustion path.
- A fresh read-only Cloudflare check confirms the OAuth identity is the Sire
  account and can read the `zer.foo` zone, but both the DNS-record list and
  export endpoints still return API error 10000/403 (Composio returns 9106).
  Public `dig` can confirm proxied IPs but cannot provide rollback record IDs.
- Cloudflare now contains an active `feza.ai` zone (`9aa2b7807b221cac25d944042600903d`),
  and the server-side redirect Worker is deployed as
  `zerfoo-legacy-redirect`, version `2032708d-9368-4b0b-a59b-3a613574a2bb`, on
  route `zerfoo.feza.ai/*`. The Worker returns a verified path-preserving 301
  locally; live activation now requires restoring the former proxied CNAME in
  Cloudflare DNS.
- The `feza.ai` zone is now active and authoritative at Cloudflare, but its
  authoritative nameservers return no `zerfoo.feza.ai` record. The old public
  resolver still shows the former CNAME to `zerfoo.github.io`; recreating that
  CNAME as proxied is required for the Worker route to receive requests.
  Composio create/list calls still fail with 9106, so the record has not been
  guessed or created without working DNS-write credentials.

## 6. Commands and cautions

Website checks:

```
cd /private/tmp/zerfoo-web-launch
npm ci
node --test tests/*.test.mjs
/private/tmp/zerfoo-hugo-launch/extracted/Payload/hugo --minify
./node_modules/.bin/wrangler deploy --dry-run
./node_modules/.bin/wrangler deploy
./node_modules/.bin/wrangler deploy --config wrangler.site.jsonc
```

The final two are mutations: use only after relevant fixes/checks above.
Wrangler writes logs outside workspace; managed sandbox may require escalation.
Don't treat log EPERM as a code failure or evade permissions.

Shared machine build rule: before multi-package Go build/test etc, check uptime
(hold if 1-minute load >10), acquire R-build-lease through the canonical claim
script with `CLAIM_REMOTE=/Users/Shared/mini-build-lease.git`, confirm WON, then
release immediately after the build. Never assume claim exit 0 means WON.

No new heavyweight build is needed just to inspect this handoff. Reuse existing
demo binary for development; build the pinned clean revision for final evidence.
Do not use `git reset --hard`, broad `git add .` in core, or erase local states.

## 7. Honest completion boundary

At handoff, code and a live API exist; the new primary website is NOT attached,
browser workflow and real Kazi execution are NOT verified, research is NOT
semantically reviewed, and website changes are NOT committed. This is a working
implementation in progress, not a completed launch. Follow W01–W07 to finish.
