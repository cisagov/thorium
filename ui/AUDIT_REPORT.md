# Thorium UI Audit Report

Audit of `ui/src` (452 TS/TSX/SCSS files) against the conventions in `ui/CLAUDE.md`, the backend API surface in `api/src/routes/`, and general React/security best practices. Each finding lists the evidence, why it matters, and a recommended fix. Priorities: **P1** (fix soon), **P2** (plan into upcoming work), **P3** (opportunistic).

---

## 1. Theme & Styling

**Overall:** The token system is genuinely good — `src/styles/colors.scss` defines ~175 CSS custom properties with 4 themes (`Dark`, `Light`, `Ocean`, `Crab`) switched via a `[theme]` attribute on `#root` (`utilities/auth.tsx:70-85`). The problems are components bypassing it and the half-finished react-bootstrap → styled-components migration.

### T1. Hardcoded color literals bypass the theme system — P1
168 hex color literals across ~30 `.ts`/`.tsx` files. Hotspots:

| File | Count |
|------|-------|
| `dashboards/IncidentSummary/styles.ts` | 21 |
| `components/pages/images/Fields.tsx` | 20 |
| `components/associations/graph/styles.ts` | 16 |
| `components/tools/displays/JSON.tsx` | 15 |
| `components/shared/alerts/AlertBanner.tsx` | 13 |

Examples: `Fields.tsx:324` `<FieldBadge field={creator} color="#305ef2" />`, `:334` `color="#6a00db"`, `:145` `background: #fff;`. `LinkBadge.tsx:22` `background-color: gray`. These render identically in all four themes and will clash in Light/Ocean/Crab.

**Fix:** Add semantic tokens for badge/accent colors (e.g. `--thorium-badge-creator`, `--thorium-badge-group`) to `colors.scss` per theme and reference them via `var(...)`. The graph node palette (`graph/styles.ts:54-70`) is data-visualization color coding and may legitimately stay fixed — but should still be centralized as named exports and given enough contrast in light themes.

### T2. Dark-biased `var()` fallbacks duplicated everywhere — P2
The pattern `var(--thorium-secondary-panel-bg, #2a2d35)` (e.g. `dashboards/IncidentSummary/styles.ts:33-35,51,68-69,91-92`, `components/pages/images/Fields.tsx:125`) hardcodes a **dark** fallback. If a token is ever renamed/missing, Light-theme users get dark fragments; and the fallback values drift between files.

**Fix:** Tokens are always defined on `:root`/`[theme]`, so drop the fallbacks entirely (`var(--thorium-secondary-panel-bg)`). Add a lint rule or grep check forbidding hex fallbacks in `var()`.

### T3. "Automatic" theme doesn't track OS changes — P3
`auth.tsx:75-80` evaluates `window.matchMedia('(prefers-color-scheme: dark)')` once when `userInfo` changes. If the OS flips theme mid-session, the UI doesn't follow. There's also no theme applied before login/userInfo load (flash of default theme).

**Fix:** Register a `matchMedia(...).addEventListener('change', ...)` listener when theme is `Automatic` (cleanup on effect teardown); apply a stored/last-known theme attribute in `index.html` or `main.tsx` before React mounts.

### T4. 1,400 lines of legacy SCSS + 113 `!important` — P2
`styles/legacy/` holds 1,397 lines (buttons 246, reactions 207, details 203, upload 184, …) and `styles/` contains **113 `!important` declarations**, mostly fighting Bootstrap specificity. This directly conflicts with the styled-components-only direction in `ui/CLAUDE.md` and makes new styled-components unpredictable when class selectors win.

**Fix:** Treat each `legacy/*.scss` file as a migration unit: when its consuming component is converted to styled-components, delete the file. Track remaining files in the migration checklist; forbid additions via review.

### T5. Migration is ~half done and mixed idioms coexist — P2
110 files still import `react-bootstrap` vs 95 files using `styled-components`. Many components mix three styling systems in one render — e.g. `TagBadge.tsx` uses Bootstrap utility classes (`ms-1 mb-1 d-flex`), SCSS classes (`tag-item`, `clickable`), and sibling components use styled-components. 41 inline `style={{...}}` blocks across 28 files (CLAUDE.md allows inline style only for 1-2 props).

**Fix:** Maintain a burn-down list of the 110 react-bootstrap importers; convert opportunistically per the "replace when touched" rule, prioritizing `Button`, `Modal`, `Row/Col` since those have the most usages and already have shared replacements planned in `components/shared/`.

---

## 2. Code Quality & Patterns

**Overall:** TypeScript hygiene is strong — only 11 `any` usages, 2 `@ts-ignore`, 6 `eslint-disable` in the whole tree. The main issues are error-handling inconsistency in the API layer, a few god components, and copy-paste duplication.

### Q1. thorpi error contract is violated — errors silently swallowed — P1
`ui/CLAUDE.md` states every thorpi function takes an `errorHandler`. But:
- Four functions hardwire `console.log` as the handler: `getImage` (`thorpi/images.ts:74`), `authUserToken` (`thorpi/users.ts:58`), `getUser` (`users.ts:115`), `whoami` (`users.ts:236`).
- ~10 call sites pass `console.log`/`console.error` *as* the errorHandler, so real failures never reach the user: `utilities/fetch.ts:102`, `pages/users/Groups.tsx:96,106`, `pages/users/UserProfile.tsx:178` (a settings **save** that can fail silently), `components/entities/browsing/configs/RepoBrowsingConfig.tsx:20`, `FileBrowsingConfig.tsx:24`, `components/entities/utilities.ts:29`, `utilities/tags.ts:73`, `components/pages/files/upload/associations.ts:29`, `components/shared/pipeline/PipelineOrderFlow.tsx:290`.

**Fix:** Restore the `errorHandler` parameter on the four functions; introduce a small global toast/alert context (or reuse `AlertBanner` via a provider) so call sites without local error UI have a real default handler instead of `console.log`.

### Q2. God components — P2
Top offenders: `pages/users/Groups.tsx` (1,214 lines), `components/associations/graph/AssociationGraph.tsx` (825), `components/pages/images/Dependencies.tsx` (798), `Fields.tsx` (781), `components/tags/EditableTags.tsx` (743), `override_pages/FileDetails.tsx` (673), `OutputCollection.tsx` (597). Also `CodeEditor/SuggestionPreview.ts` at 3,812 lines (verify how much is data tables vs logic — data belongs in JSON/generated modules).

**Fix:** Split along existing visual sections (Groups: member table / role editor / create modal; Fields: one file per badge family). The entity-config pattern already in the codebase is the model to follow.

### Q3. Copy-paste duplication — P2
- `TagBadge.tsx:59-140`: three near-identical `Modal` + badge blocks (ATT&CK, MBC, generic) differing only in URL construction. Extract an `ExternalLinkConfirmModal` + one badge renderer (LinkBadge already implements the same modal a fourth time).
- Duplicate `getCookie` implementations with *different* semantics: `thorpi/client.ts:20-35` (returns `''`) vs `utilities/auth.tsx:25-36` (returns `undefined`). Extract to one utility.
- `client.ts:63-102`: identical request interceptor registered twice for `client` and `bigIntClient`. Extract `attachAuthInterceptor(instance)`.
- Tag truncation `tagText.length > 30 ? tagText.substring(0, 30) + '...' : tagText` repeated 6× in `TagBadge.tsx` alone — make it CSS (`text-overflow: ellipsis`) or a helper.

### Q4. Unencoded path parameters in every thorpi URL — P1
`encodeURIComponent` appears **zero** times in `src/thorpi/`. URLs are built by concatenation, e.g. `thorpi/images.ts:64` `'/images/data/' + group + '/' + image`. Any resource name containing `/`, `?`, `#`, `%` or spaces produces a wrong request (and tag keys/values go into query params unencoded via object params — axios handles those, but path segments are raw).

**Fix:** `encodeURIComponent()` every interpolated path segment, ideally via a tiny `apiPath('images','data',group,image)` helper used across all modules.

### Q5. Dead/debug code — P3
- Debug logs: `components/entities/create/EntityCreateRoutes.tsx:40-41` (`console.log(entity)`); broken template literal in `DeviceDetailsConfig.tsx:63` (`'Attempting to create a new vendor: ${}'`).
- `EntityList.tsx:60-76`: the `isMountingRef` two-effect dance is dead logic — the first effect always sets the ref `true` before the second runs, so the `else` branch is unreachable and the comment doesn't match behavior.
- `Omnibar.tsx:273`: `//WARN: might have unintended consequences. test` — resolve and remove.
- Non-null assertions in omnibar state transitions (`Omnibar.tsx:112,150`) where the draft may genuinely lack the field if state transitions are reordered.

### Q6. Test pages mounted in the production router — P2 (also Security S4)
`Thorium.tsx:94-99` registers `/test/sigma`, `/test/yara`, `/test/alerts`, `/test/image-pipeline`, `/test/overlay-window`, `/test/buttons` unconditionally — they ship in the prod bundle and `/test/alerts` deliberately throws to exercise the error boundary.

**Fix:** Wrap registration in `if (import.meta.env.DEV)` (Vite tree-shakes the lazy imports out of prod builds).

---

## 3. UI Bugs & Robustness

### B1. No global 401 handling — expired sessions degrade messily — P1
`thorpi/client.ts` has request interceptors only; there is **no response interceptor**. When the token expires mid-session, every page renders scattered "Permission Denied" alerts (per `parseRequestError`, `client.ts:122`) and the user is never redirected to login — `RequireAuth` (`auth.tsx:202`) only re-checks the cookie on re-render.

**Fix:** Add a response interceptor on both axios instances: on 401, clear the cookie/auth state and redirect to `/auth` with the current path in state (the Login page already honors `state.path`, `pages/Login.tsx:55`).

### B2. Malformed cookie attribute in `buildCookie` — P2
`auth.tsx:40`:
```ts
return `THORIUM_TOKEN=${token}; Secure; SameSite=Strict; expires=${expiration}; path=/; domain: ${location.hostname}`;
```
`domain: x` (colon) is not a valid cookie attribute — browsers ignore it, and the cookie silently falls back to the default host-only scope. Currently benign, but it's a latent bug if subdomain scoping is ever expected. The revoke cookie (`auth.tsx:38`) also omits `SameSite`.

**Fix:** Use `domain=${...}` only if cross-subdomain is actually needed (host-only is more secure — probably just delete it), and mirror `SameSite=Strict` on the revoke string.

### B3. `EntityList` drops fetches and can interleave stale results — P1
`components/entities/browsing/EntityList.tsx`:
- Line 70: `if (filters != null && Object.keys(filters).length > 0 && !loading)` — a filter change that arrives **while a fetch is in flight is silently dropped**, leaving results desynced from the active filters (user types fast in BrowsingFilters/omnibar, list never updates).
- No `AbortController`/sequence guard: out-of-order responses can clobber newer results.
- Line 93: index-as-key (`key={`${type}_entity_${idx}`}`) across appended pages.

**Fix:** Remove the `!loading` guard and instead use a request-sequence ref (or `AbortController`) so the latest filters always win; key rows by entity identity (sha256/name/id from `displayEntity`'s entity).

### B4. Pagination can land on a blank page — P3
`EntityList.tsx:78-83`: clicking Next on the last cached page triggers a fetch *and* immediately sets the new page. If the cursor returns zero new rows, the user sits on an empty page (the no-results banner is suppressed because `entities.length > 0`).

**Fix:** Only advance the page after the fetch resolves with rows; null out the cursor when a page returns empty.

### B5. Blob URLs never revoked; downloads fully buffered — P2
8 `URL.createObjectURL` call sites (`components/pages/files/Download.tsx:22`, `Comments.tsx:45`, `tools/displays/files/ResultsFiles.tsx:18`, `tools/displays/Image.tsx:39`, `EntityDetails.tsx:400`, `EntityGraphicUpload.tsx:131`, `thorpi/entities.ts:125`) and only a doc comment mentions `revokeObjectURL` — zero actual calls. Each download/preview leaks the blob for the tab's lifetime. `Download.tsx` also buffers the entire sample in memory before saving — multi-GB samples can crash the tab.

**Fix:** `URL.revokeObjectURL(url)` after `link.click()` (a `setTimeout(0)` is enough); revoke preview URLs in effect cleanup. For downloads, consider streaming via `<a href={apiUrl}>` with an auth-bearing redirect/once-token, or `response.blob()` with `showSaveFilePicker` where supported.

### B6. `EntityDetails.tsx:400` creates an object URL during render — P2
`const headerImageUrl = graphicFile ? URL.createObjectURL(graphicFile) : existingImageUrl;` — a new blob URL per render (leak + churn). Move into `useMemo` with cleanup, or state set when the file changes.

### B7. Positives worth keeping
- Top-level `ErrorBoundary` in `Thorium.tsx:132` plus a per-tool boundary in `ToolResult.tsx:147` — tool render crashes don't take down the page.
- `AssociationGraph.tsx:691-701` and `OverlayWindow.tsx` clean up all listeners correctly; upload flow uses a real `AbortController` (`useFileUpload.ts:141`).
- Search omnibar input is debounced with proper timeout cleanup (`Search.tsx:205-223`).
- Catch-all 404 route exists (`Thorium.tsx:102`).

---

## 4. Security

Context: this is a **malware analysis platform** — tool output and tags are attacker-influenced data, and samples are hostile by definition.

### S1. Auth token is a JavaScript-readable cookie — High (architectural)
The token lives in a non-HttpOnly cookie written by `document.cookie` (`auth.tsx:39-41`) and read back by JS in both `auth.tsx` and `thorpi/client.ts:72` to build the `Authorization` header. Any single XSS anywhere in the app yields full token theft. Mitigations already present: react-markdown without raw HTML, DOMPurify on tool HTML (`components/tools/SafeHtml.tsx:29`), `SameSite=Strict; Secure`, and the Authorization-header pattern largely neutralizes CSRF.

**Fix (longer-term):** Have the API set an HttpOnly session cookie (it already serves the SPA, same origin) and accept it as an auth source, keeping the header flow for CLI/API clients. Short-term: keep XSS surface minimal (S2/S3) and consider a short token TTL + rotation on `whoami`.

### S2. `window.open` on tool-supplied URLs without scheme check or `noopener` — High
`components/shared/badges/LinkBadge.tsx:32` runs `window.open(url, '_blank')` where `url` comes from tool/user data (tags, results). A `javascript:` or `data:` URL passes the confirmation modal (users click Confirm) and executes with no scheme filter; all `window.open` calls (`LinkBadge.tsx:32,37`, `TagBadge.tsx:57,106`) omit `'noopener,noreferrer'`, allowing reverse tabnabbing.

**Fix:** Validate `new URL(url).protocol` ∈ {`http:`, `https:`} before enabling the badge; always pass `'noopener,noreferrer'` as the third argument.

### S3. SVG injected via `dangerouslySetInnerHTML` in `SigmaIcon` — needs verification
`components/shared/icons/SigmaIcon.tsx:15` injects `svg` markup. If the SVG is a static bundled asset this is fine; confirm it can never be sourced from API data. `SafeHtml.tsx` correctly sanitizes with DOMPurify — extend it with a DOMPurify hook forcing `rel="noopener noreferrer" target="_blank"` on anchors inside sanitized tool HTML.

### S4. Test routes shipped to production — Medium
See Q6. `/test/alerts` intentionally throws; all test pages are reachable by any authenticated user in prod. Gate behind `import.meta.env.DEV`.

### S5. Defense-in-depth notes — Low
- `RequireAdmin` (`auth.tsx:212-225`) is client-side only — fine as long as every admin route handler enforces server-side (spot-check `system.rs` handlers).
- `register()` defaults email to `thorium@sandia.gov` (`auth.tsx:144`) — silently attributes all self-registrations to one mailbox; make email required or empty.
- Positives: `sourcemap: false` in `vite.config.ts:59`; no `eval`/`new Function`; downloads are always CaRT/encrypted-zip wrapped (`Download.tsx`) so raw malware bytes never hit the browser as a navigable file; markdown rendering has no `rehype-raw`.

---

## 5. Feature Gaps vs Backend API

Method: diffed `api/src/routes/*` route tables against `ui/src/thorpi/*` exports and page usage. No reverse drift found (every UI call has a backend route).

### G1. System administration is read-only in the UI — High value
Backend (`api/src/routes/system.rs:770-781`): `PATCH /system/settings`, `PATCH /system/settings/reset`, `POST /system/settings/scan` (consistency scan), `POST /system/cleanup`, `POST /system/cache/reset`, `GET /system/backup`, `POST /system/restore`, `GET /system/nodes/` + `/details/`.
UI: `thorpi/system.ts` has only `getSystemStats`/`getSystemSettings`; `pages/system/SystemSettings.tsx` renders but cannot edit.
**Work:** settings edit form with confirm-on-dangerous-change, nodes table page, admin actions panel (scan/cleanup/cache-reset). All admin-gated via existing `RequireAdmin`.

### G2. Network policies have zero UI — High value
Backend full CRUD + defaults (`network_policies.rs:329-336`: create, list, details, get-by-name, per-group defaults). No thorpi module, no pages — yet images reference network policies for sandboxing tools.
**Work:** new `thorpi/networkPolicies.ts`, a browsing/details page (the config-driven entity system fits perfectly), and a policy selector in the image create/edit form.

### G3. Repos are list-only in the UI — High value
Backend (`repos.rs:617-634`): create, upload, download, tags, commitishes (`/repos/commitishes/...`, `/commitish-details/`), results, result-files. UI `thorpi/repos.ts` exports only `listRepos`; `RepoDetails.tsx` is a thin override page.
**Work:** repo details tabs for commits/results/download mirroring the file details layout.

### G4. Image/pipeline notifications invisible — Medium
Backend: `GET/POST/DELETE /images/notifications/{group}/{image}[/{id}]` and the pipeline equivalents (`images.rs:435-439`, `pipelines.rs:404-408`) carry build/runtime error notifications for tools. No thorpi functions or UI — users can't see why their tool is failing to schedule.
**Work:** notifications badge + panel on image/pipeline details pages.

### G5. Reaction operations are minimal — Medium
Backend extras with no UI: bulk create (`/reactions/bulk/`, `/bulk/by/user/`), control commands (`POST /reactions/handle/{group}/{id}/{cmd}` — e.g. cancel), per-status listings with details, sub-reaction trees (`/reactions/sub/...`), ephemeral files (`/reactions/ephemeral/{group}/{id}/{name}`). UI has create/get/logs/list/delete only — notably **no cancel button** on `ReactionStatus.tsx` and no sub-reaction or ephemeral-file display.
**Work:** cancel/restart actions on the reaction page; sub-reaction tree section; bulk re-run from file details.

### G6. User/group admin gaps — Medium
Backend: force logout (`GET /users/logout/{target}`), LDAP sync for users and groups (`users.rs:502`, `groups.rs:311`), email verification resend, `GET /groups/{group}/stats`. None exposed; `pages/users/Groups.tsx` and `UserBrowsing.tsx` cover membership/roles only.
**Work:** admin actions on the user row (force logout, resend verification), LDAP sync buttons, group stats panel.

### G7. Smaller gaps — Low
- `GET /files/count/` and `POST /files/exists` unused (count could power list headers; exists could pre-check uploads).
- `DELETE /files/comment/{sha256}/{id}` — comments can be created and attachments downloaded (`thorpi/comments.ts`) but never deleted in the UI.
- Events admin endpoints (`events.rs:169-172`: pop/clear/reset/cache status) — internal, but a cache-status readout would help admins debug.
- MCP server (`mcp.rs`) — consider a status/info card under system pages.

---

## Suggested priority order

1. **P1 batch (correctness/safety):** B1 global 401 interceptor · S2 URL scheme allowlist + `noopener` · Q1 error-handler contract · Q4 path encoding · B3 EntityList race/dropped fetch.
2. **P2 batch (debt that compounds):** Q6/S4 gate test routes · B2 cookie string fix · B5/B6 blob URL lifecycle · T1/T2 color token sweep · Q3 dedupe modal/getCookie/interceptor · T4 legacy SCSS burn-down.
3. **Feature work:** G1 system admin → G2 network policies → G3 repos → G4 notifications → G5 reaction controls, in that order of user value.
