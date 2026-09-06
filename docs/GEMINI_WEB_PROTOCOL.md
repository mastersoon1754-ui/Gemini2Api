# Gemini Web Protocol — Cartographie

> **Source :** trafic observé via `gemini_web2api` (commit `4ba45eb`) + scraping `gemini.google.com/app` + logs `wrb.fr`.  
> **Principe :** ne pas supposer un mécanisme natif s'il n'a pas été observé. Chaque affirmation porte un niveau `CONFIRMED` / `PROBABLE` / `UNKNOWN`.  
> **Backend étudié :** uniquement Gemini Web (`gemini.google.com`), pas `generativelanguage.googleapis.com`.

Niveaux :
- `CONFIRMED` : observé en trafic réel ou reproduit en `generate()` / `generate_stream()` avec succès.
- `PROBABLE` : déduit du JS frontend (`028-6eb337387583.js`, `WIZ_global_data`) ou du comportement, non rejoué isolément.
- `UNKNOWN` : pas d'évidence, hypothèse non vérifiée.

---

## 1. Request

### 1.1 Endpoint

```
POST https://gemini.google.com[/u/{auth_user}]/_/BardChatUi/data/assistant.lamda.BardFrontendService/StreamGenerate?bl={bl}&hl=en&_reqid={reqid}&rt=c
```
- `CONFIRMED` : seul endpoint utilisé pour la génération texte. Aucun `CreateConversation`, `GenerateContent`, `v1beta` côté web.
- `bl` : `CONFIRMED` — token build `boq_assistant-bard-web-server_YYYYMMDD.xx_p0`, scrapé dans `gemini_web2api/gemini.py:130` via regex `boq_assistant-bard-web-server_\d+\.\d+_p\d+` sur `GET https://gemini.google.com/app` (HTML contient `WIZ_global_data`). Si 405, le client rafraîchit `bl` et retry (`gemini.py:156`).
- `_reqid` : `CONFIRMED` — `int(time.time()) % 1_000_000`, random à chaque tentative (le navigateur l'incrémente séquentiellement, ici on approxime).
- `rt=c` : `CONFIRMED` — param fixe, probablement `responseType=chunked`.
- `hl=en` : `PROBABLE` — langue UI, n'affecte pas la génération.
- `/u/{auth_user}` : `CONFIRMED` — préfixe compte multi-compte Google, vide pour défaut, sinon `/{n}` + header `X-Goog-AuthUser` (`gemini.py:88`).

Autre endpoint (images uniquement) :
```
POST https://content-push.googleapis.com/upload/  (init)
POST {X-Goog-Upload-URL}                          (upload)
GET  https://gemini.google.com/app              (page tokens)
```
Voir §1.6.

### 1.2 Headers

| Header | Valeur | Statut | Notes |
|---|---|---|---|
| `Content-Type` | `application/x-www-form-urlencoded` | `CONFIRMED` | body = `f.req=...&at=...` |
| `Origin` | `https://gemini.google.com` | `CONFIRMED` | requis, sinon 403 |
| `Referer` | `https://gemini.google.com[/u/{n}]/app` | `CONFIRMED` | |
| `X-Same-Domain` | `1` | `CONFIRMED` | flag GWS RPC |
| `User-Agent` | `Mozilla/5.0...` | `CONFIRMED` | |
| `X-Goog-AuthUser` | `{auth_user}` | `CONFIRMED` si multi-compte | |
| `Cookie` | `SID=...; HSID=...; SAPISID=...` | `CONFIRMED` optionnel | vide → anonyme OK pour Flash |
| `Authorization` | `SAPISIDHASH {ts}_{sha1(ts + SAPISID + " https://gemini.google.com")}` | `CONFIRMED` si cookie | `make_sapisidhash` (`gemini.py:82`) |
| `Push-ID` etc. | — | `CONFIRMED` seulement pour upload image | |

Sans `Cookie`/`Authorization`, `StreamGenerate` répond quand même pour les modèles Flash (anonyme). `gemini-3.1-pro` retombe silencieusement sur Flash sans cookie payant (`UNKNOWN` si le backend distingue vraiment le modèle).

### 1.3 Auth / Cookies

- Fichier `cookie_file` (`config.py:cookie_file`) : soit raw `Cookie` header, soit JSON `{cookie, sapisid}`. Chargé avec cache `mtime` (`gemini.py:56`).
- Extraction `SAPISID` → `SAPISIDHASH`. Si absent, pas de `Authorization`.
- `xsrf_token` (`SNlM0e` extrait du DOM `WIZ_global_data` côté navigateur) → champ `at` POST (`gemini.py:222`). `PROBABLE` nécessaire seulement si `auth_user != ""` et requête authentifiée ; en anonyme on l'omet sans erreur.
- **Partage** : global `_cookie_cache` / `_httpx_client` → tous les threads partagent le même compte. Pas d'isolation par utilisateur (`PROBABLE` fuite d'historique si `temporary_chats=False`).

### 1.4 Body `f.req`

`CONFIRMED` structure double-JSON :

```
f.req = JSON.stringify([ null, JSON.stringify(inner) ])
body  = urlencode({ "f.req": f.req, "at": xsrf_token? })
```

Exemple minimal (anonyme, sans image) :

```python
inner = [None]*102
inner[0]  = ["Hello world", 0, None, None, None, None, 0]  # prompt
inner[1]  = ["en"]
inner[2]  = ["", "", "", None, None, None, None, None, None, ""]
inner[6]  = [0]
inner[7]  = 1
inner[10] = 1
inner[11] = 0
inner[17] = [[4]]        # think mode
inner[18] = 0
inner[27] = 1
inner[30] = [4]
inner[41] = [2]          # persistance: [2]=persistant, [1]+inner[45]=1 → temporaire
inner[53] = 0
inner[59] = "uuid4"
inner[61] = []
inner[68] = 1
inner[79] = 1            # MODE_CATEGORY 1=FAST (Flash)
outer = [None, json.dumps(inner)]
f_req = json.dumps(outer)  # → POST
```

### 1.5 Payload `inner` — cartographie des champs

Basé sur `gemini.py:195` et source JS `MODE_CATEGORY` (`models.py:3`).

| Index | Type attendu | Valeur actuelle | Signification | Confiance |
|---|---|---|---|---|
| 0 | `[prompt, 0, None, file_refs, None, None, 0]` | prompt string + refs | **Prompt + images**. `file_refs = [[None,None,"/upload/..."],...]` si images | `CONFIRMED` |
| 1 | `["en"]` | langue | UI lang | `PROBABLE` |
| 2 | `["","","",..., ""]` | placeholder | `PROBABLE` metadata vide |
| 6 | `[0]` | | `UNKNOWN` |
| 7 | `1` | | `UNKNOWN` (peut-être `singleTurn`?) |
| 10 | `1` | | `UNKNOWN` |
| 11 | `0` | | `UNKNOWN` |
| 17 | `[[think]]` | `4` ou `0` | **Thinking depth** : `0`=deepest, `4`=shallow | `CONFIRMED` via `models.py` |
| 18 | `0` | | `UNKNOWN` |
| 27 | `1` | | `UNKNOWN` |
| 30 | `[4]` | | `UNKNOWN` (peut-être `modelFamily`?) |
| 41 | `[1]` ou `[2]` | `[2]` persistant, `[1]` temporaire | **Temporary chats** (`config:temporary_chats`) | `CONFIRMED` |
| 45 | `1` ou `None` | `1` si temporaire | Paire avec 41 | `CONFIRMED` |
| 53 | `0` | | `UNKNOWN` |
| 59 | `uuid4` string | | **Request UUID** (client-side) | `CONFIRMED` |
| 61 | `[]` | | `PROBABLE` `previousConversationId` vide → stateless |
| 68 | `1` | | `UNKNOWN` |
| 79 | `int` 1-6 | `1` | **MODE_CATEGORY** : 1 FAST, 2 THINKING, 3 PRO, 4 AUTO, 5 DYNAMIC_THINK, 6 LITE | `CONFIRMED` |
| 31,80... | `extra_fields` | ex `31:2,80:3` | Variantes Pro enhanced | `PROBABLE` |
| autres 0-101 | `None` | | `UNKNOWN` — non utilisés par le proxy, mais potentiellement `conversationId`, `modelVersion`, `tools` natifs non découverts | `UNKNOWN` |

**Hypothèse tools natifs** : aucun champ `inner[*]` n'a été identifié comme portant des `functionDeclarations` natifs. Le proxy injecte les tools **dans le texte du prompt** (§5), pas dans un champ structuré. Aucune trace réseau d'un champ `tools` côté web n'a été capturée. → `UNKNOWN` si Gemini Web supporte un jour un champ natif, `PROBABLE` qu'il n'existe pas aujourd'hui pour le grand public (les tools internes Google Assistant utilisent un autre endpoint `BardFrontendService` non documenté).

### 1.6 Upload image (Scotty)

`CONFIRMED` via `multimodal.py:88` :

1. `GET https://gemini.google.com/app` avec `Cookie`/`Authorization` → scrape regex :
   - `qKIAYe` → `push_id` (ex `feeds/mcudyrk2a4khkz`) fallback hardcodé
   - `Ylro7b` → `pctx` (`CgcSBWjK7pYx`)
   - `thykhd` → `at` (non utilisé pour upload)
   Cache 600s.
2. `POST https://content-push.googleapis.com/upload/` (start) :
   ```
   Push-ID: {push_id}
   X-Tenant-Id: bard-storage
   X-Client-Pctx: {pctx}
   X-Goog-Upload-Header-Content-Length: {len}
   X-Goog-Upload-Header-Content-Type: {mime}
   X-Goog-Upload-Protocol: resumable
   X-Goog-Upload-Command: start
   Cookie + Authorization
   body: b""
   → 200 + header X-Goog-Upload-URL: https://content-push.googleapis.com/upload/..../...
   ```
3. `POST {upload_url}` (finalize) :
   ```
   X-Goog-Upload-Command: upload, finalize
   X-Goog-Upload-Offset: 0
   Content-Type: application/octet-stream
   body: image_bytes
   → 200 + body: "/upload/bard/.../image_ref"
   ```
   `file_ref` réutilisé dans `inner[0][3]`.

Nécessite `Cookie` valide (`PROBABLE` échec anonyme selon région/IP).

---

## 2. Response

### 2.1 Enveloppe `wrb.fr`

`CONFIRMED` :

- HTTP `200` `chunked`, premier bytes `)]}'` (XSSI prefix) puis lignes `\n`-séparées, chaque ligne = JSON array :
  ```json
  [null, "wrb.fr", null, "<inner_json_str>", null, null]
  // ou
  [null, "wrb.fr", "qKIAYe..."] // metadata ligne
  // ou
  [ "BardErrorInfo", "[116]" ] // erreur
  ```
- Le proxy ne garde que les lignes contenant `"wrb.fr"` et `len>=200` (`gemini.py:249`), puis `arr[0][2]` → `json.loads(inner_str)`.

Exemple ligne texte (tronqué) :
```json
[[null,"wrb.fr",null,"[null,null,null,null,[[null,[[\"Hello\"],null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null]]"]]]
```

### 2.2 Chunks / texte

`CONFIRMED` :

- `inner` parsé → `inner[4]` = liste de `parts`. Chaque `part = [?, [ "texte complet cumulé", ...], ...]`. Le texte est **cumulatif** : chaque chunk contient le texte depuis le début, pas un delta. Le proxy détecte `t[len(prev):]` (`gemini.py:399`).
- `inner[4][i][1][0]` = string cumulée. Plusieurs `parts` peuvent coexister (citations, code blocks), mais le plus long est pris (`extract_response_text` garde `max(len)`).

### 2.3 Métadonnées / fin

`CONFIRMED` terminal :

```json
inner[2] = {"11": ..., "44": ...}  // titre + flag done + token counts ?
```
`_is_terminal_line` (`gemini.py:105`) → si `inner[2]` dict contient `"11"` et `"44"` → fin. Le proxy arrête la lecture et ignore le reste.

Autres lignes `wrb.fr` plus courtes (`len<200` ou `inner[4]==null`) = metadata intermédiaires (modelVersion, safety, etc.) — ignorées actuellement.

### 2.4 Erreurs

`CONFIRMED` :

- Inline `BardErrorInfo [116]` dans le body (même sans `wrb.fr`) → `RuntimeError`.
- HTTP `405` → `bl` périmé → refresh `fetch_latest_bl()` et retry.
- HTTP `400` si `xsrf_token` invalide avec `auth_user` (`PROBABLE`).
- `500/503` → retry `retry_attempts` (3) avec `retry_delay_sec` 2s.
- Pas de code d'erreur JSON structuré côté `wrb.fr` (seulement `BardErrorInfo`).

### 2.5 Structure non utilisée

`UNKNOWN` : `inner[0][2]` (peut-être `conversationId` retourné), `inner[1]` (lang), `inner[6]` etc. n'ont pas été mappés à des champs de réponse. Aucun `usageMetadata` natif n'est parsé (le proxy estime `tokens = len//4`).

---

## 3. Conversation state

| Aspect | Observé | Confiance |
|---|---|---|
| `conversationId` dans requête `inner[61]` | Toujours `[]` vide, jamais réutilisé. Le navigateur envoie un vrai ID (`c_xxx`) pour poursuivre un chat existant | `CONFIRMED` stateless côté proxy, `PROBABLE` natif existe côté web mais non utilisé |
| `inner[41]/45` | Seuls flags touchés (`[2]` vs `[1]+1`) | `CONFIRMED` |
| Reprise conversation | Aucune API `getConversation`/`continueConversation` n'est appelée. Le multi-turn est simulé par réinjection de tout l'historique en texte | `CONFIRMED` simulé |
| Isolation multi-utilisateur | Aucune session par utilisateur, même `Cookie` global, même `uuid` par requête | `PROBABLE` fuite si `temporary_chats=False` (compte voit N chats créés) |
| Titre / metadata terminal | `inner[2]["11"]` contient probablement le titre auto-généré | `PROBABLE` |
| Mémoire côté serveur | `STALL_TIMEOUT` + read-until-terminal, pas de `conversationId` retourné stocké | `CONFIRMED` |

**Conclusion** : le proxy est **stateless**. Un vrai support conversationnel nécessiterait de capturer `conversationId` retourné dans `inner[2]` et de le renvoyer en `inner[61]` (ou autre index) — non implémenté, et la structure exacte reste `UNKNOWN`.

---

## 4. Streaming

`CONFIRMED` :

- Upstream **ne ferme jamais** la connexion (`chunked` sans terminator). Le navigateur laisse la connexion ouverte. Le proxy doit détecter la fin via **terminal line** (`inner[2]["11"]["44"]`) OU **idle 60s** (`STALL_TIMEOUT`).
- Données arrivent par **rafales** (bursts) : 1-3 lignes `wrb.fr` puis pause jusqu'à 60s (temps de génération côté Gemini, surtout pour gros tool calls / fichiers). `_read_all` et `generate_stream` bouclent `read1(65536)` / `iter_text()` jusqu'à terminal.
- **Downstream** : le proxy **reconstruit** un SSE OpenAI :
  - Non-tool : `role` chunk puis chaque `delta` tel quel
  - Avec tools : `StreamToolCallParser` (voir §5) émet `delta` immédiatement sans bufferiser toute la réponse.
- `PROBABLE` : le `reqid` devrait être incrémenté séquentiellement pour une même page ; ici random, mais Gemini l'accepte en anonyme.
- `UNKNOWN` : existe-t-il un champ `thought`/`reasoning` séparé dans le stream (pour `think=0`) ? Non observé, le thinking est inclus dans le texte.

**Robustesse** : `httpx.Timeout(request_timeout_sec, read=STALL_TIMEOUT)` (`gemini.py:49`). Si `httpx` absent, fallback non-streaming.

---

## 5. Tools / agent capabilities

| Question | Réponse | Confiance |
|---|---|---|
| Gemini Web a-t-il un champ natif `tools` / `functionDeclarations` ? | **Non observé**. Aucun index `inner` n'a porté de définition d'outil en trafic capturé. Le proxy injecte tout en **prompt texte**. | `PROBABLE` (absence d'évidence ≠ preuve) |
| Les tool calls sont-ils des champs structurés natifs dans la réponse ? | **Non**. Réponse = texte contenant ` ```tool_call` / ` ```function_call` + JSON. Pas de `functionCall` protobuf côté `wrb.fr` (contrairement à l'API officielle `generativelanguage`). | `CONFIRMED` pour le proxy, `PROBABLE` que le web n'a pas de format natif public |
| Comment fournir un tool result à Gemini Web ? | En **texte** : `[Tool result for {name}]: {content}` concaténé au prompt historique (`tools.py:299`). Aucun champ `functionResponse` natif utilisé côté `StreamGenerate`. Côté Google surface, `functionResponse` est traduit vers le même texte. | `CONFIRMED` simulé |
| Mécanismes internes Google (Assistant, Search, Code) | Utilisent probablement un autre RPC (`BardChatUi/data/assistant.lamda...` avec `inner[30]=[4]` etc.) non documenté. `inner[30]=[4]` pourrait activer un mode `tools` interne, mais non vérifié. | `UNKNOWN` |
| Streaming des tool calls | Texte streamé, `StreamToolCallParser` détecte ` ```tool_call` ... `{...}` ... ` ``` ` en temps réel, avec réparation `_repair_json` pour control chars et quotes. | `CONFIRMED` simulé |
| Validation | `validate_tool_arguments` + `generate_validated` repair loop 1 tour (`tool_validate_retry`) côté proxy, pas côté Gemini. | `CONFIRMED` proxy |
| Parallèle | Plusieurs blocs ` ```tool_call` dans **une même réponse** → `tool_calls[]` multiples. Exécutés séquentiellement par `agent.py`. | `CONFIRMED` simulé |

**Fallback actuel** : le système ` ```tool_call` est **le seul** et est assumé comme prototype. L'abstraction `generate_validated` / `StreamToolCallParser` permet de remplacer le format (`tool_call` vs `function_call`) sans changer l'orchestrateur (`server.py` utilise `parse_fn` paramétrable).

**Ce qui pourrait devenir natif** : si un champ `inner[XX] = [{name, description, parameters}]` était découvert (via diff HTML/JS ou capture navigateur avec DevTools → `__a` payload), il suffirait de modifier `_build_payload` pour y écrire les `tool_defs` au lieu du prompt, et d'ajouter un parser `inner[YY]` pour les calls retournés. Aucune evidence actuelle → marqué `UNKNOWN`.

---

## 6. Exemples minimaux

### Requête anonyme Flash

```http
POST /_/BardChatUi/data/assistant.lamda.BardFrontendService/StreamGenerate?bl=boq_assistant-bard-web-server_20260830.05_p0&hl=en&_reqid=123456&rt=c HTTP/1.1
Host: gemini.google.com
Content-Type: application/x-www-form-urlencoded
Origin: https://gemini.google.com
Referer: https://gemini.google.com/app
X-Same-Domain: 1
User-Agent: Mozilla/5.0

f.req=%5Bnull%2C%22%5B%5C%22Hello%5C%22%2C0%2Cnull%2Cnull%2Cnull%2Cnull%2C0%5D%22%5D
```

### Réponse (trim)

```
)]}'
[[null,"wrb.fr",null,"[null,null,null,null,[[null,[[\"Hello! How can I\"]]]]]"]]
[[null,"wrb.fr",null,"[null,null,null,null,[[null,[[\"Hello! How can I help you today?\"]]]]]"]]
[[null,"wrb.fr",null,"[null,null,{\"11\":\"Chat Title\",\"44\":1},null,null]"]]
```

Le proxy garde `"Hello! How can I help you today?"` (le plus long).

### Upload image

```http
POST https://content-push.googleapis.com/upload/ HTTP/1.1
Push-ID: feeds/mcudyrk2a4khkz
X-Tenant-Id: bard-storage
X-Client-Pctx: CgcSBWjK7pYx
X-Goog-Upload-Protocol: resumable
X-Goog-Upload-Command: start
...

→ X-Goog-Upload-URL: https://content-push.googleapis.com/upload/AA...

POST {upload_url} HTTP/1.1
X-Goog-Upload-Command: upload, finalize
...

→ /upload/bard/xxxx
```

Injecté : `inner[0] = ["Describe",0,None,[[null,null,"/upload/bard/xxxx"]],...]`

---

## 7. Limitations Gemini Web (pour l'API)

- **Pas de session** : chaque requête est indépendante, le contexte est rejoué en texte → coût prompt O(n²) sur longues conversations, risque de dépassement fenêtre.
- **Pas de tool natif** : tout le tool calling est prompt-engineering, sensible au wording, au jailbreak, et à la langue du modèle.
- **Anonyme** : pas de `Pro` réel, pas de mémoire persistante, pas de `conversationId` fiable.
- **Rate-limit** : Google throttles les IP Docker/NAT (note `README.md:211` host networking).
- **Fragilité `bl`/`wrb.fr`** : changement de build ou de layout `inner` casse le parsing sans erreur explicite (retour `content: null`).
- **Images** : upload via Scotty nécessite parfois un cookie valide, pas garanti en anonyme.
- **Streaming idle** : pause >60s tronque la réponse (limite `STALL_TIMEOUT`).

---

## 8. Niveaux de confiance — résumé

- `CONFIRMED` : endpoint, headers, `bl`, `reqid`, `f.req` double-JSON, `inner[0/17/41/45/59/79]`, Scotty, `wrb.fr` chunks, terminal `inner[2]["11"]["44"]`, tool injection texte, `StreamToolCallParser`, `SAPISIDHASH`, `stall` logic.
- `PROBABLE` : `hl`, `inner[1/2/6/7/10/11/18/27/30/53/61/68]` inconnus mais `61=[]` = pas de session, `30=[4]` = mode famille, `xsrf_token` seulement si auth.
- `UNKNOWN` : existence d'un champ natif tools, structure exacte `conversationId` retourné, indices non utilisés, raisonnement séparé, quota Pro, compression/encryption payload.

---

## 9. Références code

- `gemini.py:130` `fetch_latest_bl`, `56` `load_cookie`, `82` `make_sapisidhash`, `195` `_build_payload`, `228` `_get_url`, `105` `_is_terminal_line`, `247` `_extract_texts_from_line`, `360` `generate_stream`
- `tools.py:207` `messages_to_prompt`, `477` `google_contents_to_prompt`, `390` `parse_tool_calls`, `611` `StreamToolCallParser`, `810` `generate_validated`
- `server.py:76` `GeminiHandler`, `208` `_handle_chat`, `404` `_handle_responses`, `808` `_handle_anthropic`, `642` `_handle_google_generate`
- `multimodal.py:15` `_get_page_tokens`, `88` `upload_image`
- `config.py:1` `DEFAULT_CONFIG`
- `models.py:6` `MODELS`
