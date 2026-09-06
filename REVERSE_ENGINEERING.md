# REVERSE_ENGINEERING — Audit Gemini2Api

> **Phase 1 — Audit complet sans modification de code.**
> Date: 2026-05-13 — commit `4ba45eb` (post-keepalive) + audit lecture seule.
> Repo: https://github.com/mastersoon1754-ui/Gemini2Api

---

## 1. Vue d'ensemble

Gemini2Api est un proxy **OpenAI-compatible** qui expose un serveur HTTP pur Python (`http.server` + `ThreadingMixIn`) et **utilise Gemini Web comme seul backend**. Aucun appel à l'API officielle `generativelanguage.googleapis.com` n'est effectué pour la génération texte : tout passe par `https://gemini.google.com/_/BardChatUi/data/assistant.lamda.BardFrontendService/StreamGenerate`.

```
Client OpenAI / Codex / Gemini CLI / Anthropic
        ↓  POST /v1/chat/completions | /v1/responses | /v1/messages | /v1beta/models/...
GeminiHandler (gemini_web2api/server.py:76)
        ↓  messages_to_prompt / google_contents_to_prompt / _anthropic_to_internal
        ↓  _upload_images → multimodal.upload_image (Scotty)
        ↓  gemini.generate / generate_stream
Gemini Web StreamGenerate (gemini_web2api/gemini.py:334)
        ↓  wrb.fr parsing → extract_response_text / StreamToolCallParser
Client ← SSE / JSON OpenAI
```

Deux implémentations coexistent :
- **Nouvelle (modulaire)** : `gemini_web2api/` (utilisée par `Dockerfile:6` → `python -m gemini_web2api`) — c'est la référence.
- **Legacy monolithe** : `gemini_web2api.py` (1493 lignes, copy du système modulaire pour `python gemini_web2api.py` direct). Audit ci-dessous sur la version modulaire, le monolithe est identique à `±keepalive`.

---

## 2. Réception des requêtes OpenAI

**Fichier : `gemini_web2api/server.py:76`**

- `ThreadedServer(ThreadingMixIn, HTTPServer)` → un thread par requête, `daemon_threads=True`.
- `do_GET` / `do_POST` → auth d'abord, puis dispatch :
  - `GET /v1/models` → `MODELS` (`models.py:6`)
  - `GET /v1beta/models` → même liste format Google
  - `GET /` et `GET /health` (keepalive) → `{status, version, models}`
  - `POST /v1/chat/completions` → `_handle_chat` (`server.py:208`)
  - `POST /v1/responses` → `_handle_responses` (`server.py:404`)
  - `POST /v1/messages` → `_handle_anthropic` (`server.py:808`)
  - `POST /v1beta/models/{model}:generateContent|streamGenerateContent` → `_handle_google_generate` (`server.py:642`)
- `_authorized` (`server.py:129`) :
  - si `CONFIG.api_keys == []` → open
  - sinon check `Authorization: Bearer <key>`, `x-api-key`, `x-goog-api-key`, ou `?key=` — comparaison string exacte, pas de hash.
- `_read_request_body` (`server.py:103`) : supporte `Transfer-Encoding: chunked` (décodage manuel hex size + trailers) et `Content-Length`. Pas de limite de taille explicite → DoS potentiel si client envoie gros body.
- `_parse_body` : `json.loads`, return `None` → 400.
- CORS : `Access-Control-Allow-Origin: *`, `OPTIONS` → 204.

**Robustesse / manque :**
- Pas de `request_id` interne, pas de logs structurés (seulement `log()` vers stderr si `log_requests`).
- Pas de rate-limit, pas de validation JSON Schema côté entrée (seulement tool args validés plus tard).
- Erreurs upstream → `502 upstream error`, pas de code d'erreur fin.

---

## 3. Conversion messages → prompt

**Fichier : `gemini_web2api/tools.py:207`**

### OpenAI (`messages_to_prompt`)
- `tools` + `tool_choice` : si non `none`, `extract_openai_tool_defs` normalise `tools` → `[{name, description, parameters}]` (minifie `$schema/title/additionalProperties` via `_minify_schema`).
- Budget : `tool_max_tools` (défaut 96) et `tool_max_prompt_chars` (120k) dans `config.py:22`. Trie par priorité : `forced` (tool_choice dict) → `CORE_TOOLS` (bash/edit/read...) → taille JSON. Garde les plus petits, drop le reste en loguant. → **Simulation** : limitation artificielle, pas native.
- Prompt injecté en tête :
  ```
  # Tool Use
  You can call the following tools. Call format:
  ```tool_call
  {"name": "func_name", "arguments": {...}}
  ```
  ...
  Available tools:
  [{"name":"...","description":"...","parameters":{...}}]
  + constraint tool_choice (required/none/specific)
  ```
  Format **fictif**, jamais envoyé comme champ natif Gemini.
- Ensuite itère `messages` :
  - `content` liste → concatène `text`/`input_text`, pour chaque image → `_image_from_part` → ajoute à `images[]` et insère `[Image attached]` dans le texte.
  - `role` → préfixe :
    - `system` → `[System instruction]: {content}`
    - `assistant` avec `tool_calls` → `[Assistant]: {content}\n```tool_call...```*`
    - `tool` → `[Tool result for {name}]: {content}`
    - sinon `content` brut
  - Jointure `\n\n`.

### Google native (`google_contents_to_prompt` `tools.py:477`)
- Même principe mais `systemInstruction` + `tools[].functionDeclarations[]` → `build_tool_prompt` avec format ` ```function_call` + `{"name","args"}`.
- `contents[].parts[]` : `text` | `inlineData` (→ image) | `functionCall` (→ ` ```function_call`) | `functionResponse` (→ `[Tool result for ...]`).

### Anthropic (`server.py:808` `_anthropic_to_internal`)
- Convertit le format `content: [{type:text|tool_use|tool_result|image}]` vers le format OpenAI interne, puis appelle `messages_to_prompt`. `tool_choice` map : `none→none`, `any→required`, `tool→{function:{name}}`.

**Point simulé majeur** : le multi-turn conversationnel n'est pas une session id côté Gemini Web. Chaque requête reconstruit un **prompt géant** contenant tout l'historique textuel. Aucun `conversation_id` / `session_id` n'est persisté ni renvoyé à Gemini Web (voir §6). Limite `MAX_IMAGE_B64_SIZE` et `PROMPT_MAX_BYTES` implicite.

---

## 4. Appel Gemini Web

**Fichier : `gemini_web2api/gemini.py:195`**

### Payload
- `inner = [None]*102` — tableau positionnel, seuls ~10 index significatifs :
  | Index | Valeur | Source |
  |---|---|---|
  | 0 | `[prompt, 0, None, refs, None, None, 0]` | prompt + file_refs (image) |
  | 1 | `["en"]` | langue |
  | 2 | `["","","", None...,""]` | placeholder |
  | 6 | `[0]` |  |
  | 7,10 | `1` |  |
  | 17 | `[[think_mode]]` | `think` de `models.py` (0-4) |
  | 30 | `[4]` |  |
  | 41,45 | persistance (`[2]`/`None` ou `[1]`/`1` si `temporary_chats`) | `config.py:temporary_chats` |
  | 59 | `uuid4()` |  |
  | 79 | `model_id` | `MODE_CATEGORY` (1 FAST, 2 THINKING, 3 PRO, 4 AUTO, 5 DYNAMIC, 6 LITE) |
  | 31,80, etc. | `extra_fields` | ex: `{"31":2,"80":3}` pour pro-enhanced |
- `outer = [None, json.dumps(inner)]` → `f.req = json.dumps(outer)` → POST `application/x-www-form-urlencoded` avec `at=xsrf_token` si présent.
- **Headers** `_build_headers` (`gemini.py:166`) :
  ```
  Content-Type: application/x-www-form-urlencoded
  Origin: https://gemini.google.com
  Referer: https://gemini.google.com[/u/{auth_user}]/app
  X-Same-Domain: 1
  User-Agent: Mozilla/5.0...
  Cookie: <cookie_str>
  Authorization: SAPISIDHASH <ts>_<sha1(ts + sapisid + "https://gemini.google.com")>
  X-Goog-AuthUser: <auth_user> si non vide
  ```
- **URL** `_get_url` (`gemini.py:228`) :
  ```
  https://gemini.google.com[/u/{auth_user}]/_/BardChatUi/data/assistant.lamda.BardFrontendService/StreamGenerate?bl={gemini_bl}&hl=en&_reqid={rand%1e6}&rt=c
  ```
  - `bl` : token build `boq_assistant-bard-web-server_YYYYMMDD.p0` — récupéré via `fetch_latest_bl()` qui scrape `gemini.google.com/app` avec regex `boq_assistant-bard-web-server_\d+\.\d+_p\d+`. Auto-refresh si 405.
  - `_reqid` : random `int(time.time())%1e6`, incrémenté côté web normalement mais ici random à chaque tentative.

### Endpoints
- Un seul endpoint pour tout : `StreamGenerate`. Pas de `GenerateContent`, pas de `CreateConversation`.
- Les images utilisent un **autre** endpoint : `content-push.googleapis.com/upload/` (voir §10).

### Retry / erreurs
- `generate` boucle sur `CONFIG.retry_attempts+1` (défaut 3), sleep `retry_delay_sec` (2s). Si 405 → `update_bl_if_needed()` puis retry.
- `generate_stream` même, mais via `httpx` si dispo, sinon fallback `generate`.

---

## 5. Cookies / Authentification

**Fichier : `gemini_web2api/gemini.py:56`**

- `cookie_file` (`config.py:cookie_file`) — fichier raw `Cookie` header ou JSON `{cookie, sapisid}`. Chargé avec cache `mtime`.
- Extraction `SAPISID` → `make_sapisidhash` → header `Authorization: SAPISIDHASH`. Requis pour upload image et compte pro, **optionnel pour génération anonyme** (le proxy fonctionne sans cookie pour Flash).
- `auth_user` : préfixe `/u/{n}` pour comptes Google multiples. Envoyé aussi en `X-Goog-AuthUser`.
- `xsrf_token` (`SNlM0e` du DOM Gemini) → champ `at` POST. Rarement nécessaire en anonyme.
- **Partage** : `_cookie_cache` + `_httpx_client` globaux → tous les threads partagent le même cookie. Pas de fuite entre utilisateurs mais pas d'isolation non plus (un seul compte sert tout).
- **Confidentialité** : le cookie n'est jamais renvoyé au client, seulement utilisé upstream.

---

## 6. Sessions / Conversations

**État actuel : stateless simulé.**

- Aucun `conversationId`, `sessionId`, `chatId` n'est envoyé à Gemini Web.
- Chaque requête est **indépendante** : le client doit renvoyer tout l'historique `messages` ; le serveur le concatène dans un seul `inner[0][0]` (prompt). Gemini Web voit une seule entrée utilisateur géante, pas un thread.
- Flags de persistance :
  - `temporary_chats=False` → `inner[41]=[2]` (persistant côté compte si cookie)
  - `temporary_chats=True` → `inner[41]=[1], inner[45]=1` (conversation éphémère, non sauvegardée dans l'historique Gemini Web)
  - Mais comme aucun ID n'est réutilisé, même en mode persistant l'historique web reste vide/fragmenté (un chat par requête).
- **Limite** : le prompt grossit à chaque tour → risque de troncature invisible côté Gemini (pas de `max_tokens` respecté, estimation `len(prompt)//4` pour `usage` seulement).
- **Concurrency** : pas de verrou, mais pas de mélange de sessions non plus car stateless. Le problème serait inverse : **perte de contexte** si le client n'envoie pas tout l'historique.

> **Aucun mécanisme natif de reprise de conversation n'a été observé dans le trafic actuel.** À reverser en Phase 2.

---

## 7. StreamGenerate en détail

- Méthode : `POST` `application/x-www-form-urlencoded`, body `f.req` + `at`.
- Réponse : `chunked transfer-encoding` **jamais fermée** par le serveur (pas de `0\r\n\r\n`). Le client doit détecter la fin.
- Heuristique actuelle (`gemini.py:96`) :
  - `STALL_TIMEOUT = 60s` — si la socket est idle 60s, on considère la réponse finie.
  - `_is_terminal_line` : parse chaque ligne `wrb.fr`, `arr[0][2]` → `json.loads(inner)` → si `inner[2]` contient `"11"` et `"44"` → terminal (titre + flag done).
  - `_has_terminal_chunk` scanne le buffer.
  - `_read_all` (urllib) et `generate_stream` (httpx) bouclent `read1(65536)` / `iter_text()` jusqu'à terminal ou timeout.

---

## 8. Décodage réponses `wrb.fr`

**Fichier : `gemini_web2api/gemini.py:247`**

- Format : une ligne par `wrb.fr` (en réalité `)]}'\n` préfixe XSSI, puis `[[null,"wrb.fr",null,[...]], ...]`).
- Actuel ne garde que les lignes contenant `"wrb.fr"` et `len>=200`.
- `arr = json.loads(line)` → `inner_str = arr[0][2]` → `inner = json.loads(inner_str)`.
- Texte : `inner[4]` est une liste de `part` → pour chaque `part[1]` qui est `list[str]`, on collecte les strings. Le plus long `t` (par `len`) est gardé comme `last_text` (`extract_response_text:279`). D'autres chunks plus courts sont des préfixes intermédiaires (streaming).
- `BardErrorInfo` → `RuntimeError` si présent (ex: `BardErrorInfo [116]`).
- `clean_text` (`gemini.py:238`) strip :
  - ` ```python|javascript|text?code_reference&code_event_index=\d+ ... ``` ` (artefacts d'exécution)
  - `http://googleusercontent.com/card_content/...`
  - `strip()` si demandé.

**Fragilités identifiées :**
- Indices `inner[4]`, `inner[2]["11"]["44"]` **hardcodés** et non documentés officiellement → cassera si Gemini change le layout.
- Pas de parsing protobuf, regex simple `"wrb.fr"`.
- `clean_text` peut supprimer du contenu légitime contenant ces patterns.

---

## 9. Streaming

**Fichiers : `gemini_web2api/gemini.py:360` + `server.py:272`**

- **Upstream** : `generate_stream` via `httpx.Client.stream(POST)` avec `httpx.Timeout(request_timeout_sec, read=STALL_TIMEOUT)`. `iter_text()` → `buf` accumulé, split `\n`, pour chaque ligne non-terminale → `_extract_texts_from_line` → détection delta `t[len(emitted):]` → `yield clean_text(delta)`. `ReadTimeout` → fin normale (pas d'erreur).
- Fallback sans `httpx` → `generate` puis `yield text` unique (buffered).
- **Downstream** (SSE client) :
  - `server.py:272` non-stream avec tools → `generate_validated` (buffered).
  - `server.py:275` stream sans tools → envoie `role` chunk puis chaque `delta` tel quel, puis `stop`, puis `include_usage` si demandé, puis `[DONE]`.
  - `server.py:315` stream **avec tools** → utilise `StreamToolCallParser` pour **ne pas bufferiser** toute la réponse : `feed(delta)` émet immédiatement `("text",...)` ou `("tool_start",...)` → converti en `tool_calls` chunks OpenAI (`_openai_tool_event_chunks`).
  - De même pour Google (`server.py:698`) et Anthropic (`server.py:913`) avec mapping SSE différent (`event: message_start`, `content_block_delta`, etc.).
- **Limites** : `emitted_raw_text` vérifie `t.startswith(prev)` sinon `RuntimeError` — si Gemini renvoie une reformulation, le stream casse.

---

## 10. Images / Fichiers

**Fichier : `gemini_web2api/multimodal.py:88`**

- Extraction côté API : `tools._image_from_part` supporte `image_url` (OpenAI), `input_image` (Responses), `inlineData` (Google), `data:` URLs. Retourne `(bytes|url_str, mime)`.
- `_upload_images` (`server.py:53`) → pour chaque `bytes`, `detect_image_mime`, puis `multimodal.upload_image`.
- **Upload Scotty** (`multimodal.py:88`) :
  1. `GET https://gemini.google.com/app` avec cookies → scrape `push_id` (`"qKIAYe"`), `pctx` (`"Ylro7b"`), `at` (`"thykhd"`) via regex, cache 600s (`_page_tokens_cache`). Fallback hardcodé si fail.
  2. `POST https://content-push.googleapis.com/upload/` avec headers `Push-ID`, `X-Tenant-Id: bard-storage`, `X-Client-Pctx`, `X-Goog-Upload-Header-Content-Length/Type`, `X-Goog-Upload-Protocol: resumable`, `X-Goog-Upload-Command: start` → récupère `X-Goog-Upload-URL`.
  3. `POST {upload_url}` avec `X-Goog-Upload-Command: upload, finalize`, `X-Goog-Upload-Offset: 0`, body `image_bytes` → retourne `file_ref` (chemin `/...`).
- `file_ref` est injecté dans `inner[0][3] = [[None,None,ref]]`.
- **Échecs** : si upload 502 ou pas de cookie → l'image est ignorée ou 502 remonté. Compression `MAX_IMAGE_B64_SIZE` 50k avec PIL si dispo, sinon troncature.
- URL distante `https://` → `fetch_image_bytes` télécharge côté serveur (proxy), timeout 30s, pas de limite de taille → risque SSRF/boucle.

---

## 11. Tool calling actuel

**Fichiers : `gemini_web2api/tools.py:207` + `server.py:366`**

- **Aucun champ natif Gemini** n'est utilisé. Tout est **prompt injection** :
  - Le serveur injecte la définition des tools en JSON dans le prompt (voir §3).
  - Le modèle est instruit de répondre avec ` ```tool_call\n{"name":"...","arguments":{...}}\n``` `.
  - Pareil côté Google avec ` ```function_call\n{"name":"...","args":{...}}\n``` `.
- **Parsing** `parse_tool_calls` (`tools.py:390`) :
  - `re.finditer(r'```tool_call', text)` → `len(rest.lstrip)` → check `{` → `_scan_json_object` (brace matching string-aware) → `json.loads` avec `_repair_json` fallback (répare quotes non échappées, `\n`, `\r`, contrôles).
  - Success → `tool_calls[]` avec `id=call_<8hex>`, sinon garde le texte brut (pas de drop silencieux).
- **Validation** `validate_tool_arguments` (`tools.py:71`) : vérifie `type`, `enum`, `required`, `properties`, `items` (récursif, lenient sur inconnus).
- **Boucle de réparation** `generate_validated` (`tools.py:810`) :
  - 1er appel `generate_fn(prompt)` → parse → valide → si invalide et `tool_validate_retry>0` (défaut 1) → fabrique `repair_prompt = original + "[System correction]: ... invalid arguments ..." + "Previous answer: ..." + "Re-output corrected ..."` → 2e appel → re-parse → ne garde que les calls valides, drop les autres avec log.
- **Streaming** `StreamToolCallParser` (`tools.py:611`) :
  - `OPENERS = ("```tool_call","```function_call")`, `_TAIL` pour opener split.
  - `feed(chunk)` → `_scan_text` (cherche opener, émet texte safe hors tail) → une fois dans `body` → `_drain_body` (depth/in_str/esc tracking) → à `depth==0` → `_emit_tool` (json.loads + repair + auto-close si tronqué) → events `tool_start/tool_args/tool_end`.
  - `finish()` auto-close `depth` avec `}` et `"` si tronqué.
  - Utilisé pour OpenAI/Google/Anthropic en streaming pour émettre `tool_calls` deltas immédiatement.
- **Limites simulées** : tout est textuel, le modèle peut refuser, halluciner un nom, émettre du JSON invalide. La validation est côté proxy, pas côté Gemini.

---

## 12. `tool_executor_url` (agent loop côté serveur)

**Fichier : `gemini_web2api/agent.py:80`**

- Si la requête contient `tool_executor_url` (string URL) **et** des `tools`, le serveur ne renvoie pas les `tool_calls` au client. Il devient **orchestrateur** :
  ```
  history = list(messages)
  for turn in 0..max_agent_turns-1 (défaut 8):
      prompt = messages_to_prompt(history, tools)
      text, calls = generate_validated(prompt, tool_defs, ...)
      if calls: 
         history.append(assistant+tool_calls)
         for call in calls:
            result = POST {tool_executor_url} {call_id,name,arguments(dict)}
            steps.append({name,arguments,result})
            history.append({role:tool, tool_call_id, name, content:result})
         continue # next turn
      if tool_choice==required and !nudged: inject "[System: You MUST call a tool...]" et continue
      else return (text, steps)
  ```
- `execute_tool_call` (`agent.py:37`) : POST JSON, timeout `agent_tool_timeout_sec` (30s), respecte `proxy`, parse réponse `result|output|content|response` ou texte brut.
- Réponse HTTP : si `stream` → SSE avec `content` final seulement (pas de streaming des steps), sinon JSON avec `choices[0].message.content` + champ **non-standard** `agent_tool_calls: steps`.
- **Parallèle** : les `calls` multiples d'un même tour sont exécutés **séquentiellement** (boucle `for`), pas en parallèle.
- **Points simulés** : tout le loop est côté proxy, pas côté Gemini Web. Pas de cancellation, pas de timeout global, pas de détection de boucle infinie au-delà de `max_agent_turns`.

---

## 13. Réinjection des tool results

**Fichier : `tools.py:299`**

- Dans `messages_to_prompt`, chaque `role=="tool"` devient ligne :
  ```
  [Tool result for {name}]: {content}
  ```
  Quelque soit `tool_call_id`, seul `name` est conservé.
- Côté Google (`tools.py:534`) : `functionResponse` → même format.
- Côté Anthropic (`server.py:822`) : `tool_result` → converti en interne `role:tool` → même.
- Lors d'un nouvel appel Gemini, ce texte est simplement concaténé au prompt géant. Gemini Web n'a **aucun champ structuré** pour les tool results ; il les voit comme du texte utilisateur.
- Aucune distinction entre succès/échec, pas de `is_error` natif (l'Anthropic `is_error` n'est pas propagé).

---

## 14. Tous les points simulés (résumé)

| Domaine | Simulé | Natif observé | Confiance |
|---|---|---|---|
| **Génération texte** | Non | `StreamGenerate` endpoint réel, `wrb.fr` chunks | **Confirmé** |
| **Choix modèle** | Partiellement | `inner[79]=MODE_CATEGORY` + `inner[17]=think` confirmés via JS source `028-*.js` | **Confirmé** |
| **Auth** | Non | SAPISIDHASH, Cookie, bl, xsrf optionnels | **Confirmé** |
| **Upload image** | Non | Scotty `content-push.googleapis.com` | **Confirmé** |
| **Conversation / multi-turn** | **Oui** | Prompt concatenation, pas de session id, `temporary_chats` seul flag | **Simulé** |
| **Streaming** | Partiellement | SSE downstream reconstruit à partir de chunks wrb.fr + idle timeout | **Simulé downstream, natif upstream** |
| **Usage tokens** | **Oui** | `len(prompt)//4` estimation | **Simulé** |
| **Tool calling** | **Oui** | ` ```tool_call` / ` ```function_call` texte, injection JSON dans prompt, parsing + validation côté proxy | **Simulé** |
| **tool_choice** | **Oui** | Instruction texte `IMPORTANT: You MUST...` | **Simulé** |
| **Tool result** | **Oui** | `[Tool result for ...]:` texte | **Simulé** |
| **Agent loop** | **Oui** | Boucle côté proxy via `tool_executor_url`, pas côté Gemini | **Simulé** |
| **Parallèle tools** | **Oui** | Plusieurs ` ```tool_call` blocs dans un même tour, séquentiel côté executor | **Simulé** |
| **JSON validation** | **Oui** | `validate_tool_arguments` + `_repair_json` + `generate_validated` | **Proxy** |
| **Sessions concurrentes** | **Oui** | Aucun état global conversationnel, mais cache cookie global → pas d'isolation par utilisateur | **Simulé / fragile** |
| **Keepalive** | **Oui** | Thread self-ping `keepalive.py:1` | **Ajout récent** |
| **Anthropic/Google surfaces** | **Oui** | Traduction vers prompt OpenAI puis reformatage sortie | **Simulé** |

**Fragilités / tech debt :**
- Indices payload `inner[79]` etc. hardcodés, pas de protobuf.
- Regex `boq_assistant-bard-web-server` pour `bl` cassera si renommé.
- `STALL_TIMEOUT 60s` arbitraire ; pause >60s → troncature.
- `_cookie_cache` global + `ThreadingMixIn` → pas de thread-safety formelle (GIL masque mais pas garanti).
- `Transfer-Encoding: chunked` manuel, pas de limite taille, pas de `Content-Length` max.
- `_repair_json` corrige agressivement les quotes — risque de faux positifs.
- Pas de `request_id` / logs structurés, pas de `traceId` par conversation.
- `agent.py` séquentiel, pas de `asyncio.gather`, pas de cancellation.

---

## 15. Tests existants

**Fichiers : `tests/test_modular_sync.py:1` (620 lignes) + `tests/test_tool_calling.py:1` (486 lignes) — 53 tests, tous passent.**

- **test_modular_sync.py** :
  - `PayloadPersistenceTests` : `temporary_chats` flags, image refs `inner[0][3]`.
  - `MessageParsingTests` : `data:image/png;base64` vs `input_image` vs `inlineData`, malformed base64.
  - `StreamingEndpointTests` : SSE `/v1/chat/completions` stream role→content→[DONE], chunked body, image upload mock, Google stream SSE, Responses stream `response.created → completed` avec `sequence_number`, function_call stream.
  - `AgentLoopTests` : `run_agent_loop` avec webhook minimal `_ExecutorHandler`, `max_agent_turns`, `tool_choice required`, end-to-end `/v1/chat/completions` avec `tool_executor_url`.

- **test_tool_calling.py** :
  - `StreamParserTests` : mixed text+tool, parallel, opener split, truncated JSON autoclosed, non-block opener, function_call, control chars.
  - `SchemaAndValidationTests` : `_minify_schema`, `validate_tool_arguments` (required/type/enum/nested), parse parallel, Google normalized shape.
  - `RepairLoopTests` : invalid→repair (2 calls), valid no repair, unrepairable dropped, no tools passthrough.
  - `LargeToolsetTests` : 80 tools >35k old cap, budget 10, Google defs minify.
  - `OpenAIStreamingConformanceTests` : streaming deltas tool_start/tool_args, include_usage, nonstream validated, rejects invalid.
  - `GoogleSurfaceTests` : multiple functionCalls, streaming functionCall.
  - `AnthropicSurfaceTests` : tool_use, tool_result roundtrip, streaming blocks `input_json_delta`, `tool_choice any→required`.

**Couverture** : pas de test réseau réel vers Gemini Web (tout mocké), pas de test `wrb.fr` réel, pas de test cookie/auth, pas de test concurrency, pas de test timeout/stall, pas de test proxy, pas de test rate-limit.

---

## 16. Prochaines étapes (Phase 2+)

- **Phase 2** à faire : capturer le trafic réel navigateur (`gemini.google.com/app` → `StreamGenerate`, `content-push.googleapis.com`, `WIZ_global_data`) pour remplir `docs/GEMINI_WEB_PROTOCOL.md` avec niveaux `confirmed|probable|unknown`.
- Vérifier en particulier : existe-t-il un champ natif `tools`/`functionDeclarations` dans `inner` (actuellement non utilisé) ? Existe-t-il un `conversationId` réutilisable ?

---

## 17. Fichiers clés (référence rapide)

- `gemini_web2api/gemini.py:195` — `_build_payload`, `_get_url`, `generate`, `generate_stream`, `extract_response_text`, `STALL_TIMEOUT`
- `gemini_web2api/tools.py:207` — `messages_to_prompt`, `google_contents_to_prompt`, `parse_tool_calls`, `StreamToolCallParser`, `generate_validated`
- `gemini_web2api/server.py:76` — `GeminiHandler`, `_handle_chat|responses|anthropic|google_generate`
- `gemini_web2api/agent.py:80` — `run_agent_loop`, `execute_tool_call`
- `gemini_web2api/multimodal.py:88` — `upload_image`, `fetch_image_bytes`
- `gemini_web2api/config.py:1` — `DEFAULT_CONFIG`, `CONFIG`
- `gemini_web2api/models.py:6` — `MODELS`, `resolve_model`
- `gemini_web2api/keepalive.py:1` — self-ping Render
- `gemini_web2api.py` — monolithe (miroir)
- `tests/test_modular_sync.py`, `tests/test_tool_calling.py` — 53 tests

