"""
Omnizen integration — per-user API key passthrough.

Context
-------
This OpenWebUI fork is deployed at https://chat.omnizen.ai. It talks to
the Omnizen LiteLLM proxy at https://api.omnizen.ai/v1 (OpenAI-shape).

For billing to work correctly, every chat request must hit Omnizen with
the *user's own* Omnizen API key — not a single shared service key —
so that spend lands against the right account in Omnizen's
/dashboard/analytics. We resolve the right key in two ways, preferred
first:

1. **Forward-auth header** — Caddy at chat.omnizen.ai calls
   omnizen-ai-web-1's `/api/internal/openwebui-auth` to validate the
   visitor's Clerk session. That endpoint also looks up the user's
   ``users.litellm_key`` from Postgres and returns it as
   ``X-Omnizen-Api-Key``. Caddy ``copy_headers`` injects the header
   into every proxied request to OpenWebUI. When this header is
   present we use it directly — no manual paste required.

2. **Local OpenWebUI ``api_key`` table** (legacy) — for users who
   landed on the chat without going through the forward_auth flow,
   or for setups where the upstream Omnizen instance hasn't been
   patched yet, we still honour a key the user pasted in
   ``/account → API Keys``.

Failure mode is fail-open: any exception during lookup logs and
returns the global ``fallback_key``, so a transient DB / header
hiccup never blocks a chat request.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

log = logging.getLogger(__name__)

# Heuristic: an Omnizen key starts with `om_` (live) or `om_test_`. We
# don't enforce this — the user may have stored a raw OpenAI key for
# testing and we should pass it through unchanged. The prefix check is
# only used to decide whether to log a warning when a stored key
# doesn't *look* like an Omnizen key.
_OMNIZEN_KEY_PREFIXES = ('om_', 'sk-om_')

# Header set by Caddy (forwarded from omnizen-ai's
# /api/internal/openwebui-auth). Always lowercase here because Starlette
# normalises header names.
_OMNIZEN_API_KEY_HEADER = 'x-omnizen-api-key'


async def resolve_user_api_key(
    user: Any,
    fallback_key: str,
    request: Optional[Any] = None,
) -> str:
    """
    Return the API key to use for an upstream Omnizen call.

    Resolution order:
      1. ``X-Omnizen-Api-Key`` header on the inbound request (set by
         Caddy ``forward_auth`` from omnizen-ai's
         ``/api/internal/openwebui-auth`` after Clerk validation).
      2. ``Users.get_user_api_key_by_id(user.id)`` — legacy paste-it
         flow.
      3. ``fallback_key`` — the configured global ``OPENAI_API_KEY``.
    """
    # 1. Forward-auth header (preferred — fully zero-paste UX)
    if request is not None:
        try:
            headers = getattr(request, 'headers', None)
            if headers is not None:
                hdr = headers.get(_OMNIZEN_API_KEY_HEADER)
                if hdr:
                    return hdr
        except Exception as e:  # pragma: no cover — defensive
            log.warning('omnizen: header lookup raised %s', e)

    # 2. Local OpenWebUI api_key table (legacy paste-flow)
    if user is None or not getattr(user, 'id', None):
        return fallback_key

    try:
        # Imported lazily to avoid a circular import at module load
        # (open_webui.models imports config, which transitively imports
        # routers in some deployments).
        from open_webui.models.users import Users

        stored: Optional[str] = await Users.get_user_api_key_by_id(user.id)
    except Exception as e:  # pragma: no cover — defensive
        log.warning(
            'omnizen: failed to look up per-user API key for %s: %s — '
            'falling back to global key',
            getattr(user, 'id', '<unknown>'),
            e,
        )
        return fallback_key

    if not stored:
        return fallback_key

    if not stored.startswith(_OMNIZEN_KEY_PREFIXES):
        log.debug(
            'omnizen: user %s has a stored API key that does not match '
            'the Omnizen prefix; passing through unchanged',
            user.id,
        )

    return stored
