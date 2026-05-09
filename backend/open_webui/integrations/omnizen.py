"""
Omnizen integration — per-user API key passthrough.

Context
-------
This OpenWebUI fork is deployed at https://chat.omnizen.ai. It talks to
the Omnizen LiteLLM proxy at https://api.omnizen.ai/v1 (OpenAI-shape).

For billing to work correctly, every chat request must hit Omnizen with
the *user's own* Omnizen API key — not a single shared service key —
so that spend lands against the right account in Omnizen's
/dashboard/analytics. OpenWebUI ships with a built-in concept of a
per-user API key (stored in the `api_key` table, managed via the
existing /account UI), and we simply reuse that field as the user's
Omnizen API key.

This module exposes one function — `resolve_user_api_key` — which is
called from `routers/openai.py::generate_chat_completion` right after
the global key is looked up. If the user has stored their own Omnizen
key in their OpenWebUI account, we substitute it. Otherwise we fall
through to the global key (preserves admin-test / preview behaviour).

Failure mode is fail-open: any exception during key lookup logs and
returns the original global key, so a transient DB hiccup never blocks
a chat request.
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


async def resolve_user_api_key(user: Any, fallback_key: str) -> str:
    """
    Return the API key to use for an upstream Omnizen call.

    Looks up `user`'s stored API key via Users.get_user_api_key_by_id.
    Falls back to `fallback_key` (the global OPENAI_API_KEYS[idx]) on
    any miss or error.
    """
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
