"""
WebSocket handlers for real-time communication
"""
from flask import request
from flask_socketio import emit

from .auth import is_authorized
from .socket_events import EVENT_TRANSLATION_UPDATE, LLM_RESPONSE_TYPES


def configure_websocket_handlers(socketio, state_manager):
    """Configure WebSocket event handlers"""

    @socketio.on('connect')
    def handle_websocket_connect(auth):
        # Reject handshakes that don't carry the per-session token (issue #210).
        # The SPA passes it via io({ auth: { token } }); a cross-origin page can
        # neither read nor guess it. Returning False refuses the connection.
        token = auth.get('token') if isinstance(auth, dict) else None
        if not is_authorized(token):
            print(f'🚫 WebSocket handshake rejected (bad token): {request.sid}')
            return False
        print(f'🔌 WebSocket client connected: {request.sid}')
        emit('connected', {'message': 'Connected to translation server via WebSocket'})

    @socketio.on('disconnect')
    def handle_websocket_disconnect():
        print(f'🔌 WebSocket client disconnected: {request.sid}')


def emit_update(socketio, translation_id, data_to_emit, state_manager):
    """
    Emit WebSocket update for translation progress.

    Stats are NOT auto-attached. Callers that need to send progress stats must
    include them explicitly via `data_to_emit['stats']`. Auto-attaching stats
    on every log/status emit used to create races: a log emit on the main loop
    would read the state snapshot at log time and emit it later, possibly
    overtaking a fresher stats emit on the wire and rolling the progress bar
    backward. Now only the dedicated stats callbacks touch the progress bar.

    Args:
        socketio: SocketIO instance
        translation_id (str): Translation job ID
        data_to_emit (dict): Data to send (must include 'stats' to push progress)
        state_manager: Translation state manager instance
    """
    if not state_manager.exists(translation_id):
        return

    data_to_emit['translation_id'] = translation_id
    try:
        # Store last translation for UI restoration after browser refresh.
        # Both the translate path (`llm_response`) and the refine path
        # (`refinement_response`) produce displayable LLM output — keep the
        # preview in sync for either.
        log_entry = data_to_emit.get('log_entry')
        if (log_entry and log_entry.get('type') in LLM_RESPONSE_TYPES and
            log_entry.get('data', {}).get('response')):
            state_manager.set_translation_field(
                translation_id, 'last_translation', log_entry['data']['response']
            )

        socketio.emit(EVENT_TRANSLATION_UPDATE, data_to_emit, namespace='/')
    except Exception as e:
        print(f"WebSocket emission error for {translation_id}: {e}")