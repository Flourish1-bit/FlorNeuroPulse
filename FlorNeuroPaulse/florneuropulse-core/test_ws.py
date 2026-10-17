import websocket, json
ws = websocket.create_connection('ws://localhost:8000/ws/stream')
ws.send(json.dumps({'session_token':'test-001','audio_15d':[0.2]*15,'vision_5d':[0.5]*5}))
print(ws.recv())
ws.close()
